#!/usr/bin/env python3
"""Closed-loop validation for structured LLM map-maintenance policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from activemap.evaluation.episode_utility import (
    score_episode_profiles,
    utility_protocol,
)

AVAILABLE_METHODS = {
    "oracle",
    "greedy",
    "edit_conditioned_selector",
    "edit_conditioned_pair_closure",
    "generic_selector",
    "uncertainty",
    "random",
    "acquire_all",
    "cheapest",
    "quality_first",
    "mapex",
    "greedy_utility",
    "qwen3_4b_sft",
    "forced_tools",
    "qwen3_4b_sft_tools_no_belief",
    "qwen3_4b_sft_tool_to_belief",
    "qwen3_4b_sft_calibrated_tools_no_belief",
    "qwen3_4b_sft_calibrated_tool_to_belief",
    "edit_conditioned_proactive_tools",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if path.is_file():
        return {"path": str(path.resolve()), "sha256": _sha256(path)}
    if not path.is_dir():
        raise FileNotFoundError(path)
    inference_names = {
        "adapter_config.json",
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    files = sorted(
        item
        for item in path.iterdir()
        if item.is_file()
        and (item.name in inference_names or item.name.startswith("adapter_model."))
    )
    if not files:
        raise ValueError(f"directory has no inference artifacts to fingerprint: {path}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item.relative_to(path)).encode())
        digest.update(_sha256(item).encode())
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "file_count": len(files),
    }


def _parse_methods(value: str) -> set[str]:
    methods = {item.strip() for item in value.split(",") if item.strip()}
    if not methods:
        raise ValueError("at least one rollout method is required")
    unknown = sorted(methods - AVAILABLE_METHODS)
    if unknown:
        raise ValueError(f"unknown rollout methods: {unknown}")
    return methods


def _select_samples(
    samples: list[Any],
    *,
    limit: int | None,
    order: str,
    seed: int,
) -> list[Any]:
    """Select a reproducible subset without inheriting source-file label order."""

    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    if order == "seeded-hash":
        samples = sorted(
            samples,
            key=lambda sample: hashlib.sha256(
                f"{seed}:{sample.sample_id}".encode("utf-8")
            ).digest(),
        )
    elif order != "source":
        raise ValueError(f"unknown sample order: {order}")
    return samples if limit is None else samples[:limit]


def _sample_task_id(sample: Any) -> str:
    """Return the task identity used to align tool-positive coverage."""

    metadata = getattr(sample, "metadata", {})
    task_id = metadata.get("task_id") if hasattr(metadata, "get") else None
    if task_id not in (None, ""):
        return str(task_id)
    source_episode = (
        metadata.get("source_episode") if hasattr(metadata, "get") else None
    )
    if source_episode not in (None, ""):
        from activemap.agent.identifiers import public_task_id

        return public_task_id(str(source_episode))
    for key in ("episode_id",):
        value = metadata.get(key) if hasattr(metadata, "get") else None
        if value not in (None, ""):
            return str(value)
    raise ValueError("selector sample lacks task_id/source_episode metadata")


def _ensure_task_coverage(
    all_samples: list[Any],
    selected: list[Any],
    required_task_ids: set[str],
    *,
    limit: int | None,
    allow_missing_required_tasks: bool = False,
) -> tuple[list[Any], int]:
    """Inject one deterministic sample for every required task when requested.

    The recurrent GRPO smoke must contain states on which a tool can actually
    be useful.  Source-prefix sampling can silently omit those states, so the
    injection is explicit and recorded in the protocol rather than hidden in
    the evaluator.
    """

    if not required_task_ids:
        return selected, 0
    by_task: dict[str, list[Any]] = {}
    for sample in all_samples:
        by_task.setdefault(_sample_task_id(sample), []).append(sample)
    missing_from_states = sorted(required_task_ids - set(by_task))
    if missing_from_states:
        if not allow_missing_required_tasks:
            raise ValueError(
                "tool-positive supervision tasks are absent from selector states: "
                + ", ".join(missing_from_states[:8])
            )
        required_task_ids = required_task_ids - set(missing_from_states)
        if not required_task_ids:
            return selected, 0
    if limit is not None and limit < len(required_task_ids):
        raise ValueError(
            "--limit is smaller than the number of available tool-positive tasks: "
            f"{limit} < {len(required_task_ids)}"
        )
    selected_tasks = {_sample_task_id(sample) for sample in selected}
    missing = sorted(required_task_ids - selected_tasks)
    if not missing:
        return selected, 0

    selected_ids = {str(sample.sample_id) for sample in selected}
    replacements = [
        sorted(
            (sample for sample in by_task[task] if str(sample.sample_id) not in selected_ids),
            key=lambda sample: str(sample.sample_id),
        )[0]
        for task in missing
    ]
    removable = [
        index
        for index in range(len(selected) - 1, -1, -1)
        if _sample_task_id(selected[index]) not in required_task_ids
    ]
    if len(removable) < len(replacements):
        raise ValueError("cannot inject tool-positive tasks without duplicating samples")
    for index, replacement in zip(
        removable[: len(replacements)], replacements, strict=True
    ):
        selected[index] = replacement
    selected.sort(key=lambda sample: str(sample.sample_id))
    return selected, len(replacements)


def _teacher_forced_completion_logprobs(
    model: Any,
    sequences: Any,
    prompt_length: int,
) -> Any:
    """Recover policy log-probs when generation scores are unavailable/invalid.

    Some Qwen generation backends return an all-zero ``scores`` tuple even
    with ``output_scores=True``.  GRPO must use the frozen policy likelihood,
    so recompute it with a teacher-forced forward pass over the generated
    sequence instead of silently persisting zeros.
    """

    import torch

    if sequences.ndim != 2 or prompt_length <= 0:
        raise ValueError("invalid sequence shape for teacher-forced log-probs")
    completion_length = int(sequences.shape[1] - prompt_length)
    if completion_length <= 0:
        raise ValueError("generated completion is empty")
    with torch.inference_mode():
        outputs = model(
            input_ids=sequences,
            attention_mask=torch.ones_like(sequences),
            use_cache=False,
        )
        logits = outputs.logits[:, :-1].float()
        targets = sequences[:, 1:]
        log_probs = logits.log_softmax(-1)
        token_logprobs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    start = prompt_length - 1
    return token_logprobs[:, start : start + completion_length]


def _portable_path(value: str) -> Any:
    return PurePosixPath(value) if value.startswith("/") else Path(value)


def _parse_asset_root_maps(values: list[str]) -> tuple[tuple[Any, Any], ...]:
    mappings = []
    for value in values:
        if "=" not in value:
            raise ValueError("asset root maps must use SOURCE=TARGET")
        source_text, target_text = value.split("=", 1)
        source, target = _portable_path(source_text), _portable_path(target_text)
        if not source.is_absolute() or not target.is_absolute():
            raise ValueError("asset root maps must contain absolute paths")
        mappings.append((source, target))
    return tuple(mappings)


def _remap_asset_path(
    value: str | None, mappings: tuple[tuple[Any, Any], ...]
) -> str | None:
    if value is None:
        return None
    original = _portable_path(value)
    for source, target in mappings:
        try:
            return str(target / original.relative_to(source))
        except ValueError:
            continue
    return value


def _remap_episode_assets(episode: Any, mappings: tuple[tuple[Any, Any], ...]) -> Any:
    if not mappings:
        return episode
    catalog = [
        item.model_copy(
            update={
                "image_path": _remap_asset_path(item.image_path, mappings),
                "udm_path": _remap_asset_path(item.udm_path, mappings),
            }
        )
        for item in episode.evidence_catalog
    ]
    return episode.model_copy(update={"evidence_catalog": catalog})


def _load_tool_positive_task_ids(path: Path, *, allow_test: bool = False) -> set[str]:
    task_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("oracle_tool_stage", 0)) <= 0:
                continue
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"tool supervision row {line_number} lacks messages")
            observation = json.loads(messages[1]["content"])
            if observation.get("split") == "test" and not allow_test:
                raise ValueError("tool-positive subgroup refuses test supervision")
            task_ids.add(str(observation["task_id"]))
    if not task_ids:
        raise ValueError("tool supervision contains no positive-utility tasks")
    return task_ids


def _extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("action output is not an object")
    return value


def _normalize_structured_action_payload(
    payload: dict[str, Any], observation: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve a semantic selected-evidence pointer into an executable action.

    ``call_id`` and opaque evidence handles are generated executor fields.  A
    model instead selects ``evidence_index`` from the current observation.  The
    raw completion is preserved separately for policy likelihoods and auditing.
    """

    normalized = dict(payload)
    resolution: dict[str, Any] = {"mode": "none", "evidence_index": None}
    if normalized.get("action") != "USE_TOOL":
        return normalized, resolution
    tool_call = normalized.get("tool_call")
    if not isinstance(tool_call, dict):
        return normalized, resolution
    normalized_call = dict(tool_call)
    inputs = normalized_call.get("inputs")
    if not isinstance(inputs, dict):
        return normalized, resolution
    normalized_inputs = dict(inputs)
    evidence_index = normalized_inputs.get("evidence_index")
    if isinstance(evidence_index, str) and re.fullmatch(r"0|[1-9][0-9]*", evidence_index):
        evidence_index = int(evidence_index)
    if isinstance(evidence_index, int) and not isinstance(evidence_index, bool):
        selected = list(getattr(observation, "selected_evidence_ids", []))
        if 0 <= evidence_index < len(selected):
            normalized_inputs.pop("evidence_index", None)
            normalized_inputs["evidence_id"] = str(selected[evidence_index])
            resolution = {
                "mode": "selected_evidence_index_v1",
                "evidence_index": evidence_index,
            }
    if "evidence_id" not in normalized_inputs:
        return normalized, resolution
    tool = str(normalized_call.get("tool", "unknown")).lower()
    task = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(getattr(observation, "task_id", "task")))
    call_id = f"policy-{task[:48]}-step{int(getattr(observation, 'step', 0))}-{tool}"
    if resolution["mode"] == "selected_evidence_index_v1":
        # The model selects the semantic tool action, not a globally unique
        # executor identifier. Always replace an incidental generated call ID.
        normalized_call["call_id"] = call_id
    else:
        normalized_call.setdefault("call_id", call_id)
    normalized_call["inputs"] = normalized_inputs
    normalized["tool_call"] = normalized_call
    return normalized, resolution


def _terminal_key(action: Any) -> str:
    return action.key


def _summarize_rows(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    summaries = []
    for budget in sorted({row["budget"] for row in rows}):
        selected = [row for row in rows if row["budget"] == budget]
        summaries.append(
            {
                "method": name,
                "budget": budget,
                "sample_count": len(selected),
                "terminal_accuracy": float(
                    np.mean([row["terminal_correct"] for row in selected])
                ),
                "false_edit_rate": float(np.mean([row["false_edit"] for row in selected])),
                "missed_edit_rate": float(
                    np.mean([row["missed_edit"] for row in selected])
                ),
                "mean_cost": float(np.mean([row["spent_cost"] for row in selected])),
                "mean_acquisitions": float(
                    np.mean([row["acquisitions"] for row in selected])
                ),
                "mean_tool_calls": float(
                    np.mean([row["tool_calls"] for row in selected])
                ),
                "mean_tool_cost": float(np.mean([row["tool_cost"] for row in selected])),
                "tool_success_rate": (
                    float(sum(row["tool_successes"] for row in selected))
                    / max(sum(row["tool_calls"] for row in selected), 1)
                ),
                "mean_tool_belief_l1_delta": float(
                    np.mean([row["mean_tool_belief_l1_delta"] for row in selected])
                ),
                "tool_action_flip_rate": (
                    float(sum(row["tool_action_flips"] for row in selected))
                    / max(sum(row["tool_calls"] for row in selected), 1)
                ),
                "terminal_edit_flip_rate": float(
                    np.mean(
                        [row["terminal_edit_changed_after_tools"] for row in selected]
                    )
                ),
                "mean_steps": float(np.mean([row["steps"] for row in selected])),
                "mean_final_evidence_quality": float(
                    np.mean([row["final_evidence_quality"] for row in selected])
                ),
                "mean_evidence_quality_gain": float(
                    np.mean([row["evidence_quality_gain"] for row in selected])
                ),
                "mean_quality_cost_utility": float(
                    np.mean([row["quality_cost_utility"] for row in selected])
                ),
                "mean_joint_utility": float(
                    np.mean([row["joint_utility"] for row in selected])
                ),
                "mean_episode_utility_v2_proxy_balanced": float(
                    np.mean(
                        [row["episode_utility_v2_proxy_balanced"] for row in selected]
                    )
                ),
                "mean_episode_utility_v2_proxy_safety": float(
                    np.mean([row["episode_utility_v2_proxy_safety"] for row in selected])
                ),
                "mean_episode_utility_v2_proxy_cost_aware": float(
                    np.mean(
                        [row["episode_utility_v2_proxy_cost_aware"] for row in selected]
                    )
                ),
            }
        )
    return summaries


def _grounded_action_completions(observation: Any) -> list[tuple[str, str]]:
    """Enumerate executable JSON actions for structured policy decoding.

    The controller is responsible for choosing among semantic actions; opaque
    evidence IDs and call IDs remain executor-owned details.  Restricting the
    candidate set to actions the environment can execute prevents malformed
    text generations from being converted into a hidden heuristic fallback.
    """

    from activemap.agent.records import AgentAction, AgentActionType
    from activemap.agent.tool_sft import policy_action_json
    from activemap.geo_tools.records import GeoToolCall, GeoToolName
    from activemap.models import EditOperation

    candidates: list[tuple[str, str]] = []
    terminal_actions = [
        ("REJECT", AgentAction(action=AgentActionType.REJECT)),
        *[
            (
                f"COMMIT:{operation.value}",
                AgentAction(action=AgentActionType.COMMIT, edit=operation),
            )
            for operation in (
                EditOperation.ADD,
                EditOperation.DELETE,
                EditOperation.RESHAPE,
            )
        ],
    ]
    for key, action in terminal_actions:
        candidates.append((key, policy_action_json(action, observation)))

    selected = list(getattr(observation, "selected_evidence_ids", []))
    available_tools = sorted(
        (
            GeoToolName(str(getattr(tool, "value", tool)))
            for tool in getattr(observation, "available_tools", [])
        ),
        key=lambda item: item.value,
    )
    for evidence_index, evidence_id in enumerate(selected):
        for tool in available_tools:
            action = AgentAction(
                action=AgentActionType.USE_TOOL,
                tool_call=GeoToolCall(
                    call_id=f"candidate-{tool.value.lower()}-{evidence_index}",
                    tool=tool,
                    inputs={"evidence_id": str(evidence_id)},
                ),
            )
            candidates.append(
                (
                    f"USE_TOOL:{tool.value}:{evidence_index}",
                    policy_action_json(action, observation),
                )
            )
    return candidates


class StructuredLLMPolicy:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: str,
        max_length: int,
        system_prompt: str,
        tool_system_prompt: str | None = None,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        action_decoder: str = "free",
        candidate_score_batch_size: int = 1,
        record_training_payload: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.tool_system_prompt = tool_system_prompt
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        if action_decoder not in {"free", "candidate-sample"}:
            raise ValueError(f"unknown action decoder: {action_decoder}")
        if action_decoder == "candidate-sample" and not do_sample:
            raise ValueError("candidate-sample decoding requires do_sample=True")
        if candidate_score_batch_size <= 0:
            raise ValueError("candidate_score_batch_size must be positive")
        self.action_decoder = action_decoder
        self.candidate_score_batch_size = candidate_score_batch_size
        self.record_training_payload = record_training_payload
        self.records: list[dict[str, Any]] = []

    def prompt_for(self, observation: Any) -> str:
        # Tool-capable policies need the executable call schema before their
        # first decision; otherwise the initial state exposes only a placeholder.
        if self.tool_system_prompt is not None:
            return self.tool_system_prompt
        return self.system_prompt

    def _sample_grounded_completion(
        self, encoded: Any, observation: Any
    ) -> tuple[str, list[int], list[float], dict[str, Any]]:
        """Sample a valid structured action from normalized model likelihoods.

        This is a policy decoder, not a post-hoc action override: every action
        is scored by the current model under the exact prompt, then sampled
        from the temperature-scaled categorical distribution.  Per-token
        likelihoods are retained for a later recurrent objective.
        """

        import torch

        candidates = _grounded_action_completions(observation)
        if not candidates:
            raise RuntimeError("structured decoder has no executable actions")
        keys, completions = zip(*candidates, strict=True)
        prompt_ids = encoded["input_ids"][0]
        prompt_length = int(prompt_ids.shape[0])
        completion_ids = [
            self.tokenizer(text, add_special_tokens=False)["input_ids"]
            for text in completions
        ]
        if any(not token_ids for token_ids in completion_ids):
            raise RuntimeError("structured decoder produced an empty completion")
        pad_token_id = int(self.tokenizer.pad_token_id)
        sequence_scores: list[float] = []
        per_candidate_token_logprobs: list[list[float]] = []
        for start in range(0, len(completion_ids), self.candidate_score_batch_size):
            batch = completion_ids[start : start + self.candidate_score_batch_size]
            max_length = prompt_length + max(len(token_ids) for token_ids in batch)
            input_ids = torch.full(
                (len(batch), max_length),
                pad_token_id,
                dtype=prompt_ids.dtype,
                device=self.device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for batch_index, token_ids in enumerate(batch):
                length = prompt_length + len(token_ids)
                input_ids[batch_index, :prompt_length] = prompt_ids
                input_ids[batch_index, prompt_length:length] = torch.tensor(
                    token_ids, dtype=prompt_ids.dtype, device=self.device
                )
                attention_mask[batch_index, :length] = 1
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                token_log_probs = outputs.logits[:, :-1].float().log_softmax(-1)
            for batch_index, token_ids in enumerate(batch):
                length = len(token_ids)
                position = slice(prompt_length - 1, prompt_length - 1 + length)
                targets = input_ids[
                    batch_index, prompt_length : prompt_length + length
                ]
                values = token_log_probs[batch_index, position].gather(
                    -1, targets.unsqueeze(-1)
                ).squeeze(-1)
                per_candidate_token_logprobs.append(values.detach().cpu().tolist())
                # Normalize by completion length: terminal actions should not win
                # solely because their JSON serialisation is shorter.
                sequence_scores.append(float(values.mean().item()))
        scores = torch.tensor(sequence_scores, dtype=torch.float32, device=self.device)
        probabilities = torch.softmax(scores / self.temperature, dim=0)
        chosen_index = int(torch.multinomial(probabilities, 1).item())
        entropy = float(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum().item()
        )
        details = {
            "decoder": "candidate-sample",
            "candidate_keys": list(keys),
            # Persist the exact categorical action support.  Recurrent GRPO
            # must optimize the same distribution that was sampled here; the
            # chosen completion alone is not enough to reconstruct it.
            "candidate_completions": list(completions),
            "candidate_completion_token_ids": completion_ids,
            "candidate_probabilities": probabilities.detach().cpu().tolist(),
            "candidate_length_normalized_logprobs": sequence_scores,
            "candidate_score_batch_size": self.candidate_score_batch_size,
            "candidate_temperature": self.temperature,
            "candidate_entropy": entropy,
            "chosen_key": keys[chosen_index],
            "chosen_index": chosen_index,
        }
        return (
            completions[chosen_index],
            completion_ids[chosen_index],
            per_candidate_token_logprobs[chosen_index],
            details,
        )

    def act(self, observation: Any) -> Any:
        import torch

        from activemap.agent.records import AgentAction, AgentActionType
        from activemap.models import EditOperation

        system_prompt = self.prompt_for(observation)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": observation.model_dump_json(exclude_none=True)},
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        decoder_details: dict[str, Any] | None = None
        candidate_training_payload: tuple[list[int], list[float]] | None = None
        if self.action_decoder == "candidate-sample":
            raw, candidate_ids, candidate_logprobs, decoder_details = (
                self._sample_grounded_completion(encoded, observation)
            )
            candidate_training_payload = (candidate_ids, candidate_logprobs)
        else:
            with torch.inference_mode():
                generation_kwargs = {
                    "do_sample": self.do_sample,
                    "max_new_tokens": 96,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                if self.do_sample:
                    generation_kwargs.update(
                        {"temperature": self.temperature, "top_p": self.top_p}
                    )
                generated = self.model.generate(
                    **encoded,
                    **generation_kwargs,
                    return_dict_in_generate=self.record_training_payload,
                    output_scores=self.record_training_payload,
                )
            sequences = generated.sequences if self.record_training_payload else generated
            continuation = sequences[:, encoded["input_ids"].shape[1] :]
            raw = self.tokenizer.batch_decode(continuation, skip_special_tokens=True)[0]
        training_payload = None
        if self.record_training_payload:
            if candidate_training_payload is not None:
                candidate_ids, candidate_logprobs = candidate_training_payload
                token_logprobs = torch.tensor(candidate_logprobs, dtype=torch.float32)
                continuation_ids = candidate_ids
                logprob_source = "candidate_teacher_forced_forward"
            else:
                token_logprobs = self.model.compute_transition_scores(
                    sequences, generated.scores, normalize_logits=True
                )[0]
                continuation_ids = continuation[0].detach().cpu().tolist()
                logprob_source = "generation_scores"
                if (
                    token_logprobs.numel() == 0
                    or not bool(torch.isfinite(token_logprobs).all())
                    or bool(torch.allclose(token_logprobs, torch.zeros_like(token_logprobs)))
                ):
                    token_logprobs = _teacher_forced_completion_logprobs(
                        self.model,
                        sequences,
                        encoded["input_ids"].shape[1],
                    )[0]
                    logprob_source = "teacher_forced_forward"
            if token_logprobs.numel() != len(continuation_ids) or not bool(
                torch.isfinite(token_logprobs).all()
            ):
                raise RuntimeError("policy log-prob recovery returned invalid length or values")
            training_payload = {
                "prompt": prompt,
                # ``encoded`` is the exact truncated prompt evaluated by the
                # behaviour policy.  Saving ids avoids a tokenizer/truncation
                # mismatch when the recurrent objective recomputes scores.
                "prompt_token_ids": encoded["input_ids"][0].detach().cpu().tolist(),
                "completion": raw,
                "completion_token_ids": continuation_ids,
                "old_token_logprobs": token_logprobs.detach().cpu().tolist(),
                "old_sequence_logprob": float(token_logprobs.sum().item()),
                "old_logprob_source": logprob_source,
            }
            if decoder_details is not None:
                training_payload["structured_decoder"] = decoder_details
        schema_valid = False
        executable = False
        error = None
        pointer_resolution: dict[str, Any] = {"mode": "none", "evidence_index": None}
        try:
            payload, pointer_resolution = _normalize_structured_action_payload(
                _extract_json(raw), observation
            )
            action = AgentAction.model_validate(payload)
            schema_valid = True
            if action.action == AgentActionType.ACQUIRE:
                executable = action.evidence_id in {
                    candidate.evidence_id for candidate in observation.candidates
                }
            elif action.action == AgentActionType.USE_TOOL:
                evidence_id = (
                    action.tool_call.inputs.get("evidence_id")
                    if action.tool_call is not None
                    else None
                )
                executable = (
                    action.tool_call is not None
                    and action.tool_call.tool in observation.available_tools
                    and evidence_id in observation.selected_evidence_ids
                )
            else:
                executable = True
        except Exception as exc:
            error = str(exc)
            action = None
        if not executable:
            predicted = observation.belief.predicted_edit
            action = (
                AgentAction(action=AgentActionType.REJECT)
                if predicted == EditOperation.KEEP
                else AgentAction(action=AgentActionType.COMMIT, edit=predicted)
            )
        self.records.append(
            {
                "task_id": observation.task_id,
                "split": observation.split,
                "budget": observation.initial_budget,
                "step": observation.step,
                "schema_valid": schema_valid,
                "executable": executable,
                "raw_output": raw,
                "executed_action": action.key,
                "fallback_used": not executable,
                "pointer_resolution": pointer_resolution,
                "error": error,
                "structured_decoder": decoder_details,
                "training_payload": training_payload,
            }
        )
        return action


class OraclePolicy:
    def __init__(self) -> None:
        self.environment = None

    def act(self, observation: Any) -> Any:
        if self.environment is None:
            raise RuntimeError("oracle policy has no environment")
        return self.environment.oracle_action()


class ForcedToolPolicy:
    """Acquire one ranked item, run the fixed tool pair, then stop."""

    def act(self, observation: Any) -> Any:
        from activemap.agent.records import AgentAction, AgentActionType
        from activemap.geo_tools.records import GeoToolCall, GeoToolName
        from activemap.models import EditOperation

        acquired = observation.selected_evidence_ids[1:]
        for evidence_id in acquired:
            completed = {
                result.tool
                for result in observation.tool_history
                if str(result.outputs.get("evidence_id")) == evidence_id
            }
            for tool in (GeoToolName.IMAGE_QUALITY, GeoToolName.TEMPORAL_CHANGE):
                if tool not in completed and tool in observation.available_tools:
                    return AgentAction(
                        action=AgentActionType.USE_TOOL,
                        tool_call=GeoToolCall(
                            call_id=f"forced-{observation.step}-{tool.value.lower()}",
                            tool=tool,
                            inputs={"evidence_id": evidence_id},
                        ),
                    )
        if observation.candidates:
            return AgentAction(
                action=AgentActionType.ACQUIRE,
                evidence_id=observation.candidates[0].evidence_id,
            )
        predicted = observation.belief.predicted_edit
        return (
            AgentAction(action=AgentActionType.REJECT)
            if predicted == EditOperation.KEEP
            else AgentAction(action=AgentActionType.COMMIT, edit=predicted)
        )


def _evaluate_policy(
    samples: list[Any],
    name: str,
    policy_factory: Any,
    score_fn: Any,
    *,
    episodes: dict[str, Any] | None = None,
    tool_registry: Any | None = None,
    tool_belief_updater_factory: Any | None = None,
    max_tool_calls: int = 0,
    tool_out_size: int = 512,
):
    from tqdm.auto import tqdm

    from activemap.agent.environment import MapMaintenanceEnv, rollout_agent_policy
    from activemap.agent.records import AgentActionType
    from activemap.agent.tools import CounterfactualBeliefUpdater
    from activemap.models import EditOperation

    rows = []
    action_counts: Counter[str] = Counter()
    for sample in tqdm(samples, desc=f"Agent rollout {name}"):
        budget = float(sample.metadata["budget"])
        episode_id = str(sample.metadata.get("source_episode", ""))
        episode = episodes.get(episode_id) if episodes is not None else None
        if tool_registry is not None and episode is None:
            raise ValueError(f"missing episode assets for tool rollout: {episode_id}")
        asset_paths = (
            {item.evidence_id: item.image_path for item in episode.evidence_catalog}
            if episode is not None
            else None
        )
        tool_parameters = None
        if episode is not None:
            tool_parameters = {}
            for item in episode.evidence_catalog:
                x_min, y_min, x_max, y_max = item.region
                tool_parameters[item.evidence_id] = {
                    "pixel_window": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "out_size": [tool_out_size, tool_out_size],
                }
        tool_belief_updater = (
            tool_belief_updater_factory()
            if tool_belief_updater_factory is not None
            else None
        )
        environment = MapMaintenanceEnv(
            sample,
            budget=budget,
            score_fn=score_fn,
            belief_updater=CounterfactualBeliefUpdater(sample),
            tool_registry=tool_registry,
            tool_belief_updater=tool_belief_updater,
            asset_paths=asset_paths,
            tool_parameters=tool_parameters,
            public_identifiers=True,
        )
        policy = policy_factory()
        if isinstance(policy, OraclePolicy):
            policy.environment = environment
        trajectory = rollout_agent_policy(
            environment,
            policy,
            max_acquisitions=len(sample.evidence_ids),
            max_tool_calls=max_tool_calls,
            max_steps=len(sample.evidence_ids) + max_tool_calls + 1,
        )
        terminal = trajectory.transitions[-1]
        target = EditOperation(str(sample.metadata["gt_edit"]))
        predicted_key = _terminal_key(terminal.action)
        target_key = "REJECT" if target == EditOperation.KEEP else f"COMMIT:{target.value}"
        false_edit = (
            target == EditOperation.KEEP and terminal.action.action == AgentActionType.COMMIT
        )
        missed_edit = (
            target != EditOperation.KEEP and terminal.action.action == AgentActionType.REJECT
        )
        terminal_correct = predicted_key == target_key
        wrong_edit = not terminal_correct and not false_edit and not missed_edit
        spent_cost = float(terminal.observation.spent_cost)
        cost_weight = float(sample.metadata.get("cost_weight", 0.0))
        joint_utility = float(terminal.reward) - cost_weight * spent_cost
        evidence_quality = float(environment.current_gain)
        evidence_quality_gain = evidence_quality - float(environment.initial_gain)
        tool_transitions = [
            transition
            for transition in trajectory.transitions
            if transition.action.action == AgentActionType.USE_TOOL
        ]
        tool_cost = float(
            sum(
                transition.next_observation.tool_history[-1].cost
                for transition in tool_transitions
                if transition.next_observation is not None
            )
        )
        tool_successes = sum(
            int(transition.next_observation.tool_history[-1].success)
            for transition in tool_transitions
            if transition.next_observation is not None
        )
        tool_diagnostics = []
        for transition in tool_transitions:
            if transition.next_observation is None:
                continue
            result = transition.next_observation.tool_history[-1]
            tool_diagnostics.append(
                {
                    "tool": result.tool.value,
                    "requested_evidence_id": transition.action.tool_call.inputs.get(
                        "evidence_id"
                    )
                    if transition.action.tool_call is not None
                    else None,
                    "success": result.success,
                    "cost": result.cost,
                    "error": result.error,
                    "artifacts": list(result.artifacts),
                }
            )
        belief_l1_deltas = [
            float(
                np.abs(
                    np.asarray(transition.next_observation.belief.edit_probabilities)
                    - np.asarray(transition.observation.belief.edit_probabilities)
                ).sum()
            )
            for transition in tool_transitions
            if transition.next_observation is not None
        ]
        tool_action_flips = sum(
            int(
                transition.next_observation.belief.predicted_edit
                != transition.observation.belief.predicted_edit
            )
            for transition in tool_transitions
            if transition.next_observation is not None
        )
        pre_tool_edit = (
            tool_transitions[0].observation.belief.predicted_edit.value
            if tool_transitions
            else None
        )
        final_edit = (
            EditOperation.KEEP.value
            if terminal.action.action == AgentActionType.REJECT
            else terminal.action.edit.value
        )
        quality_cost_utility = (
            evidence_quality_gain - float(environment.spent_penalty) - tool_cost
        )
        proxy_utilities = score_episode_profiles(
            final_map_quality=float(terminal_correct),
            prior_map_quality=float(target == EditOperation.KEEP),
            spent_cost=spent_cost,
            budget=budget,
            false_edit=false_edit,
            missed_edit=missed_edit,
            wrong_edit=wrong_edit,
        )
        for transition in trajectory.transitions:
            action_counts[transition.action.action.value] += 1
        rows.append(
            {
                "sample_id": sample.sample_id,
                "task_id": terminal.observation.task_id,
                "split": terminal.observation.split,
                "budget": budget,
                "target": target_key,
                "prediction": predicted_key,
                "terminal_correct": terminal_correct,
                "false_edit": false_edit,
                "missed_edit": missed_edit,
                "wrong_edit": wrong_edit,
                "spent_cost": spent_cost,
                "acquisitions": trajectory.metadata["acquisition_count"],
                "steps": len(trajectory.transitions),
                "terminal_reward": float(terminal.reward),
                "trajectory_return": float(trajectory.total_reward),
                "initial_evidence_quality": float(environment.initial_gain),
                "final_evidence_quality": evidence_quality,
                "evidence_quality_gain": evidence_quality_gain,
                "evidence_penalty": float(environment.spent_penalty),
                "tool_calls": len(tool_transitions),
                "tool_successes": tool_successes,
                "tool_diagnostics": tool_diagnostics,
                "tool_cost": tool_cost,
                "mean_tool_belief_l1_delta": (
                    float(np.mean(belief_l1_deltas)) if belief_l1_deltas else 0.0
                ),
                "tool_action_flips": tool_action_flips,
                "pre_tool_edit": pre_tool_edit,
                "terminal_edit_changed_after_tools": (
                    pre_tool_edit is not None and pre_tool_edit != final_edit
                ),
                "matched_tool_pairs": int(
                    getattr(tool_belief_updater, "matched_pairs", 0)
                ),
                "unmatched_temporal_calls": int(
                    getattr(tool_belief_updater, "unmatched_temporal", 0)
                ),
                "redundant_quality_calls": int(
                    getattr(tool_belief_updater, "redundant_quality", 0)
                ),
                "quality_cost_utility": quality_cost_utility,
                "joint_utility": joint_utility,
                "episode_utility_v2_proxy": proxy_utilities,
                "episode_utility_v2_proxy_balanced": proxy_utilities["balanced"]["value"],
                "episode_utility_v2_proxy_safety": proxy_utilities["safety"]["value"],
                "episode_utility_v2_proxy_cost_aware": proxy_utilities["cost_aware"]["value"],
                "selected_evidence_ids": list(environment.selected),
            }
        )
    return _summarize_rows(rows, name), rows, dict(action_counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("states", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--adapter")
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--generic-checkpoint", action="append", type=Path)
    parser.add_argument("--episodes", type=Path)
    parser.add_argument("--asset-root-map", action="append", default=[])
    parser.add_argument("--tool-belief-checkpoint", type=Path)
    parser.add_argument("--tool-artifact-root", type=Path)
    parser.add_argument("--max-tool-calls", type=int, default=2)
    parser.add_argument("--tool-out-size", type=int, default=512)
    parser.add_argument("--tool-supervision-jsonl", type=Path)
    parser.add_argument("--tool-need-gate", type=Path)
    parser.add_argument("--tool-need-threshold", type=float)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--oracle-step",
        type=int,
        default=0,
        help="evaluate states at this recurrent stage; 0 is the initial selector stage",
    )
    parser.add_argument("--budgets", default="1.5,3.0,4.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selector-device", default="cpu")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--action-decoder",
        choices=("free", "candidate-sample"),
        default="free",
        help=(
            "free generates JSON text directly; candidate-sample samples only from "
            "currently executable structured actions using model likelihoods"
        ),
    )
    parser.add_argument(
        "--candidate-score-batch-size",
        type=int,
        default=1,
        help="maximum candidate completions scored in one forward pass",
    )
    parser.add_argument("--record-training-payload", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sample-order",
        choices=("source", "seeded-hash"),
        default="source",
        help="seeded-hash prevents ordered source files from collapsing label support",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        help="fixed subset seed; defaults to the policy decoding seed",
    )
    parser.add_argument(
        "--ensure-tool-positive",
        action="store_true",
        help="inject one sample for every tool-positive supervision task",
    )
    parser.add_argument(
        "--allow-missing-tool-positive-tasks",
        action="store_true",
        help=(
            "allow positive supervision tasks absent at this oracle step; "
            "record the missing IDs in summary.json instead of hiding them"
        ),
    )
    parser.add_argument(
        "--methods",
        default="oracle,greedy,edit_conditioned_selector,uncertainty,qwen3_4b_sft",
    )
    parser.add_argument("--temporal-pair-closure-margin", type=float, default=0.0)
    args = parser.parse_args()
    if args.oracle_step < 0:
        parser.error("--oracle-step must be non-negative")
    if args.allow_missing_tool_positive_tasks and not args.ensure_tool_positive:
        parser.error(
            "--allow-missing-tool-positive-tasks requires --ensure-tool-positive"
        )
    asset_root_maps = _parse_asset_root_maps(args.asset_root_map)
    if args.split == "test":
        from scripts.frozen_test_access import assert_frozen_test_access

        assert_frozen_test_access()
    if args.tool_out_size <= 0:
        raise ValueError("--tool-out-size must be positive")
    if args.do_sample and (args.temperature <= 0 or not 0 < args.top_p <= 1):
        raise ValueError("sampled decoding requires temperature > 0 and 0 < top-p <= 1")
    if args.action_decoder == "candidate-sample" and not args.do_sample:
        raise ValueError("candidate-sample decoding requires --do-sample")
    if args.candidate_score_batch_size <= 0:
        raise ValueError("--candidate-score-batch-size must be positive")
    if args.temporal_pair_closure_margin < 0.0:
        raise ValueError("--temporal-pair-closure-margin must be non-negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(
            f"refusing to overwrite non-empty rollout directory: {args.output_dir}"
        )

    from activemap.agent.heuristics import (
        AcquireUntilBudgetPolicy,
        acquire_all_scores,
        cheapest_scores,
        greedy_utility_scores,
        mapex_scores,
        quality_first_scores,
        stable_random_scores,
        uncertainty_scores,
    )
    from activemap.agent.tools import GreedyAgentPolicy, TemporalPairClosurePolicy
    from activemap.inference import SelectorEnsemblePredictor
    from activemap.models import EpisodeRecord
    from activemap.training.data import load_selector_samples

    budgets = {float(value) for value in args.budgets.split(",")}
    requested_methods = _parse_methods(args.methods)
    tool_positive_task_ids = (
        _load_tool_positive_task_ids(
            args.tool_supervision_jsonl,
            allow_test=args.split == "test",
        )
        if args.tool_supervision_jsonl is not None
        else set()
    )
    all_samples = [
        sample
        for sample in load_selector_samples(args.states, split=args.split)
        if int(sample.metadata.get("oracle_step", 0)) == args.oracle_step
        and float(sample.metadata.get("budget", -1.0)) in budgets
    ]
    sample_seed = args.seed if args.sample_seed is None else args.sample_seed
    samples = _select_samples(
        all_samples, limit=args.limit, order=args.sample_order, seed=sample_seed
    )
    available_task_ids = {_sample_task_id(sample) for sample in all_samples}
    missing_tool_positive_tasks = sorted(
        tool_positive_task_ids - available_task_ids
    )
    injected_tool_positive_tasks = 0
    if args.ensure_tool_positive:
        if args.tool_supervision_jsonl is None:
            raise ValueError("--ensure-tool-positive requires --tool-supervision-jsonl")
        samples, injected_tool_positive_tasks = _ensure_task_coverage(
            all_samples,
            samples,
            tool_positive_task_ids,
            limit=args.limit,
            allow_missing_required_tasks=args.allow_missing_tool_positive_tasks,
        )
    sample_label_counts = dict(
        sorted(Counter(str(sample.metadata.get("gt_edit", "UNKNOWN")) for sample in samples).items())
    )
    selector = SelectorEnsemblePredictor(args.checkpoint, device=args.selector_device)
    generic_selector = (
        SelectorEnsemblePredictor(args.generic_checkpoint, device=args.selector_device)
        if args.generic_checkpoint
        else None
    )
    if "generic_selector" in requested_methods and generic_selector is None:
        raise ValueError("generic_selector requires --generic-checkpoint")
    tool_methods = {
        "forced_tools",
        "qwen3_4b_sft_tools_no_belief",
        "qwen3_4b_sft_tool_to_belief",
        "qwen3_4b_sft_calibrated_tools_no_belief",
        "qwen3_4b_sft_calibrated_tool_to_belief",
        "edit_conditioned_proactive_tools",
    }
    requested_tool_methods = requested_methods & tool_methods
    if requested_tool_methods and args.episodes is None:
        raise ValueError("tool rollout methods require --episodes")
    if (
        requested_methods
        & {
            "forced_tools",
            "qwen3_4b_sft_tool_to_belief",
            "qwen3_4b_sft_calibrated_tool_to_belief",
        }
        and args.tool_belief_checkpoint is None
    ):
        raise ValueError("recurrent tool methods require --tool-belief-checkpoint")
    calibrated_no_belief_method = "qwen3_4b_sft_calibrated_tools_no_belief"
    calibrated_method = "qwen3_4b_sft_calibrated_tool_to_belief"
    calibrated_methods = {
        calibrated_no_belief_method,
        calibrated_method,
        "edit_conditioned_proactive_tools",
    }
    if requested_methods & calibrated_methods and args.tool_need_gate is None:
        raise ValueError("calibrated tool policy requires --tool-need-gate")
    episodes: dict[str, Any] | None = None
    tool_registry = None
    paired_updater = None
    if requested_tool_methods:
        from activemap.agent.tool_belief_model import PairedToolBeliefUpdater
        from activemap.geo_tools.raster import ImageQualityTool, TemporalChangeTool
        from activemap.geo_tools.registry import GeoToolRegistry

        episodes = {}
        with args.episodes.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    episode = EpisodeRecord.model_validate_json(line)
                except Exception as exc:
                    raise ValueError(
                        f"invalid episode at {args.episodes}:{line_number}"
                    ) from exc
                episode = _remap_episode_assets(episode, asset_root_maps)
                if episode.split == "test" and args.split != "test":
                    raise ValueError("tool rollout setup refuses test episodes")
                if episode.split == args.split:
                    episodes[episode.episode_id] = episode
        if not episodes:
            raise ValueError(f"tool rollout setup found no {args.split} episodes")
        artifact_root = args.tool_artifact_root or (args.output_dir / "tool_artifacts")
        tool_registry = GeoToolRegistry()
        tool_registry.register(ImageQualityTool())
        tool_registry.register(TemporalChangeTool(artifact_root / "temporal_change"))
        if args.tool_belief_checkpoint is not None:
            paired_updater = PairedToolBeliefUpdater.from_checkpoint(
                args.tool_belief_checkpoint,
                device=args.selector_device,
            )
    llm_policies: dict[str, StructuredLLMPolicy] = {}
    tool_need_records: dict[str, list[dict[str, Any]]] = {}
    requested_llm_methods = requested_methods & {
        "qwen3_4b_sft",
        "qwen3_4b_sft_tools_no_belief",
        "qwen3_4b_sft_tool_to_belief",
        "qwen3_4b_sft_calibrated_tools_no_belief",
        "qwen3_4b_sft_calibrated_tool_to_belief",
    }
    if requested_llm_methods:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        from activemap.agent.tool_sft import TOOL_SYSTEM_PROMPT
        from activemap.agent.trajectories import SYSTEM_PROMPT

        set_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, dtype=torch.bfloat16
        )
        if args.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter)
        model = model.to(args.device).eval()
        for method in requested_llm_methods:
            llm_policies[method] = StructuredLLMPolicy(
                model,
                tokenizer,
                device=args.device,
                max_length=args.max_length,
                system_prompt=(
                    SYSTEM_PROMPT
                ),
                tool_system_prompt=(
                    None if method == "qwen3_4b_sft" else TOOL_SYSTEM_PROMPT
                ),
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                action_decoder=args.action_decoder,
                candidate_score_batch_size=args.candidate_score_batch_size,
                record_training_payload=args.record_training_payload,
            )
    available = {
        "oracle": (OraclePolicy, selector.action_scores),
        # Historical results call this executor "greedy". Keep the alias while
        # exposing its actual learned-selector identity in new evaluations.
        "greedy": (GreedyAgentPolicy, selector.action_scores),
        "edit_conditioned_selector": (GreedyAgentPolicy, selector.action_scores),
        "edit_conditioned_pair_closure": (
            lambda: TemporalPairClosurePolicy(
                closure_margin=args.temporal_pair_closure_margin
            ),
            selector.action_scores,
        ),
        "random": (AcquireUntilBudgetPolicy, stable_random_scores),
        "acquire_all": (AcquireUntilBudgetPolicy, acquire_all_scores),
        "cheapest": (AcquireUntilBudgetPolicy, cheapest_scores),
        "quality_first": (AcquireUntilBudgetPolicy, quality_first_scores),
        "uncertainty": (AcquireUntilBudgetPolicy, uncertainty_scores),
        "mapex": (AcquireUntilBudgetPolicy, mapex_scores),
        "greedy_utility": (GreedyAgentPolicy, greedy_utility_scores),
    }
    if generic_selector is not None:
        available["generic_selector"] = (
            GreedyAgentPolicy,
            generic_selector.action_scores,
        )
    if "qwen3_4b_sft" in llm_policies:
        available["qwen3_4b_sft"] = (
            lambda: llm_policies["qwen3_4b_sft"],
            selector.action_scores,
        )
    if requested_tool_methods:
        from activemap.agent.tool_belief_model import SequentialPairedToolBeliefUpdater

        recurrent_factory = (
            (lambda: SequentialPairedToolBeliefUpdater(paired_updater))
            if paired_updater is not None
            else None
        )
        available["forced_tools"] = (ForcedToolPolicy, selector.action_scores)
        if "qwen3_4b_sft_tools_no_belief" in llm_policies:
            available["qwen3_4b_sft_tools_no_belief"] = (
                lambda: llm_policies["qwen3_4b_sft_tools_no_belief"],
                selector.action_scores,
            )
        if "qwen3_4b_sft_tool_to_belief" in llm_policies:
            available["qwen3_4b_sft_tool_to_belief"] = (
                lambda: llm_policies["qwen3_4b_sft_tool_to_belief"],
                selector.action_scores,
            )
        tool_need_gate = None
        tool_need_threshold = None
        tool_need_records = {
            name: [] for name in requested_methods & calibrated_methods
        }
        if requested_methods & calibrated_methods:
            from joblib import load

            from activemap.agent.tool_need_gate import CalibratedToolNeedPolicy

            tool_need_gate = load(args.tool_need_gate)
            gate_summary_path = args.tool_need_gate.parent / "summary.json"
            gate_summary = json.loads(gate_summary_path.read_text(encoding="utf-8"))
            tool_need_threshold = (
                args.tool_need_threshold
                if args.tool_need_threshold is not None
                else float(gate_summary["selected"]["threshold"])
            )
            if calibrated_method in llm_policies:
                available[calibrated_method] = (
                    lambda: CalibratedToolNeedPolicy(
                        llm_policies[calibrated_method],
                        tool_need_gate,
                        threshold=tool_need_threshold,
                        records=tool_need_records[calibrated_method],
                        proactive_pair=True,
                        controller="qwen3_4b_sft",
                    ),
                    selector.action_scores,
                )
            if calibrated_no_belief_method in llm_policies:
                available[calibrated_no_belief_method] = (
                    lambda: CalibratedToolNeedPolicy(
                        llm_policies[calibrated_no_belief_method],
                        tool_need_gate,
                        threshold=tool_need_threshold,
                        records=tool_need_records[calibrated_no_belief_method],
                        proactive_pair=True,
                        controller="qwen3_4b_sft",
                    ),
                    selector.action_scores,
                )
            proactive_selector = "edit_conditioned_proactive_tools"
            if proactive_selector in requested_methods:
                available[proactive_selector] = (
                    lambda: CalibratedToolNeedPolicy(
                        GreedyAgentPolicy(),
                        tool_need_gate,
                        threshold=tool_need_threshold,
                        records=tool_need_records[proactive_selector],
                        proactive_pair=True,
                        controller="edit_conditioned_selector",
                    ),
                    selector.action_scores,
                )
    method_order = (
        "oracle",
        "greedy",
        "generic_selector",
        "edit_conditioned_selector",
        "edit_conditioned_pair_closure",
        "edit_conditioned_proactive_tools",
        "random",
        "acquire_all",
        "cheapest",
        "quality_first",
        "uncertainty",
        "mapex",
        "greedy_utility",
        "qwen3_4b_sft",
        "forced_tools",
        "qwen3_4b_sft_tools_no_belief",
        "qwen3_4b_sft_tool_to_belief",
        "qwen3_4b_sft_calibrated_tools_no_belief",
        "qwen3_4b_sft_calibrated_tool_to_belief",
    )
    methods = [(name, *available[name]) for name in method_order if name in requested_methods]
    summaries = []
    details = {}
    action_counts = {}
    subgroup_summaries = []
    for name, factory, score_fn in methods:
        if name in llm_policies and args.do_sample:
            from transformers import set_seed

            set_seed(args.seed)
        is_tool_method = name in tool_methods
        method_summary, rows, counts = _evaluate_policy(
            samples,
            name,
            factory,
            score_fn,
            episodes=episodes if is_tool_method else None,
            tool_registry=tool_registry if is_tool_method else None,
            tool_belief_updater_factory=(
                recurrent_factory
                if name
                in {
                    "forced_tools",
                    "qwen3_4b_sft_tool_to_belief",
                    "qwen3_4b_sft_calibrated_tool_to_belief",
                    "edit_conditioned_proactive_tools",
                }
                else None
            ),
            max_tool_calls=args.max_tool_calls if is_tool_method else 0,
            tool_out_size=args.tool_out_size,
        )
        summaries.extend(method_summary)
        for row in rows:
            row["tool_positive_episode"] = row["task_id"] in tool_positive_task_ids
            row["evaluation_seed"] = args.seed
        positive_rows = [row for row in rows if row["tool_positive_episode"]]
        if positive_rows:
            subgroup_summaries.extend(_summarize_rows(positive_rows, name))
        details[name] = rows
        action_counts[name] = counts
    llm_validity_by_method = {
        name: {
            "call_count": len(policy.records),
            "schema_valid_rate": float(
                np.mean([row["schema_valid"] for row in policy.records])
            )
            if policy.records
            else None,
            "executable_valid_rate": float(
                np.mean([row["executable"] for row in policy.records])
            )
            if policy.records
            else None,
            "fallback_rate": float(
                np.mean([row["fallback_used"] for row in policy.records])
            )
            if policy.records
            else None,
            "pointer_resolution_counts": dict(
                sorted(
                    Counter(
                        str(
                            row.get("pointer_resolution", {}).get("mode", "none")
                        )
                        for row in policy.records
                    ).items()
                )
            ),
        }
        for name, policy in llm_policies.items()
    }
    legacy_validity = llm_validity_by_method.get("qwen3_4b_sft")
    if legacy_validity is None and len(llm_validity_by_method) == 1:
        legacy_validity = next(iter(llm_validity_by_method.values()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "split": args.split,
                    "evaluation_seed": args.seed,
                    "oracle_step": args.oracle_step,
                    "decoding": {
                        "do_sample": args.do_sample,
                        "temperature": args.temperature if args.do_sample else None,
                        "top_p": args.top_p if args.do_sample else None,
                        "action_decoder": args.action_decoder,
                        "candidate_score_batch_size": args.candidate_score_batch_size,
                        "max_new_tokens": 96,
                        "record_training_payload": args.record_training_payload,
                    },
                    "budgets": sorted(budgets),
                    "sample_count": len(samples),
                    "sample_order": args.sample_order,
                    "sample_seed": sample_seed,
                    "tool_positive_coverage": {
                        "requested": args.ensure_tool_positive,
                        "allow_missing_tasks": args.allow_missing_tool_positive_tasks,
                        "injected_task_count": injected_tool_positive_tasks,
                        "positive_task_count": len(tool_positive_task_ids),
                        "available_at_oracle_step_count": len(available_task_ids & tool_positive_task_ids),
                        "missing_from_oracle_step_count": len(missing_tool_positive_tasks),
                        "missing_from_oracle_step_task_ids": missing_tool_positive_tasks,
                        "selected_task_count": len(
                            {
                                _sample_task_id(sample)
                                for sample in samples
                                if _sample_task_id(sample) in tool_positive_task_ids
                            }
                        ),
                    },
                    "sample_label_counts": sample_label_counts,
                    "test_assets_read": args.split == "test",
                    "invalid_action_fallback": "calibrated belief terminal action",
                    "primary_closed_loop_utility": (
                        "evidence_quality_gain - evidence_cost_and_false_edit_penalty - tool_cost"
                    ),
                    "evidence_quality": "0.5*raster_iou + 0.5*edit_correct",
                    "final_map_writeback_quality_available": False,
                    "legacy_joint_utility": "terminal_reward - cost_weight * spent_cost",
                    "episode_utility_v2_proxy": utility_protocol(
                        quality_source="terminal_operation_exactness_proxy",
                        paper_primary=False,
                    ),
                    "edit_conditioned_checkpoints": [str(path) for path in args.checkpoint],
                    "generic_checkpoints": (
                        [str(path) for path in args.generic_checkpoint]
                        if args.generic_checkpoint
                        else []
                    ),
                    "legacy_greedy_alias": "edit_conditioned_selector",
                    "heuristic_protocol": {
                        "seed_semantics": "deterministic_sha256_or_fixed_rule",
                        "label_free": True,
                        "budget_filling": (
                            "random/acquire_all/cheapest/quality_first/uncertainty/mapex acquire "
                            "ranked affordable evidence until budget exhaustion"
                        ),
                        "random": "sha256(sample_id,evidence_id) ranking",
                        "acquire_all": (
                            "fixed catalog-order acquisition of every affordable item"
                        ),
                        "cheapest": "minimum evidence cost ranking",
                        "quality_first": (
                            "maximum clarity*candidate-predictive-confidence ranking"
                        ),
                        "uncertainty": "maximum candidate predictive-entropy ranking",
                        "mapex": (
                            "candidate-to-belief JS divergence*(0.25+0.75*clarity)"
                            "/cost ranking"
                        ),
                        "greedy_utility": (
                            "0.45*quality+0.35*uncertainty+0.20*(1-cost_norm)"
                            "-0.50*false_edit_risk with observable STOP"
                        ),
                    },
                    "tool_methods": sorted(requested_tool_methods),
                    "asset_root_maps": [
                        {"source": str(source), "target": str(target)}
                        for source, target in asset_root_maps
                    ],
                    "tool_belief_checkpoint": (
                        str(args.tool_belief_checkpoint)
                        if args.tool_belief_checkpoint is not None
                        else None
                    ),
                    "max_tool_calls": args.max_tool_calls,
                    "tool_out_size": args.tool_out_size,
                    "tool_positive_task_count": len(tool_positive_task_ids),
                    "tool_positive_source": (
                        str(args.tool_supervision_jsonl)
                        if args.tool_supervision_jsonl is not None
                        else None
                    ),
                    "tool_need_gate": (
                        {
                            "threshold": tool_need_threshold,
                            "protocol": "natural-prior-proactive-paired-tool-gate-v2",
                            "methods": {
                                name: {
                                    "decisions": len(records),
                                    "admitted": sum(row["admitted"] for row in records),
                                    "proactive_calls": sum(
                                        row["source"] in {
                                            "proactive_quality_pair",
                                            "proactive_temporal_pair",
                                        }
                                        for row in records
                                    ),
                                    "eligible_checks": sum(
                                        row.get("eligible") is True for row in records
                                    ),
                                }
                                for name, records in tool_need_records.items()
                            },
                        }
                        if requested_methods & calibrated_methods
                        else None
                    ),
                    "source_fingerprints": {
                        "states": _source_record(args.states),
                        "episodes": _source_record(args.episodes),
                        "adapter": _source_record(Path(args.adapter)) if args.adapter else None,
                        "selector_checkpoints": [
                            _source_record(path) for path in args.checkpoint
                        ],
                        "generic_selector_checkpoints": [
                            _source_record(path)
                            for path in (args.generic_checkpoint or [])
                        ],
                        "tool_belief_checkpoint": _source_record(
                            args.tool_belief_checkpoint
                        ),
                        "tool_supervision": _source_record(
                            args.tool_supervision_jsonl
                        ),
                        "tool_need_gate": _source_record(args.tool_need_gate),
                    },
                },
                "results": summaries,
                "tool_positive_results": subgroup_summaries,
                "action_counts": action_counts,
                "llm_validity": legacy_validity,
                "llm_validity_by_method": llm_validity_by_method,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for name, rows in details.items():
        with (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    for name, policy in llm_policies.items():
        for row in policy.records:
            row["evaluation_seed"] = args.seed
        with (args.output_dir / f"llm_calls_{name}.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in policy.records:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    for name, records in tool_need_records.items():
        with (args.output_dir / f"tool_need_gate_records_{name}.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in records:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    if "qwen3_4b_sft" in llm_policies:
        with (args.output_dir / "llm_calls.jsonl").open("w", encoding="utf-8") as handle:
            for row in llm_policies["qwen3_4b_sft"].records:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
