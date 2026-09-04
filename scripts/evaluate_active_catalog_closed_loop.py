#!/usr/bin/env python3
"""Model-driven recurrent active-catalog rollout with frozen updater belief fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from activemap.agent.active_catalog import (
    ACTIVE_CATALOG_SYSTEM_PROMPT,
    active_catalog_state_from_observation,
    observable_shortlist_scores,
    terminal_agent_action,
)
from activemap.agent.active_catalog_candidate_ranker import CandidateUtilityRankerPredictor
from activemap.agent.active_catalog_joint import joint_transition_from_agent_transition
from activemap.agent.active_catalog_tool_gate import SelectiveToolPolicy
from activemap.agent.environment import MapMaintenanceEnv, rollout_agent_policy
from activemap.agent.records import AgentAction, AgentActionType, AgentObservation
from activemap.agent.sequential_controller import (
    ControllerStage,
    SelectionDecision,
    SequentialControllerAction,
)
from activemap.agent.tools import CounterfactualBeliefUpdater
from activemap.evaluation.episode_utility import (
    score_episode_profiles,
    utility_protocol,
)
from activemap.geo_tools.records import GeoToolCall, GeoToolName
from activemap.models import EditOperation, EpisodeRecord
from activemap.selector_records import SelectorSample
from scripts.build_active_catalog_gate_sft import GATE_SYSTEM_PROMPT
from scripts.evaluate_active_catalog_gate_vlm import parse_gate_output
from scripts.evaluate_active_catalog_selector import _extract_json_object


REACT_SYSTEM_PROMPT = """You are a budget-aware visual map-maintenance controller.
At each step, inspect the current image, editable-map belief, acquired evidence,
available evidence, prior tool observations, and remaining budget. Return exactly
one JSON object with a short reason and one executable action:
{"reason":"brief evidence-based reason","action":"ACQUIRE","evidence_id":"..."}
{"reason":"brief evidence-based reason","action":"USE_TOOL","tool":"IMAGE_QUALITY","evidence_id":"..."}
{"reason":"brief evidence-based reason","action":"STOP"}
ACQUIRE may name only an available candidate. USE_TOOL may name only an available
tool and already acquired evidence. STOP executes the current frozen belief as the
terminal editable-map decision. Prefer STOP when additional evidence is unlikely to
improve map quality enough to justify its cost. Do not invent identifiers or fields."""

PLAN_EXECUTE_SYSTEM_PROMPT = """You are an open-loop visual map-maintenance planner.
Inspect only the initial image, editable-map belief, candidate catalog, and budget.
Return one JSON plan containing zero, one, or two available evidence identifiers:
{"evidence_ids":["evidence-id-1","evidence-id-2"]}
The plan will be executed without seeing intermediate belief updates. Use an empty
list when added evidence is not worth its cost. Do not invent identifiers or fields."""

GEOMMAGENT_SYSTEM_PROMPT = """You are a GeoMMAgent-style coordinator for editable
map maintenance. Recurrently coordinate planning, retrieval, perception, reasoning,
and self-evaluation under the stated budget. Return exactly one JSON object:
{"stage":"PLAN|EXECUTE|SELF_EVALUATE","reason":"brief grounded reason",
 "action":"ACQUIRE|USE_TOOL|STOP|ACCEPT|REEXECUTE", ...}
ACQUIRE must name one available evidence_id. USE_TOOL must name one available tool
and an already acquired evidence_id. REEXECUTE must include either an available
evidence_id or an available tool plus acquired evidence_id. ACCEPT and STOP execute
the current belief. Self-evaluation may request more evidence only when it is likely
to improve executable map quality enough to justify cost. Never use target, oracle,
or invented fields."""

SENSESEARCH_SYSTEM_PROMPT = """You are a SenseSearch-style high-resolution
search-reasoning controller for editable map maintenance. Adaptively interleave:
image search (ACQUIRE an available evidence_id), image crop or fine-grained
perception (USE_TOOL with RASTER_CROP or RASTER_SEGMENT), map-context lookup
(USE_TOOL with VECTOR_INSPECT), and reasoning. Return exactly one JSON object:
{"stage":"SEARCH|REASON","reason":"brief grounded reason",
 "action":"ACQUIRE|USE_TOOL|STOP", ...}
Tool calls require already acquired evidence. Stop when further search is not worth
its budget cost. Never invent identifiers, tools, target information, or oracle
outcomes."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authorize_test_split(split: str) -> None:
    if split == "test":
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()


def load_initial_samples(
    path: Path,
    *,
    split: str = "val",
    limit: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> list[SelectorSample]:
    _authorize_test_split(split)
    rows = []
    identities = set()
    with path.open(encoding="utf-8") as handle:
        split_index = 0
        for line in handle:
            if not line.strip():
                continue
            row = SelectorSample.model_validate_json(line)
            if row.split != split or int(row.metadata.get("oracle_step", -1)) != 0:
                continue
            identity = (str(row.metadata["source_episode"]), float(row.metadata["budget"]))
            if identity in identities:
                raise ValueError(f"duplicate initial rollout state: {identity}")
            identities.add(identity)
            if split_index % num_shards != shard_index:
                split_index += 1
                continue
            split_index += 1
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no {split} step-0 selector states")
    return rows


def load_episodes(path: Path, *, split: str = "val") -> dict[str, EpisodeRecord]:
    _authorize_test_split(split)
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            episode = EpisodeRecord.model_validate_json(line)
            if episode.split != split:
                continue
            if episode.episode_id in result:
                raise ValueError(f"duplicate validation episode: {episode.episode_id}")
            result[episode.episode_id] = episode
    if not result:
        raise ValueError(f"no {split} episodes")
    return result


def parse_root_map(value: str) -> tuple[str, str]:
    source, separator, target = value.partition("=")
    if not separator or not source or not target:
        raise argparse.ArgumentTypeError("asset root maps must use SOURCE=TARGET")
    return source.rstrip("/"), target.rstrip("/")


def remap_asset_path(path: str, root_maps: list[tuple[str, str]]) -> str:
    for source, target in root_maps:
        if path == source or path.startswith(source + "/"):
            return target + path[len(source) :]
    return path


def build_model_tool_registry(policy_mode: str, output_root: Path) -> Any:
    from activemap.geo_tools.raster import (
        ImageQualityTool,
        RasterCropTool,
        RasterSegmentTool,
        TemporalChangeTool,
    )
    from activemap.geo_tools.registry import GeoToolRegistry
    from activemap.geo_tools.vector import VectorInspectTool

    registry = GeoToolRegistry()
    registry.register(ImageQualityTool())
    registry.register(TemporalChangeTool(output_root / "temporal_change"))
    if policy_mode == "sensesearch_style":
        registry.register(RasterCropTool(output_root / "raster_crop"))
        registry.register(RasterSegmentTool(output_root / "raster_segment"))
        registry.register(VectorInspectTool())
    return registry


def image_index(
    sft_path: Path, evaluation_index_path: Path, *, split: str = "val"
) -> dict[str, Path]:
    _authorize_test_split(split)
    rows = [
        json.loads(line)
        for line in sft_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no visual prompt records in {sft_path}")
    hidden = {
        str(row["example_id"]): row
        for row in (
            json.loads(line)
            for line in evaluation_index_path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    result = {}
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) not in {2, 3}:
            raise ValueError("visual prompt row must contain system/user messages")
        if [message.get("role") for message in messages[:2]] != ["system", "user"]:
            raise ValueError("visual prompt roles must begin with system and user")
        if row.get("split") != split:
            raise ValueError("visual prompt split mismatch")
        evaluation = hidden.get(str(row["example_id"]))
        if evaluation is None or evaluation.get("split") != split:
            raise ValueError("SFT/evaluation image index mismatch")
        episode_id = str(evaluation["source_episode"])
        image_parts = [
            item
            for item in messages[1]["content"]
            if item.get("type") == "image"
        ]
        if len(image_parts) != 1:
            raise ValueError("closed-loop prompt requires exactly one image")
        image = Path(str(image_parts[0]["image"]))
        if not image.is_absolute():
            image = (sft_path.parent / image).resolve()
        previous = result.get(episode_id)
        if previous is not None and previous != image:
            raise ValueError(f"episode has inconsistent prompt images: {episode_id}")
        result[episode_id] = image
    return result


def metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("empty closed-loop evaluation")
    model_rows = [row for row in rows if row.get("model_action_count", 0) > 0]
    operation_names = ("KEEP", "ADD", "DELETE", "RESHAPE")
    prediction_rows = [
        row
        for row in rows
        if row.get("predicted_edit") in operation_names
        and row.get("target_edit") in operation_names
    ]
    prediction_counts = {
        operation: sum(row["predicted_edit"] == operation for row in prediction_rows)
        for operation in operation_names
    }
    prediction_probabilities = np.asarray(
        [prediction_counts[operation] for operation in operation_names],
        dtype=np.float64,
    )
    if prediction_probabilities.sum() > 0:
        prediction_probabilities /= prediction_probabilities.sum()
    nonzero = prediction_probabilities[prediction_probabilities > 0]
    normalized_entropy = (
        float(-np.sum(nonzero * np.log(nonzero)) / np.log(len(operation_names)))
        if len(nonzero)
        else 0.0
    )
    result = {
        "episodes": float(len(rows)),
        "terminal_accuracy": sum(row["terminal_correct"] for row in rows) / len(rows),
        "false_edit_rate": sum(row["false_edit"] for row in rows) / len(rows),
        "missed_edit_rate": sum(row["missed_edit"] for row in rows) / len(rows),
        "wrong_edit_rate": sum(row.get("wrong_edit", False) for row in rows) / len(rows),
        "mean_acquisitions": float(np.mean([row["acquisitions"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_cost": float(np.mean([row["spent_cost"] for row in rows])),
        "mean_tool_calls": float(np.mean([row.get("tool_calls", 0) for row in rows])),
        "mean_tool_cost": float(
            np.mean([row.get("tool_cost", 0.0) for row in rows])
        ),
        "tool_call_episode_rate": float(
            np.mean([row.get("tool_calls", 0) > 0 for row in rows])
        ),
        "mean_tool_belief_l1_delta": float(
            np.mean([row.get("tool_belief_l1_delta", 0.0) for row in rows])
        ),
        "mean_quality_gain": float(np.mean([row["quality_gain"] for row in rows])),
        "mean_quality_cost_utility": float(
            np.mean([row["quality_cost_utility"] for row in rows])
        ),
        "mean_episode_utility_v2_proxy_balanced": float(
            np.mean([row.get("episode_utility_v2_proxy_balanced", 0.0) for row in rows])
        ),
        "mean_episode_utility_v2_proxy_safety": float(
            np.mean([row.get("episode_utility_v2_proxy_safety", 0.0) for row in rows])
        ),
        "mean_episode_utility_v2_proxy_cost_aware": float(
            np.mean([row.get("episode_utility_v2_proxy_cost_aware", 0.0) for row in rows])
        ),
        "valid_action_rate": float(
            np.mean(
                [
                    row["valid_action_count"] / row["model_action_count"]
                    for row in model_rows
                ]
            )
            if model_rows
            else 1.0
        ),
        "fallback_episode_rate": (
            sum(row["fallback_count"] > 0 for row in model_rows) / len(model_rows)
            if model_rows
            else 0.0
        ),
        "zero_acquisition_rate": float(
            np.mean([row["acquisitions"] == 0 for row in rows])
        ),
        "two_plus_acquisition_rate": float(
            np.mean([row["acquisitions"] >= 2 for row in rows])
        ),
        "prediction_entropy_normalized": normalized_entropy,
    }
    for operation in operation_names:
        suffix = operation.lower()
        result[f"predicted_{suffix}_rate"] = (
            prediction_counts[operation] / len(prediction_rows)
            if prediction_rows
            else 0.0
        )
        target_rows = [
            row for row in prediction_rows if row["target_edit"] == operation
        ]
        result[f"recall_{suffix}"] = (
            sum(row["predicted_edit"] == operation for row in target_rows)
            / len(target_rows)
            if target_rows
            else 0.0
        )
    return result


def trajectory_row(
    sample: SelectorSample,
    environment: MapMaintenanceEnv,
    trajectory: Any,
    *,
    policy_name: str,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    events = events or []
    terminal = trajectory.transitions[-1]
    target = EditOperation(str(sample.metadata["gt_edit"]))
    prediction = (
        EditOperation.KEEP
        if terminal.action.action == AgentActionType.REJECT
        else terminal.action.edit
    )
    if prediction is None:
        raise ValueError("terminal COMMIT action lacks an edit")
    quality_gain = environment.current_gain - environment.initial_gain
    tool_transitions = [
        item
        for item in trajectory.transitions
        if item.action.action == AgentActionType.USE_TOOL
    ]
    tool_belief_delta = sum(
        sum(
            abs(after - before)
            for before, after in zip(
                item.observation.belief.edit_probabilities,
                item.next_observation.belief.edit_probabilities,
                strict=True,
            )
        )
        for item in tool_transitions
        if item.next_observation is not None
    )
    false_edit = target == EditOperation.KEEP and prediction != EditOperation.KEEP
    missed_edit = target != EditOperation.KEEP and prediction == EditOperation.KEEP
    wrong_edit = (
        target != EditOperation.KEEP
        and prediction != EditOperation.KEEP
        and prediction != target
    )
    proxy_utilities = score_episode_profiles(
        final_map_quality=float(prediction == target),
        prior_map_quality=float(target == EditOperation.KEEP),
        spent_cost=float(terminal.observation.spent_cost),
        budget=float(sample.metadata["budget"]),
        false_edit=false_edit,
        missed_edit=missed_edit,
        wrong_edit=wrong_edit,
    )
    return {
        "sample_id": sample.sample_id,
        "source_episode": str(sample.metadata["source_episode"]),
        "aoi_id": str(sample.metadata["aoi_id"]),
        "split": sample.split,
        "policy": policy_name,
        "budget": float(sample.metadata["budget"]),
        "target_edit": target.value,
        "predicted_edit": prediction.value,
        "terminal_correct": prediction == target,
        "false_edit": false_edit,
        "missed_edit": missed_edit,
        "wrong_edit": wrong_edit,
        "acquisitions": int(trajectory.metadata["acquisition_count"]),
        "steps": len(trajectory.transitions),
        "spent_cost": float(terminal.observation.spent_cost),
        "tool_calls": len(tool_transitions),
        "tool_cost": float(sum(result.cost for result in environment.tool_history)),
        "tool_belief_l1_delta": float(tool_belief_delta),
        # This is a frozen-teacher proxy, not an independent map-quality measurement.
        "quality_gain": float(quality_gain),
        "quality_cost_utility": float(quality_gain - environment.spent_penalty),
        "episode_utility_v2_proxy": proxy_utilities,
        "episode_utility_v2_proxy_balanced": proxy_utilities["balanced"]["value"],
        "episode_utility_v2_proxy_safety": proxy_utilities["safety"]["value"],
        "episode_utility_v2_proxy_cost_aware": proxy_utilities["cost_aware"]["value"],
        "selected_evidence_ids": list(environment.selected),
        "model_action_count": len(events),
        "valid_action_count": sum(event["valid_action"] for event in events),
        "fallback_count": sum(not event["valid_action"] for event in events),
        "events": events,
        "test_assets_read": sample.split == "test",
    }


def grouped_bootstrap(
    rows: list[dict[str, Any]], repetitions: int, seed: int
) -> dict[str, Any]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["aoi_id"])].append(row)
    if len(groups) < 2 or repetitions <= 0:
        raise ValueError("AOI bootstrap requires multiple AOIs and repetitions")
    names = (
        "terminal_accuracy",
        "false_edit_rate",
        "mean_cost",
        "mean_quality_gain",
        "mean_quality_cost_utility",
        "mean_episode_utility_v2_proxy_balanced",
        "mean_episode_utility_v2_proxy_safety",
        "mean_episode_utility_v2_proxy_cost_aware",
    )
    observed = metrics(rows)
    samples = {name: [] for name in names}
    rng = random.Random(seed)
    group_ids = sorted(groups)
    for _ in range(repetitions):
        selected = rng.choices(group_ids, k=len(group_ids))
        value = metrics([row for group in selected for row in groups[group]])
        for name in names:
            samples[name].append(value[name])
    return {
        "group_key": "aoi_id",
        "group_count": len(group_ids),
        "repetitions": repetitions,
        "intervals": {
            name: {
                "observed": observed[name],
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
            }
            for name, values in samples.items()
        },
    }


class QwenActiveCatalogPolicy:
    def __init__(
        self,
        processor: Any,
        model: Any,
        device: str,
        sample: SelectorSample,
        episode: EpisodeRecord,
        image_path: Path,
        *,
        policy_snapshot: str,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> None:
        self.processor = processor
        self.model = model
        self.device = device
        self.sample = sample
        self.episode = episode
        self.image_path = image_path
        self.policy_snapshot = policy_snapshot
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.events: list[dict[str, Any]] = []

    def generate_raw(self, state: dict[str, Any], system_prompt: str) -> str:
        import torch
        from PIL import Image

        image = Image.open(self.image_path).convert("RGB")
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": json.dumps(state, separators=(",", ":"))},
                ],
            },
        ]
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.device)
        generation = {
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
        }
        if self.do_sample:
            generation.update({"temperature": self.temperature, "top_p": self.top_p})
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                **generation,
            )
        continuation = generated[:, encoded["input_ids"].shape[1] :]
        return self.processor.tokenizer.batch_decode(
            continuation, skip_special_tokens=True
        )[0]

    def act(self, observation: AgentObservation) -> AgentAction:
        state = active_catalog_state_from_observation(
            observation,
            self.sample,
            self.episode,
            policy_snapshot=self.policy_snapshot,
        )
        raw = self.generate_raw(state, ACTIVE_CATALOG_SYSTEM_PROMPT)
        error = None
        selector_action = {"stage": "SELECT", "selection": "STOP"}
        try:
            selection = SequentialControllerAction.model_validate(
                _extract_json_object(raw)
            )
            candidate_ids = {item.evidence_id for item in observation.candidates}
            if selection.stage != ControllerStage.SELECT:
                raise ValueError("generated action is not SELECT")
            if (
                selection.selection == SelectionDecision.ACQUIRE
                and selection.evidence_id not in candidate_ids
            ):
                raise ValueError("generated evidence_id is unavailable")
            action = (
                AgentAction(
                    action=AgentActionType.ACQUIRE,
                    evidence_id=selection.evidence_id,
                )
                if selection.selection == SelectionDecision.ACQUIRE
                else terminal_agent_action(observation)
            )
            selector_action = selection.model_dump(mode="json", exclude_none=True)
        except Exception as exception:
            error = str(exception)
            action = terminal_agent_action(observation)
        self.events.append(
            {
                "step": observation.step,
                "raw_output": raw,
                "parse_error": error,
                "valid_action": error is None,
                "executed_action": action.model_dump(mode="json", exclude_none=True),
                "selector_action": selector_action,
                "observable_state": state,
            }
        )
        return action


class EpsilonExploreSelector:
    """Collect executed counterfactual actions around a frozen selector policy."""

    def __init__(
        self,
        selector: Any,
        *,
        epsilon: float,
        rng: random.Random,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("selector epsilon must be in [0, 1]")
        self.selector = selector
        self.epsilon = epsilon
        self.rng = rng

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.selector.events

    def act(self, observation: AgentObservation) -> AgentAction:
        model_action = self.selector.act(observation)
        event = self.events[-1]
        event["exploration_override"] = False
        event["behavior_policy"] = "epsilon_mixture"
        event["selector_epsilon"] = self.epsilon
        if (
            not event.get("valid_action")
            or self.epsilon <= 0.0
            or self.rng.random() >= self.epsilon
        ):
            return model_action

        alternatives = [terminal_agent_action(observation)]
        alternatives.extend(
            AgentAction(
                action=AgentActionType.ACQUIRE,
                evidence_id=candidate.evidence_id,
            )
            for candidate in observation.candidates
        )
        alternatives = [
            action
            for action in alternatives
            if (
                action.action != model_action.action
                or action.evidence_id != model_action.evidence_id
            )
        ]
        if not alternatives:
            return model_action

        executed = self.rng.choice(alternatives)
        event["model_selector_action"] = event["selector_action"]
        event["model_executed_action"] = event["executed_action"]
        event["selector_action"] = (
            {"stage": "SELECT", "selection": "STOP"}
            if executed.action != AgentActionType.ACQUIRE
            else {
                "stage": "SELECT",
                "selection": "ACQUIRE",
                "evidence_id": executed.evidence_id,
            }
        )
        event["executed_action"] = executed.model_dump(mode="json", exclude_none=True)
        event["exploration_override"] = True
        return executed


class QwenReactPolicy(QwenActiveCatalogPolicy):
    """Clean-room ReAct-style observation/reason/action controller."""

    system_prompt = REACT_SYSTEM_PROMPT
    controller_stage = "REACT"
    allowed_stages = frozenset({"REACT", "SELECT"})
    allowed_tools: frozenset[GeoToolName] | None = None
    call_prefix = "react"
    protocol_name = "react"
    legacy_action_tokens = frozenset(
        {"STOP", "ACCEPT", "REEXECUTE", "ACQUIRE", "USE_TOOL"}
    )

    def __init__(
        self,
        *args: Any,
        tool_costs: dict[str, float],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tool_costs = dict(tool_costs)

    @staticmethod
    def _tool_observations(observation: AgentObservation) -> list[dict[str, Any]]:
        return [
            {
                "call_id": result.call_id,
                "tool": result.tool.value,
                "success": result.success,
                "outputs": result.outputs,
                "cost": result.cost,
                "error": result.error,
            }
            for result in observation.tool_history[-4:]
        ]

    def _normalize_protocol_payload(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], str]:
        normalized = dict(payload)
        raw_stage = str(
            normalized.get("stage", self.controller_stage)
        ).strip().upper()
        compound = [part.strip() for part in raw_stage.split("|")]
        if (
            len(compound) == 2
            and compound[0] in self.allowed_stages
            and compound[1] in self.legacy_action_tokens
            and "action" not in normalized
            and "selection" not in normalized
        ):
            normalized["stage"] = compound[0]
            normalized["action"] = compound[1]
            normalized["legacy_compound_stage_action"] = raw_stage
            return normalized, "compound_stage_action"
        if "action" in normalized:
            serialization = "action"
        elif "selection" in normalized:
            serialization = "selection"
        else:
            serialization = "missing"
            reason = str(normalized.get("reason", "")).strip().upper()
            stage = str(
                normalized.get("stage", self.controller_stage)
            ).strip().upper()
            if reason in self.legacy_action_tokens:
                normalized["action"] = reason
                serialization = "reason_action"
            elif stage in self.legacy_action_tokens:
                normalized["action"] = stage
                serialization = "stage_action"
        raw_stage = str(
            normalized.get("stage", self.controller_stage)
        ).strip().upper()
        if raw_stage not in self.allowed_stages and raw_stage in self.legacy_action_tokens:
            normalized["legacy_stage_action"] = raw_stage
            normalized["stage"] = self.controller_stage
            if serialization == "missing":
                normalized["action"] = raw_stage
                serialization = "stage_action"
            else:
                serialization += "_terminal_stage"
        return normalized, serialization

    def act(self, observation: AgentObservation) -> AgentAction:
        state = active_catalog_state_from_observation(
            observation,
            self.sample,
            self.episode,
            policy_snapshot=self.policy_snapshot,
        )
        state["controller_stage"] = self.controller_stage
        state["available_tools"] = [
            {
                "tool": tool.value,
                "cost": self.tool_costs[tool.value],
            }
            for tool in observation.available_tools
            if self.allowed_tools is None or tool in self.allowed_tools
        ]
        state["tool_observations"] = self._tool_observations(observation)
        raw = self.generate_raw(state, self.system_prompt)
        error = None
        parsed: dict[str, Any] = {"action": "STOP"}
        raw_parsed: dict[str, Any] = {}
        serialization = "fallback"
        try:
            raw_parsed = _extract_json_object(raw)
            parsed, serialization = self._normalize_protocol_payload(raw_parsed)
            stage = str(parsed.get("stage", self.controller_stage)).upper()
            if stage not in self.allowed_stages:
                raise ValueError(f"unsupported {self.protocol_name} stage: {stage}")
            # The shared SFT adapter was trained with `selection`, while a
            # generic ReAct prompt uses `action`. Normalize equivalent
            # serialization before applying executable action checks.
            decision = str(
                parsed.get("action", parsed.get("selection", ""))
            ).upper()
            if decision == "ACCEPT":
                decision = "STOP"
            elif decision == "REEXECUTE":
                decision = "USE_TOOL" if parsed.get("tool") else "ACQUIRE"
            if decision == "STOP":
                action = terminal_agent_action(observation)
            elif decision == "ACQUIRE":
                evidence_id = str(parsed.get("evidence_id", ""))
                candidate_ids = {item.evidence_id for item in observation.candidates}
                if evidence_id not in candidate_ids:
                    raise ValueError(
                        f"{self.protocol_name} selected an unavailable evidence_id"
                    )
                action = AgentAction(
                    action=AgentActionType.ACQUIRE,
                    evidence_id=evidence_id,
                )
            elif decision == "USE_TOOL":
                tool = GeoToolName(str(parsed.get("tool", "")).upper())
                if tool not in observation.available_tools:
                    raise ValueError(f"{self.protocol_name} selected an unavailable tool")
                if self.allowed_tools is not None and tool not in self.allowed_tools:
                    raise ValueError(f"{self.protocol_name} selected a disallowed tool")
                evidence_id = str(parsed.get("evidence_id", ""))
                if evidence_id not in observation.selected_evidence_ids:
                    raise ValueError(
                        f"{self.protocol_name} tool call requires acquired evidence"
                    )
                identity = (
                    f"{observation.task_id}|{observation.step}|"
                    f"{tool.value}|{evidence_id}|{len(observation.tool_history)}"
                )
                action = AgentAction(
                    action=AgentActionType.USE_TOOL,
                    tool_call=GeoToolCall(
                        call_id=(
                            self.call_prefix
                            + "-"
                            + hashlib.sha256(identity.encode()).hexdigest()[:20]
                        ),
                        tool=tool,
                        inputs={"evidence_id": evidence_id},
                    ),
                )
            else:
                raise ValueError(f"unsupported {self.protocol_name} action: {decision}")
        except Exception as exception:
            error = str(exception)
            action = terminal_agent_action(observation)
        self.events.append(
            {
                "step": observation.step,
                "raw_output": raw,
                "parse_error": error,
                "valid_action": error is None,
                "executed_action": action.model_dump(mode="json", exclude_none=True),
                "react_action": parsed,
                "protocol_action": parsed,
                "protocol_action_raw": raw_parsed,
                "controller_protocol": self.protocol_name,
                "react_action_serialization": serialization,
                "observable_state": state,
            }
        )
        return action


class QwenGeoMMAgentPolicy(QwenReactPolicy):
    """Clean-room GeoMMAgent-style plan/execute/self-evaluate coordinator."""

    system_prompt = GEOMMAGENT_SYSTEM_PROMPT
    controller_stage = "PLAN"
    allowed_stages = frozenset(
        {"PLAN", "EXECUTE", "SELF_EVALUATE", "SELECT", "REACT"}
    )
    call_prefix = "geommagent"
    protocol_name = "geommagent_style"


class QwenSenseSearchPolicy(QwenReactPolicy):
    """Clean-room SenseSearch-style iterative image search and crop policy."""

    system_prompt = SENSESEARCH_SYSTEM_PROMPT
    controller_stage = "SEARCH"
    allowed_stages = frozenset({"SEARCH", "REASON", "SELECT", "REACT"})
    allowed_tools = frozenset(
        {
            GeoToolName.RASTER_CROP,
            GeoToolName.RASTER_SEGMENT,
            GeoToolName.VECTOR_INSPECT,
        }
    )
    call_prefix = "sensesearch"
    protocol_name = "sensesearch_style"


class QwenPlanThenExecutePolicy(QwenActiveCatalogPolicy):
    """Plan evidence once, then execute without model replanning."""

    def __init__(self, *args: Any, max_plan_acquisitions: int = 2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_plan_acquisitions = int(max_plan_acquisitions)
        self.plan: list[str] | None = None

    def _planned_action(self, observation: AgentObservation) -> AgentAction:
        assert self.plan is not None
        if not self.plan:
            return terminal_agent_action(observation)
        evidence_id = self.plan.pop(0)
        if evidence_id not in {item.evidence_id for item in observation.candidates}:
            raise ValueError("planned evidence is no longer available")
        return AgentAction(action=AgentActionType.ACQUIRE, evidence_id=evidence_id)

    def act(self, observation: AgentObservation) -> AgentAction:
        if self.plan is not None:
            error = None
            try:
                action = self._planned_action(observation)
            except Exception as exception:
                error = str(exception)
                self.plan.clear()
                action = terminal_agent_action(observation)
            self.events.append(
                {
                    "step": observation.step,
                    "stage": "EXECUTE_OPEN_LOOP_PLAN",
                    "parse_error": error,
                    "valid_action": error is None,
                    "executed_action": action.model_dump(mode="json", exclude_none=True),
                    "remaining_plan": list(self.plan),
                }
            )
            return action

        state = active_catalog_state_from_observation(
            observation,
            self.sample,
            self.episode,
            policy_snapshot=self.policy_snapshot,
        )
        raw = self.generate_raw(state, PLAN_EXECUTE_SYSTEM_PROMPT)
        error = None
        try:
            payload = _extract_json_object(raw)
            planned = payload.get("evidence_ids")
            if planned is None and str(payload.get("selection", "")).upper() == "ACQUIRE":
                planned = [payload.get("evidence_id")]
            if planned is None and str(payload.get("selection", "")).upper() == "STOP":
                planned = []
            if not isinstance(planned, list):
                raise ValueError("plan must contain evidence_ids")
            self.plan = [str(value) for value in planned]
            if len(self.plan) > self.max_plan_acquisitions:
                raise ValueError("plan exceeds the acquisition limit")
            if len(self.plan) != len(set(self.plan)):
                raise ValueError("plan repeats an evidence identifier")
            candidates = {item.evidence_id: item for item in observation.candidates}
            if any(evidence_id not in candidates for evidence_id in self.plan):
                raise ValueError("plan contains unavailable evidence")
            if (
                sum(candidates[evidence_id].cost for evidence_id in self.plan)
                > observation.remaining_budget + 1e-8
            ):
                raise ValueError("plan exceeds the remaining budget")
            action = self._planned_action(observation)
        except Exception as exception:
            error = str(exception)
            self.plan = []
            action = terminal_agent_action(observation)
        self.events.append(
            {
                "step": observation.step,
                "stage": "PLAN_OPEN_LOOP",
                "raw_output": raw,
                "parse_error": error,
                "valid_action": error is None,
                "executed_action": action.model_dump(mode="json", exclude_none=True),
                "remaining_plan": list(self.plan),
                "observable_state": state,
            }
        )
        return action


class AlwaysStopPolicy:
    """Nonparametric control that executes the current belief immediately."""

    def __init__(
        self,
        _processor: Any,
        _model: Any,
        _device: str,
        sample: SelectorSample,
        episode: EpisodeRecord,
        _image_path: Path,
        *,
        policy_snapshot: str,
        **_kwargs: Any,
    ) -> None:
        self.sample = sample
        self.episode = episode
        self.policy_snapshot = policy_snapshot
        self.events: list[dict[str, Any]] = []

    def act(self, observation: AgentObservation) -> AgentAction:
        state = active_catalog_state_from_observation(
            observation,
            self.sample,
            self.episode,
            policy_snapshot=self.policy_snapshot,
        )
        action = terminal_agent_action(observation)
        self.events.append(
            {
                "step": observation.step,
                "raw_output": None,
                "parse_error": None,
                "valid_action": True,
                "executed_action": action.model_dump(
                    mode="json", exclude_none=True
                ),
                "selector_action": {
                    "stage": ControllerStage.SELECT.value,
                    "selection": SelectionDecision.STOP.value,
                },
                "observable_state": state,
            }
        )
        return action


class QwenGateRankerPolicy(QwenActiveCatalogPolicy):
    """Compose a visual ACQUIRE/STOP gate with a frozen observable ranker."""

    def __init__(self, *args: Any, ranker: CandidateUtilityRankerPredictor, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ranker = ranker

    def act(self, observation: AgentObservation) -> AgentAction:
        state = active_catalog_state_from_observation(
            observation,
            self.sample,
            self.episode,
            policy_snapshot=self.policy_snapshot,
        )
        raw = self.generate_raw(state, GATE_SYSTEM_PROMPT)
        error = None
        selection = "STOP"
        scores: dict[str, float] = {}
        evidence_id = None
        try:
            selection = parse_gate_output(raw)
            scores = self.ranker.score_state(state)
            if selection == "ACQUIRE":
                if not scores:
                    raise ValueError("gate requested ACQUIRE without an affordable candidate")
                evidence_id = max(scores, key=lambda value: (scores[value], value))
                candidate_ids = {item.evidence_id for item in observation.candidates}
                if evidence_id not in candidate_ids:
                    raise ValueError("ranker selected an unavailable evidence item")
                action = AgentAction(
                    action=AgentActionType.ACQUIRE,
                    evidence_id=evidence_id,
                )
            else:
                action = terminal_agent_action(observation)
        except Exception as exception:
            error = str(exception)
            action = terminal_agent_action(observation)
        self.events.append(
            {
                "step": observation.step,
                "raw_output": raw,
                "parse_error": error,
                "valid_action": error is None,
                "executed_action": action.model_dump(mode="json", exclude_none=True),
                "selector_action": {
                    "stage": "SELECT",
                    "selection": selection,
                    "evidence_id": evidence_id,
                },
                "ranker_scores": scores,
                "ranker_safety_margin_diagnostic_only": self.ranker.safety_margin,
                "observable_state": state,
            }
        )
        return action


class QwenResidualRankerPolicy(QwenActiveCatalogPolicy):
    """Select evidence using current-policy residual utility predictions."""

    def __init__(self, *args: Any, ranker: CandidateUtilityRankerPredictor, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ranker = ranker

    def act(self, observation: AgentObservation) -> AgentAction:
        state = active_catalog_state_from_observation(
            observation,
            self.sample,
            self.episode,
            policy_snapshot=self.policy_snapshot,
        )
        selection, evidence_id, score = self.ranker.decide(state)
        scores = self.ranker.score_state(state)
        if selection == "ACQUIRE" and evidence_id is not None:
            action = AgentAction(action=AgentActionType.ACQUIRE, evidence_id=evidence_id)
        else:
            action = terminal_agent_action(observation)
        self.events.append(
            {
                "step": observation.step,
                "raw_output": None,
                "parse_error": None,
                "valid_action": True,
                "executed_action": action.model_dump(mode="json", exclude_none=True),
                "selector_action": {
                    "stage": "SELECT",
                    "selection": selection,
                    "evidence_id": evidence_id,
                },
                "predicted_residual_utility": score,
                "residual_safety_margin": self.ranker.safety_margin,
                "ranker_scores": scores,
                "observable_state": state,
            }
        )
        return action


class QwenUtilityHeadRankerPolicy(QwenActiveCatalogPolicy):
    """Use a frozen VLM state and calibrated utility head for structured control."""

    def __init__(
        self,
        *args: Any,
        ranker: CandidateUtilityRankerPredictor,
        utility_head: Any,
        utility_threshold: float,
        utility_pooling: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ranker = ranker
        self.utility_head = utility_head
        self.utility_threshold = utility_threshold
        self.utility_pooling = utility_pooling

    def predict_utility(self, state: dict[str, Any]) -> tuple[float, Any]:
        import torch
        from PIL import Image

        from activemap.agent.visual_gate import pool_prompt_state

        image = Image.open(self.image_path).convert("RGB")
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": ACTIVE_CATALOG_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": json.dumps(state, separators=(",", ":"))},
                ],
            },
        ]
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.device)
        with torch.inference_mode():
            output = self.model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            pooled = pool_prompt_state(
                output.hidden_states[-1],
                encoded["attention_mask"],
                mode=self.utility_pooling,
            )
        state_embedding = pooled.float().cpu().numpy()[0]
        return (
            float(self.utility_head.predict(state_embedding.reshape(1, -1))[0]),
            state_embedding,
        )

    def act(self, observation: AgentObservation) -> AgentAction:
        state = active_catalog_state_from_observation(
            observation,
            self.sample,
            self.episode,
            policy_snapshot=self.policy_snapshot,
        )
        predicted_utility, state_embedding = self.predict_utility(state)
        scores = self.ranker.score_state(state, state_embedding)
        evidence_id = None
        error = None
        selection = "STOP"
        try:
            if predicted_utility >= self.utility_threshold:
                if not scores:
                    raise ValueError("utility head requested ACQUIRE without a candidate")
                evidence_id = max(scores, key=lambda value: (scores[value], value))
                candidate_ids = {item.evidence_id for item in observation.candidates}
                if evidence_id not in candidate_ids:
                    raise ValueError("ranker selected an unavailable evidence item")
                selection = "ACQUIRE"
                action = AgentAction(
                    action=AgentActionType.ACQUIRE,
                    evidence_id=evidence_id,
                )
            else:
                action = terminal_agent_action(observation)
        except Exception as exception:
            error = str(exception)
            action = terminal_agent_action(observation)
        self.events.append(
            {
                "step": observation.step,
                "raw_output": None,
                "parse_error": error,
                "valid_action": error is None,
                "executed_action": action.model_dump(mode="json", exclude_none=True),
                "selector_action": {
                    "stage": "SELECT",
                    "selection": selection,
                    "evidence_id": evidence_id,
                },
                "predicted_acquire_utility": predicted_utility,
                "utility_threshold": self.utility_threshold,
                "utility_pooling": self.utility_pooling,
                "ranker_scores": scores,
                "observable_state": state,
            }
        )
        return action


class QwenHybridResidualRankerPolicy(QwenUtilityHeadRankerPolicy):
    """Fuse frozen VLM utility context with current-policy residual ranking."""

    def act(self, observation: AgentObservation) -> AgentAction:
        import numpy as np

        state = active_catalog_state_from_observation(
            observation,
            self.sample,
            self.episode,
            policy_snapshot=self.policy_snapshot,
        )
        predicted_utility, _ = self.predict_utility(state)
        scores = self.ranker.score_state(
            state, np.asarray([predicted_utility], dtype=np.float32)
        )
        evidence_id = max(scores, key=lambda value: (scores[value], value)) if scores else None
        score = scores[evidence_id] if evidence_id is not None else 0.0
        selection = "ACQUIRE" if evidence_id is not None and score > self.ranker.safety_margin else "STOP"
        if selection == "ACQUIRE":
            action = AgentAction(action=AgentActionType.ACQUIRE, evidence_id=evidence_id)
        else:
            action = terminal_agent_action(observation)
        self.events.append(
            {
                "step": observation.step,
                "raw_output": None,
                "parse_error": None,
                "valid_action": True,
                "executed_action": action.model_dump(mode="json", exclude_none=True),
                "selector_action": {
                    "stage": "SELECT",
                    "selection": selection,
                    "evidence_id": evidence_id if selection == "ACQUIRE" else None,
                },
                "predicted_acquire_utility": predicted_utility,
                "predicted_residual_utility": score,
                "residual_safety_margin": self.ranker.safety_margin,
                "ranker_scores": scores,
                "observable_state": state,
            }
        )
        return action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("states", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("val_sft", type=Path)
    parser.add_argument("val_evaluation_index", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--max-acquisitions", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--selector-epsilon", type=float, default=0.0)
    parser.add_argument(
        "--policy-mode",
        choices=(
            "full_action",
            "react",
            "geommagent_style",
            "sensesearch_style",
            "always_stop",
            "plan_execute",
            "gate_ranker",
            "utility_head_ranker",
            "residual_ranker",
            "hybrid_residual_ranker",
        ),
        default="full_action",
    )
    parser.add_argument("--ranker-checkpoint", type=Path)
    parser.add_argument("--utility-head", type=Path)
    parser.add_argument("--utility-head-summary", type=Path)
    parser.add_argument("--utility-threshold-override", type=float)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--frozen-test", action="store_true")
    parser.add_argument(
        "--tool-mode",
        choices=("none", "forced", "selective", "model"),
        default="none",
    )
    parser.add_argument(
        "--belief-mode",
        choices=("recurrent", "frozen_prior", "identity"),
        default="recurrent",
    )
    parser.add_argument("--tool-belief-checkpoint", type=Path)
    parser.add_argument("--tool-gate", type=Path)
    parser.add_argument("--tool-gate-summary", type=Path)
    parser.add_argument("--tool-artifact-root", type=Path)
    parser.add_argument("--tool-out-size", type=int, default=256)
    parser.add_argument(
        "--asset-root-map",
        action="append",
        type=parse_root_map,
        default=[],
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    test_assets_read = args.split == "test"
    if test_assets_read:
        if not args.frozen_test:
            raise PermissionError("test evaluation requires --frozen-test")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    elif args.frozen_test:
        raise ValueError("--frozen-test is valid only for the test split")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.max_candidates <= 0 or args.max_acquisitions < 0:
        raise ValueError("invalid closed-loop limits")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid evaluation shard")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("invalid generation sampling settings")
    if not 0.0 <= args.selector_epsilon <= 1.0:
        raise ValueError("selector epsilon must be in [0, 1]")
    if args.selector_epsilon > 0.0 and args.split == "test":
        raise ValueError("selector exploration is forbidden on test")
    if args.policy_mode in {"gate_ranker", "utility_head_ranker", "residual_ranker", "hybrid_residual_ranker"} and args.ranker_checkpoint is None:
        raise ValueError(f"{args.policy_mode} policy requires --ranker-checkpoint")
    if args.policy_mode in {"utility_head_ranker", "hybrid_residual_ranker"} and (
        args.utility_head is None or args.utility_head_summary is None
    ):
        raise ValueError("utility_head_ranker requires its head and summary")
    if args.policy_mode not in {"utility_head_ranker", "hybrid_residual_ranker"} and (
        args.utility_head is not None
        or args.utility_head_summary is not None
        or args.utility_threshold_override is not None
    ):
        raise ValueError("utility-head arguments require utility_head_ranker")
    if args.policy_mode == "full_action" and args.ranker_checkpoint is not None:
        raise ValueError("--ranker-checkpoint requires a structured ranker policy")
    model_tool_policies = {"react", "geommagent_style", "sensesearch_style"}
    if (args.policy_mode in model_tool_policies) != (args.tool_mode == "model"):
        raise ValueError(
            "ReAct/GeoMMAgent/SenseSearch policies require --tool-mode model, "
            "and model tool mode is restricted to those policies"
        )
    if args.policy_mode == "always_stop" and args.tool_mode != "none":
        raise ValueError("always_stop is a no-tool control")
    if args.belief_mode != "recurrent" and args.tool_mode == "none":
        raise ValueError("belief ablations require tool use")

    import torch
    from peft import PeftModel
    from tqdm.auto import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.manual_seed(args.seed)
    exploration_rng = random.Random(args.seed ^ 0xA17ECA7)

    samples = load_initial_samples(
        args.states,
        split=args.split,
        limit=args.limit,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    episodes = load_episodes(args.episodes, split=args.split)
    images = image_index(args.val_sft, args.val_evaluation_index, split=args.split)
    missing = sorted(
        {
            str(sample.metadata["source_episode"])
            for sample in samples
            if str(sample.metadata["source_episode"]) not in episodes
            or str(sample.metadata["source_episode"]) not in images
        }
    )
    if missing:
        raise ValueError(f"missing episode/image assets: {missing[0]}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()
    ranker = (
        CandidateUtilityRankerPredictor(str(args.ranker_checkpoint), "cpu")
        if args.ranker_checkpoint is not None
        else None
    )
    utility_head = None
    utility_head_summary = None
    utility_threshold = 0.0
    utility_pooling = "last"
    if args.utility_head is not None and args.utility_head_summary is not None:
        from joblib import load

        utility_head = load(args.utility_head)
        utility_head_summary = json.loads(
            args.utility_head_summary.read_text(encoding="utf-8")
        )
        utility_threshold = float(utility_head_summary["selected"]["threshold"])
        if args.utility_threshold_override is not None:
            utility_threshold = args.utility_threshold_override
        train_features = Path(utility_head_summary["sources"]["train"]["path"])
        feature_summary_path = train_features / "summary.json"
        feature_summary = json.loads(feature_summary_path.read_text(encoding="utf-8"))
        if _sha256(feature_summary_path) != utility_head_summary["sources"]["train"]["summary_sha256"]:
            raise ValueError("utility-head feature summary hash mismatch")
        utility_pooling = str(feature_summary["pooling"])
    gate = None
    gate_threshold = 0.0
    paired_updater = None
    tool_registry = None
    sequential_updater_class = None
    if args.tool_mode != "none":
        if args.tool_belief_checkpoint is None or args.tool_artifact_root is None:
            raise ValueError("tool modes require a Tool-Belief checkpoint and artifact root")
        from activemap.agent.tool_belief_model import (
            FrozenPriorSequentialPairedToolBeliefUpdater,
            IdentitySequentialPairedToolBeliefUpdater,
            PairedToolBeliefUpdater,
            SequentialPairedToolBeliefUpdater,
        )
        paired_updater = PairedToolBeliefUpdater.from_checkpoint(
            args.tool_belief_checkpoint, device="cpu"
        )
        sequential_updater_class = {
            "recurrent": SequentialPairedToolBeliefUpdater,
            "frozen_prior": FrozenPriorSequentialPairedToolBeliefUpdater,
            "identity": IdentitySequentialPairedToolBeliefUpdater,
        }[args.belief_mode]
        tool_registry = build_model_tool_registry(
            args.policy_mode, args.tool_artifact_root
        )
        if args.tool_mode == "selective":
            if args.tool_gate is None or args.tool_gate_summary is None:
                raise ValueError("selective mode requires a frozen gate and summary")
            gate_summary = json.loads(args.tool_gate_summary.read_text(encoding="utf-8"))
            if gate_summary.get("schema_version") == "uncertainty-tool-gate-summary-v1":
                from activemap.agent.active_catalog_tool_gate import BeliefUncertaintyGate

                gate = BeliefUncertaintyGate()
            else:
                from joblib import load

                gate = load(args.tool_gate)
            gate_threshold = float(gate_summary["selected"]["threshold"])
    args.output_dir.mkdir(parents=True)
    trace_path = args.output_dir / "traces.jsonl"
    transition_path = args.output_dir / "joint_transitions.jsonl"
    rows = []
    transition_count = 0
    with (
        trace_path.open("x", encoding="utf-8", buffering=1) as trace_file,
        transition_path.open("x", encoding="utf-8", buffering=1) as transition_file,
    ):
        for sample in tqdm(samples, desc="Active-catalog closed loop"):
            episode_id = str(sample.metadata["source_episode"])
            environment = MapMaintenanceEnv(
                sample,
                budget=float(sample.metadata["budget"]),
                score_fn=lambda current: observable_shortlist_scores(
                    current, args.max_candidates
                ),
                belief_updater=CounterfactualBeliefUpdater(sample),
                top_k=args.max_candidates,
                public_identifiers=True,
                tool_registry=tool_registry,
                tool_belief_updater=(
                    sequential_updater_class(paired_updater)
                    if paired_updater is not None and sequential_updater_class is not None
                    else None
                ),
                asset_paths={
                    item.evidence_id: remap_asset_path(
                        item.image_path, args.asset_root_map
                    )
                    for item in episodes[episode_id].evidence_catalog
                },
                tool_parameters={
                    item.evidence_id: {
                        "pixel_window": [
                            item.region[0], item.region[1],
                            item.region[2] - item.region[0],
                            item.region[3] - item.region[1],
                        ],
                        "out_size": [args.tool_out_size, args.tool_out_size],
                    }
                    for item in episodes[episode_id].evidence_catalog
                },
                tool_inputs_by_evidence=(
                    {
                        item.evidence_id: {
                            "vector_path": remap_asset_path(
                                episodes[episode_id].map_before,
                                args.asset_root_map,
                            )
                        }
                        for item in episodes[episode_id].evidence_catalog
                    }
                    if args.policy_mode == "sensesearch_style"
                    else None
                ),
            )
            policy_class = {
                "full_action": QwenActiveCatalogPolicy,
                "react": QwenReactPolicy,
                "geommagent_style": QwenGeoMMAgentPolicy,
                "sensesearch_style": QwenSenseSearchPolicy,
                "always_stop": AlwaysStopPolicy,
                "plan_execute": QwenPlanThenExecutePolicy,
                "gate_ranker": QwenGateRankerPolicy,
                "utility_head_ranker": QwenUtilityHeadRankerPolicy,
                "residual_ranker": QwenResidualRankerPolicy,
                "hybrid_residual_ranker": QwenHybridResidualRankerPolicy,
            }[args.policy_mode]
            policy_kwargs: dict[str, Any] = {}
            if ranker is not None:
                policy_kwargs["ranker"] = ranker
            if args.policy_mode in {"utility_head_ranker", "hybrid_residual_ranker"}:
                policy_kwargs.update(
                    {
                        "utility_head": utility_head,
                        "utility_threshold": utility_threshold,
                        "utility_pooling": utility_pooling,
                    }
                )
            if args.policy_mode in model_tool_policies:
                assert tool_registry is not None
                policy_kwargs["tool_costs"] = {
                    tool.value: tool_registry.cost(tool)
                    for tool in tool_registry.names()
                }
            if args.policy_mode == "plan_execute":
                policy_kwargs["max_plan_acquisitions"] = args.max_acquisitions
            selector_policy = policy_class(
                processor,
                model,
                args.device,
                sample,
                episodes[episode_id],
                images[episode_id],
                policy_snapshot=str(args.adapter.resolve()),
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                **policy_kwargs,
            )
            if args.selector_epsilon > 0.0:
                selector_policy = EpsilonExploreSelector(
                    selector_policy,
                    epsilon=args.selector_epsilon,
                    rng=exploration_rng,
                )
            policy = (
                SelectiveToolPolicy(
                    selector_policy,
                    mode=args.tool_mode,
                    gate=gate,
                    threshold=gate_threshold,
                )
                if args.tool_mode in {"forced", "selective"}
                else selector_policy
            )
            trajectory = rollout_agent_policy(
                environment,
                policy,
                max_acquisitions=args.max_acquisitions,
                max_tool_calls=(
                    0 if args.tool_mode == "none" else 2 * args.max_acquisitions
                ),
                max_steps=(
                    args.max_acquisitions + 1
                    if args.tool_mode == "none"
                    else 3 * args.max_acquisitions + 1
                ),
            )
            row = trajectory_row(
                sample,
                environment,
                trajectory,
                policy_name={
                    "full_action": "qwen_active_catalog",
                    "react": "react_style_qwen",
                    "geommagent_style": "geommagent_style_qwen",
                    "sensesearch_style": "sensesearch_style_qwen",
                    "always_stop": "always_stop",
                    "plan_execute": "qwen_open_loop_plan_execute",
                    "gate_ranker": "qwen_gate_ranker",
                    "utility_head_ranker": "qwen_utility_head_ranker",
                    "residual_ranker": "current_policy_residual_ranker",
                    "hybrid_residual_ranker": "qwen_hybrid_current_policy_residual_ranker",
                }[args.policy_mode],
                events=policy.events,
            )
            rows.append(row)
            trace_file.write(json.dumps(row, separators=(",", ":")) + "\n")
            for transition in trajectory.transitions:
                exported = joint_transition_from_agent_transition(
                    transition,
                    source_episode=episode_id,
                    aoi_id=str(sample.metadata["aoi_id"]),
                    policy_snapshot=str(args.adapter.resolve()),
                    target_edit=EditOperation(str(sample.metadata["gt_edit"])),
                    selected_by_model=True,
                    operation_update_threshold=float(
                        sample.metadata["operation_update_threshold"]
                    ),
                )
                if exported is not None:
                    transition_file.write(exported.model_dump_json() + "\n")
                    transition_count += 1
    summary = {
        "schema_version": "active-catalog-closed-loop-evaluation-v1",
        "evaluation_role": (
            "nonparametric_control"
            if args.policy_mode == "always_stop"
            else (
                "model_driven_open_loop_plan_execute"
                if args.policy_mode == "plan_execute"
                else "model_driven_recurrent_evidence_acquisition"
            )
        ),
        "sample_count": len(rows),
        "shard": {"index": args.shard_index, "count": args.num_shards},
        "split": args.split,
        "belief_mode": args.belief_mode,
        "joint_transition_count": transition_count,
        "episode_count": len({row["source_episode"] for row in rows}),
        "aoi_count": len({row["aoi_id"] for row in rows}),
        "metrics": metrics(rows),
        "aoi_bootstrap": (
            grouped_bootstrap(rows, args.bootstrap_repetitions, args.seed)
            if args.bootstrap_repetitions > 0
            else None
        ),
        "protocol": {
            "model_selected_state_transitions": True,
            "oracle_next_state_replay": False,
            "joint_transition_export": "active-catalog-joint-transition-v1",
            "joint_transition_oracle_action_exported": False,
            "belief_updater": "frozen_counterfactual_updater_cumulative_fusion",
            "tool_mode": args.tool_mode,
            "policy_mode": args.policy_mode,
            "registered_tools": (
                [tool.value for tool in tool_registry.names()]
                if tool_registry is not None
                else []
            ),
            "asset_root_maps": [
                {"source": source, "target": target}
                for source, target in args.asset_root_map
            ],
            "controller_protocol": (
                (
                    "react_observation_reason_action"
                    if args.policy_mode == "react"
                    else (
                        "geommagent_plan_execute_self_evaluate"
                        if args.policy_mode == "geommagent_style"
                        else (
                            "sensesearch_iterative_search_crop_reason"
                            if args.policy_mode == "sensesearch_style"
                            else None
                        )
                    )
                )
                if args.policy_mode in model_tool_policies
                else (
                    "always_stop_control"
                    if args.policy_mode == "always_stop"
                    else "activemap_structured_control"
                )
            ),
            "react_action_parser": (
                "semantic_action_or_selection_v2"
                if args.policy_mode in model_tool_policies
                else None
            ),
            "explicit_geospatial_tool_calls": args.tool_mode != "none",
            "selective_tool_calling": args.tool_mode == "selective",
            "tool_belief_reliability_gate": (
                bool(paired_updater.model.reliability_gate)
                if paired_updater is not None
                else None
            ),
            "max_candidates": args.max_candidates,
            "max_acquisitions": args.max_acquisitions,
            "stochastic_policy_sampling": args.do_sample,
            "selector_epsilon": args.selector_epsilon,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "utility_head": (
                {
                    "threshold": utility_threshold,
                    "pooling": utility_pooling,
                    "score_type": utility_head_summary["score_type"],
                    "threshold_source": (
                        "cli_validation_calibration"
                        if args.utility_threshold_override is not None
                        else "train_grouped_oof"
                    ),
                }
                if utility_head_summary is not None
                else None
            ),
            "quality_gain_semantics": "frozen_teacher_proxy",
            "episode_utility_v2_proxy": utility_protocol(
                quality_source="terminal_operation_exactness_proxy",
                paper_primary=False,
            ),
        },
        "sources": {
            "model": str(args.model.resolve()),
            "adapter": str(args.adapter.resolve()),
            "candidate_ranker": (
                {
                    "path": str(args.ranker_checkpoint.resolve()),
                    "sha256": _sha256(args.ranker_checkpoint),
                    "safety_margin_diagnostic_only": ranker.safety_margin,
                }
                if args.ranker_checkpoint is not None and ranker is not None
                else None
            ),
            "utility_head": (
                {
                    "path": str(args.utility_head.resolve()),
                    "sha256": _sha256(args.utility_head),
                    "summary_path": str(args.utility_head_summary.resolve()),
                    "summary_sha256": _sha256(args.utility_head_summary),
                }
                if args.utility_head is not None and args.utility_head_summary is not None
                else None
            ),
            "states": str(args.states.resolve()),
            "episodes": str(args.episodes.resolve()),
            "tool_belief_checkpoint": (
                {
                    "path": str(args.tool_belief_checkpoint.resolve()),
                    "sha256": _sha256(args.tool_belief_checkpoint),
                }
                if args.tool_mode != "none"
                else None
            ),
            "tool_gate": (
                {
                    "path": str(args.tool_gate.resolve()),
                    "sha256": _sha256(args.tool_gate),
                    "summary_path": str(args.tool_gate_summary.resolve()),
                    "summary_sha256": _sha256(args.tool_gate_summary),
                }
                if args.tool_mode == "selective"
                else None
            ),
        },
        "test_assets_read": test_assets_read,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
