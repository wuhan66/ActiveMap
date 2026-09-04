#!/usr/bin/env python3
"""Diagnose SELECT collapse by scoring both valid action continuations."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from activemap.agent.sequential_controller import (
    ControllerStage,
    SelectionDecision,
    SequentialControllerAction,
)
from activemap.agent.vlm_sft import load_vlm_sft_rows
from scripts.evaluate_sequential_selector import selector_metrics, task_bootstrap


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _user_state(row: dict[str, Any]) -> dict[str, Any]:
    text = next(
        part["text"]
        for part in row["messages"][1]["content"]
        if part.get("type") == "text"
    )
    return json.loads(text)


def candidate_actions(row: dict[str, Any]) -> dict[SelectionDecision, str]:
    evidence_id = str(_user_state(row)["evidence_id"])
    return {
        SelectionDecision.STOP: SequentialControllerAction(
            stage=ControllerStage.SELECT,
            selection=SelectionDecision.STOP,
        ).model_dump_json(exclude_none=True),
        SelectionDecision.ACQUIRE: SequentialControllerAction(
            stage=ControllerStage.SELECT,
            selection=SelectionDecision.ACQUIRE,
            evidence_id=evidence_id,
        ).model_dump_json(exclude_none=True),
    }


def binary_auc(labels: list[bool], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=True) if label]
    negatives = [score for label, score in zip(labels, scores, strict=True) if not label]
    if not positives or not negatives:
        raise ValueError("AUC requires both selector classes")
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right, strict=False):
        if left_token != right_token:
            break
        length += 1
    if length >= min(len(left), len(right)):
        raise ValueError("selector candidates do not have a divergent decision token")
    return length


def _score_action(model, processor, messages, action: str, device: str) -> dict[str, float]:
    import torch

    prompt = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)
    full_messages = [
        *messages,
        {"role": "assistant", "content": [{"type": "text", "text": action}]},
    ]
    full = processor.apply_chat_template(
        full_messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=False,
    ).to(device)
    prompt_ids = prompt["input_ids"][0]
    full_ids = full["input_ids"][0]
    prompt_length = int(prompt_ids.numel())
    if full_ids.shape[0] <= prompt_length or not torch.equal(
        full_ids[:prompt_length], prompt_ids
    ):
        raise ValueError("candidate action does not preserve the generation prompt prefix")
    action_ids = processor.tokenizer(
        action, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0].to(device)
    action_length = int(action_ids.numel())
    if not torch.equal(
        full_ids[prompt_length : prompt_length + action_length], action_ids
    ):
        raise ValueError("candidate action tokenization changes at the prompt boundary")
    with torch.inference_mode():
        logits = model(**full).logits[0]
        token_logits = logits[prompt_length - 1 : prompt_length + action_length - 1]
        token_log_probabilities = torch.log_softmax(token_logits.float(), dim=-1)
        selected = token_log_probabilities.gather(1, action_ids[:, None]).squeeze(1)
    return {
        "sum_log_probability": float(selected.sum().item()),
        "mean_log_probability": float(selected.mean().item()),
        "token_count": action_length,
        "token_log_probabilities": selected.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("selector_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    import torch
    from peft import PeftModel
    from tqdm.auto import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from activemap.agent.vlm_sft import materialize_vlm_messages

    rows = load_vlm_sft_rows(args.selector_jsonl)
    if args.max_samples is not None:
        if args.max_samples < 2:
            raise ValueError("likelihood diagnostic requires at least two samples")
        rows = rows[: args.max_samples]
    if args.expected_records is not None and len(rows) != args.expected_records:
        raise ValueError(f"expected {args.expected_records} rows, found {len(rows)}")
    if {row["stage"] for row in rows} != {ControllerStage.SELECT.value}:
        raise ValueError("likelihood diagnostic contains non-SELECT records")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()
    traces = []
    for row in tqdm(rows, desc="Sequential selector likelihood diagnostic"):
        messages = materialize_vlm_messages(row["messages"][:2])
        candidates = candidate_actions(row)
        candidate_token_ids = {
            decision: processor.tokenizer(action, add_special_tokens=False)["input_ids"]
            for decision, action in candidates.items()
        }
        decision_index = common_prefix_length(
            candidate_token_ids[SelectionDecision.STOP],
            candidate_token_ids[SelectionDecision.ACQUIRE],
        )
        stop_score = _score_action(
            model, processor, messages, candidates[SelectionDecision.STOP], args.device
        )
        acquire_score = _score_action(
            model,
            processor,
            messages,
            candidates[SelectionDecision.ACQUIRE],
            args.device,
        )
        margin = (
            acquire_score["mean_log_probability"]
            - stop_score["mean_log_probability"]
        )
        decision_margin = (
            acquire_score["token_log_probabilities"][decision_index]
            - stop_score["token_log_probabilities"][decision_index]
        )
        target = SequentialControllerAction.model_validate_json(
            next(
                part["text"]
                for part in row["messages"][2]["content"]
                if part.get("type") == "text"
            )
        ).selection
        traces.append(
            {
                "trajectory_id": row["trajectory_id"],
                "task_id": row["task_id"],
                "split": row["split"],
                "target_selection": target.value,
                "predicted_selection": (
                    SelectionDecision.ACQUIRE.value
                    if decision_margin > 0.0
                    else SelectionDecision.STOP.value
                ),
                "policy_relative_advantage": float(row["policy_relative_advantage"]),
                "acquire_mean_log_probability": acquire_score["mean_log_probability"],
                "stop_mean_log_probability": stop_score["mean_log_probability"],
                "mean_log_probability_margin": margin,
                "decision_token_log_probability_margin": decision_margin,
                "decision_token_index": decision_index,
                "acquire_sum_log_probability": acquire_score["sum_log_probability"],
                "stop_sum_log_probability": stop_score["sum_log_probability"],
                "acquire_token_count": acquire_score["token_count"],
                "stop_token_count": stop_score["token_count"],
            }
        )
    labels = [row["target_selection"] == "ACQUIRE" for row in traces]
    margins = [float(row["mean_log_probability_margin"]) for row in traces]
    decision_margins = [
        float(row["decision_token_log_probability_margin"]) for row in traces
    ]
    positive_margins = [margin for label, margin in zip(labels, margins, strict=True) if label]
    negative_margins = [margin for label, margin in zip(labels, margins, strict=True) if not label]
    positive_decision_margins = [
        margin for label, margin in zip(labels, decision_margins, strict=True) if label
    ]
    negative_decision_margins = [
        margin for label, margin in zip(labels, decision_margins, strict=True) if not label
    ]
    metrics = selector_metrics(traces)
    bootstrap = task_bootstrap(
        traces, repetitions=args.bootstrap_repetitions, seed=args.seed
    )
    args.output_dir.mkdir(parents=True)
    traces_path = args.output_dir / "traces.jsonl"
    with traces_path.open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "sequential-selector-likelihood-diagnostic-v1",
        "diagnostic_only": True,
        "validation_threshold_selection_allowed": False,
        "sample_count": len(traces),
        "ranking": {
            "decision_token_margin_auc": binary_auc(labels, decision_margins),
            "positive_decision_margin_mean": statistics.fmean(
                positive_decision_margins
            ),
            "positive_decision_margin_median": statistics.median(
                positive_decision_margins
            ),
            "negative_decision_margin_mean": statistics.fmean(
                negative_decision_margins
            ),
            "negative_decision_margin_median": statistics.median(
                negative_decision_margins
            ),
            "mean_log_probability_margin_auc": binary_auc(labels, margins),
            "positive_margin_mean": statistics.fmean(positive_margins),
            "positive_margin_median": statistics.median(positive_margins),
            "negative_margin_mean": statistics.fmean(negative_margins),
            "negative_margin_median": statistics.median(negative_margins),
        },
        "zero_margin_policy_metrics": metrics,
        "zero_margin_policy_task_bootstrap": bootstrap,
        "sources": {
            "model": str(args.model.resolve()),
            "adapter": str(args.adapter.resolve()),
            "selector": {
                "path": str(args.selector_jsonl.resolve()),
                "sha256": _sha256(args.selector_jsonl),
            },
            "trace_sha256": _sha256(traces_path),
        },
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
