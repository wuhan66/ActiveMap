#!/usr/bin/env python3
"""Evaluate an explicit visual CALL/STOP gate followed by the semantic VLM editor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("adapter")
    parser.add_argument("rollout_jsonl", type=Path)
    parser.add_argument("gate_features", type=Path)
    parser.add_argument("gate_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--false-edit-delta", type=float, default=0.02)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_gate_probabilities(
    feature_root: Path, gate_root: Path
) -> tuple[dict[str, float], float, dict[str, Any]]:
    """Load a frozen gate and return one probability for every feature example."""

    import numpy as np
    from joblib import load

    feature_summary_path = feature_root / "summary.json"
    gate_summary_path = gate_root / "summary.json"
    feature_summary = json.loads(feature_summary_path.read_text(encoding="utf-8"))
    gate_summary = json.loads(gate_summary_path.read_text(encoding="utf-8"))
    if feature_summary.get("test_assets_read") is not False:
        raise ValueError("feature source violates frozen-test protocol")
    if gate_summary.get("test_assets_read") is not False:
        raise ValueError("gate source violates frozen-test protocol")
    expected_hash = gate_summary["sources"]["val"]["summary_sha256"]
    if _sha256(feature_summary_path) != expected_hash:
        raise ValueError("gate was not calibrated for the supplied validation features")
    features = np.load(feature_root / "features.npy").astype(np.float32)
    records = [
        json.loads(line)
        for line in (feature_root / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != len(features):
        raise ValueError("feature and metadata counts differ")
    model = load(gate_root / "gate.joblib")
    score_type = gate_summary.get("score_type", "call_probability")
    if score_type == "call_probability":
        probabilities = model.predict_proba(features)[:, 1]
    elif score_type == "predicted_utility":
        probabilities = model.predict(features)
    else:
        raise ValueError(f"unsupported gate score type: {score_type}")
    result: dict[str, float] = {}
    for row, probability in zip(records, probabilities, strict=True):
        example_id = str(row["example_id"])
        if example_id in result:
            raise ValueError(f"duplicate gate feature: {example_id}")
        result[example_id] = float(probability)
    return result, float(gate_summary["selected"]["threshold"]), gate_summary


def _metric_bundle(targets: list[str], predictions: list[str]) -> dict[str, Any]:
    from activemap.agent.vlm_evaluation import multiclass_metrics
    from activemap.models import EditOperation
    from scripts.evaluate_semantic_vlm_rollouts import _rates

    false_edit, missed_edit = _rates(targets, predictions)
    return {
        "operation_metrics": multiclass_metrics(
            targets, predictions, [operation.value for operation in EditOperation]
        ),
        "false_edit_rate": false_edit,
        "missed_edit_rate": missed_edit,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.false_edit_delta < 0.0:
        raise ValueError("false-edit-delta must be non-negative")

    import torch
    from peft import PeftModel
    from tqdm.auto import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from activemap.agent.records import AgentAction
    from activemap.agent.tool_sft import terminal_reward
    from activemap.agent.vlm_evaluation import (
        majority_operation,
        message_text,
        operation_from_action,
        realized_tool_utility,
    )
    from activemap.agent.vlm_sft import load_vlm_sft_rows
    from activemap.models import EditOperation
    from scripts.evaluate_semantic_vlm_rollouts import (
        _generate,
        _parse_action,
        _terminal_operation,
        index_pairs,
    )

    pairs = index_pairs(load_vlm_sft_rows(args.rollout_jsonl))
    if args.limit is not None:
        pairs = pairs[: args.limit]
    gate_probability, threshold, gate_summary = load_gate_probabilities(
        args.gate_features, args.gate_dir
    )
    pair_ids = {str(pre["example_id"]) for pre, _ in pairs}
    missing = pair_ids - set(gate_probability)
    if missing:
        raise ValueError(f"missing gate probabilities for {len(missing)} rollout examples")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.adapter).to(args.device).eval()

    targets: list[str] = []
    hierarchical_predictions: list[str] = []
    direct_predictions: list[str] = []
    belief_predictions: list[str] = []
    forced_predictions: list[str] = []
    hierarchical_utilities: list[float] = []
    direct_utilities: list[float] = []
    belief_utilities: list[float] = []
    forced_utilities: list[float] = []
    costs: list[float] = []
    traces = []
    target_calls = predicted_calls = true_calls = false_calls = 0
    direct_schema_valid = direct_terminal_valid = 0
    post_schema_valid = post_terminal_valid = 0

    for pre, post in tqdm(pairs, desc="Hierarchical visual agent"):
        example_id = str(pre["example_id"])
        target_action = AgentAction.model_validate_json(message_text(post["messages"][2]))
        target = operation_from_action(target_action)
        assert target is not None
        observation = json.loads(message_text(pre["messages"][1]))
        consensus = pre["consensus"]
        belief = EditOperation(consensus["baseline_operation"])
        forced = majority_operation(consensus["semantic_operations"])
        target_call = bool(pre["oracle_use_tool"])
        probability = gate_probability[example_id]
        called = probability >= threshold

        raw_direct = _generate(model, processor, pre, args)
        direct_action, direct_error = _parse_action(raw_direct)
        direct_schema_valid += int(direct_action is not None)
        direct = _terminal_operation(direct_action)
        direct_terminal_valid += int(direct is not None)
        if direct is None:
            direct = belief

        raw_post = None
        post_error = None
        post_operation = None
        if called:
            raw_post = _generate(model, processor, post, args)
            post_action, post_error = _parse_action(raw_post)
            post_schema_valid += int(post_action is not None)
            post_operation = _terminal_operation(post_action)
            post_terminal_valid += int(post_operation is not None)
            hierarchical = post_operation if post_operation is not None else belief
            cost = float(observation["semantic_tool_cost"])
        else:
            hierarchical = direct
            cost = 0.0

        belief_utility = terminal_reward(target, belief)
        direct_utility = terminal_reward(target, direct)
        forced_utility = realized_tool_utility(
            target, forced, float(observation["semantic_tool_cost"])
        )
        hierarchical_utility = terminal_reward(target, hierarchical) - cost
        targets.append(target.value)
        hierarchical_predictions.append(hierarchical.value)
        direct_predictions.append(direct.value)
        belief_predictions.append(belief.value)
        forced_predictions.append(forced.value)
        belief_utilities.append(belief_utility)
        direct_utilities.append(direct_utility)
        forced_utilities.append(forced_utility)
        hierarchical_utilities.append(hierarchical_utility)
        costs.append(cost)
        target_calls += int(target_call)
        predicted_calls += int(called)
        true_calls += int(called and target_call)
        false_calls += int(called and not target_call)
        traces.append(
            {
                "example_id": example_id,
                "task_id": pre["task_id"],
                "split": pre["split"],
                "target_operation": target.value,
                "belief_operation": belief.value,
                "direct_operation": direct.value,
                "forced_operation": forced.value,
                "hierarchical_operation": hierarchical.value,
                "policy_operation": hierarchical.value,
                "baseline_operation": direct.value,
                "target_use_tool": target_call,
                "predicted_use_tool": called,
                "call_probability": probability,
                "call_threshold": threshold,
                "direct_terminal_valid": _terminal_operation(direct_action) is not None,
                "post_terminal_valid": post_operation is not None if called else None,
                "belief_utility": belief_utility,
                "direct_utility": direct_utility,
                "forced_utility": forced_utility,
                "hierarchical_utility": hierarchical_utility,
                "policy_utility": hierarchical_utility,
                "baseline_utility": direct_utility,
                "tool_cost": cost,
                "raw_direct": raw_direct,
                "raw_post": raw_post,
                "direct_error": direct_error,
                "post_error": post_error,
            }
        )

    hierarchical_metrics = _metric_bundle(targets, hierarchical_predictions)
    direct_metrics = _metric_bundle(targets, direct_predictions)
    belief_metrics = _metric_bundle(targets, belief_predictions)
    forced_metrics = _metric_bundle(targets, forced_predictions)
    precision = true_calls / max(predicted_calls, 1)
    recall = true_calls / max(target_calls, 1)
    mean_hierarchical = sum(hierarchical_utilities) / len(pairs)
    mean_direct = sum(direct_utilities) / len(pairs)
    promotion = {
        "macro_f1_not_below_direct": hierarchical_metrics["operation_metrics"]["macro_f1"]
        >= direct_metrics["operation_metrics"]["macro_f1"],
        "utility_above_direct": mean_hierarchical > mean_direct,
        "false_edit_within_delta": hierarchical_metrics["false_edit_rate"]
        <= direct_metrics["false_edit_rate"] + args.false_edit_delta,
        "nonzero_calls": predicted_calls > 0,
        "tool_recall_at_least_0_10": recall >= 0.10,
        "call_rate_at_most_0_50": predicted_calls / len(pairs) <= 0.50,
    }
    summary = {
        "schema_version": "hierarchical-semantic-vlm-evaluation-v1",
        "model": str(Path(args.model).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "model_training_seed": args.seed,
        "sample_count": len(pairs),
        "hierarchical": hierarchical_metrics,
        "direct_vlm": direct_metrics,
        "belief_baseline": belief_metrics,
        "forced_tool": forced_metrics,
        "mean_hierarchical_utility": mean_hierarchical,
        "mean_direct_utility": mean_direct,
        "mean_belief_utility": sum(belief_utilities) / len(pairs),
        "mean_forced_utility": sum(forced_utilities) / len(pairs),
        "hierarchical_minus_direct_utility": mean_hierarchical - mean_direct,
        "hierarchical_minus_direct_macro_f1": hierarchical_metrics["operation_metrics"][
            "macro_f1"
        ]
        - direct_metrics["operation_metrics"]["macro_f1"],
        "tool_metrics": {
            "target_calls": target_calls,
            "predicted_calls": predicted_calls,
            "call_rate": predicted_calls / len(pairs),
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
            "false_call_rate": false_calls / max(len(pairs) - target_calls, 1),
            "mean_cost": sum(costs) / len(pairs),
        },
        "execution_validity": {
            "direct_schema_valid_rate": direct_schema_valid / len(pairs),
            "direct_terminal_valid_rate": direct_terminal_valid / len(pairs),
            "post_schema_valid_rate": post_schema_valid / max(predicted_calls, 1),
            "post_terminal_valid_rate": post_terminal_valid / max(predicted_calls, 1),
        },
        "gate": {
            "threshold": threshold,
            "selection_protocol": gate_summary["selection_protocol"],
            "feature_protocol": gate_summary["feature_protocol"],
            "summary_sha256": _sha256(args.gate_dir / "summary.json"),
        },
        "promotion_gate": {**promotion, "passed": all(promotion.values())},
        "utility_protocol": "realized-terminal-reward-minus-executed-tool-cost",
        "fallback_protocol": "invalid-direct-or-post-preserves-belief-baseline",
        "max_tool_calls_per_episode": 1,
        "test_assets_read": False,
        "limit": args.limit,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
