#!/usr/bin/env python3
"""Evaluate a frozen policy-relative gate from cached direct/post VLM branches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_cached_branches(
    rows: list[dict[str, Any]], probabilities: dict[str, float], threshold: float
) -> list[dict[str, Any]]:
    ids = {str(row["example_id"]) for row in rows}
    if len(ids) != len(rows):
        raise ValueError("cached branches contain duplicate example IDs")
    if ids != set(probabilities):
        raise ValueError("gate probabilities and cached branch IDs differ")
    selected = []
    for row in rows:
        example_id = str(row["example_id"])
        probability = float(probabilities[example_id])
        called = probability >= threshold
        operation = row["post_tool_operation"] if called else row["direct_operation"]
        utility = row["post_tool_utility"] if called else row["direct_utility"]
        selected.append(
            {
                **row,
                "call_probability": probability,
                "call_threshold": threshold,
                "predicted_use_tool": called,
                "policy_operation": operation,
                "policy_utility": float(utility),
            }
        )
    return selected


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate_features", type=Path)
    parser.add_argument("gate_dir", type=Path)
    parser.add_argument("branch_cache", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--false-edit-delta", type=float, default=0.02)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.false_edit_delta < 0.0:
        raise ValueError("false-edit-delta must be non-negative")

    from scripts.evaluate_hierarchical_semantic_vlm import (
        _metric_bundle,
        load_gate_probabilities,
    )

    branch_summary = json.loads(
        (args.branch_cache / "summary.json").read_text(encoding="utf-8")
    )
    if branch_summary.get("test_assets_read") is not False:
        raise ValueError("branch cache violates the frozen-test protocol")
    branch_rows = [
        json.loads(line)
        for line in (args.branch_cache / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    probabilities, threshold, gate_summary = load_gate_probabilities(
        args.gate_features, args.gate_dir
    )
    rows = select_cached_branches(branch_rows, probabilities, threshold)
    targets = [str(row["target_operation"]) for row in rows]
    policy_operations = [str(row["policy_operation"]) for row in rows]
    direct_operations = [str(row["direct_operation"]) for row in rows]
    oracle_operations = [
        str(row["post_tool_operation"])
        if row["policy_relative_use_tool"]
        else str(row["direct_operation"])
        for row in rows
    ]
    static_operations = [
        str(row["post_tool_operation"])
        if row["static_use_tool"]
        else str(row["direct_operation"])
        for row in rows
    ]
    policy_utilities = [float(row["policy_utility"]) for row in rows]
    direct_utilities = [float(row["direct_utility"]) for row in rows]
    oracle_utilities = [
        max(float(row["direct_utility"]), float(row["post_tool_utility"])) for row in rows
    ]
    static_utilities = [
        float(row["post_tool_utility"])
        if row["static_use_tool"]
        else float(row["direct_utility"])
        for row in rows
    ]
    policy_metrics = _metric_bundle(targets, policy_operations)
    direct_metrics = _metric_bundle(targets, direct_operations)
    oracle_metrics = _metric_bundle(targets, oracle_operations)
    static_metrics = _metric_bundle(targets, static_operations)
    predicted_calls = sum(bool(row["predicted_use_tool"]) for row in rows)
    target_calls = sum(bool(row["policy_relative_use_tool"]) for row in rows)
    true_calls = sum(
        bool(row["predicted_use_tool"] and row["policy_relative_use_tool"]) for row in rows
    )
    false_calls = predicted_calls - true_calls
    precision = true_calls / max(predicted_calls, 1)
    recall = true_calls / max(target_calls, 1)
    mean_policy = _mean(policy_utilities)
    mean_direct = _mean(direct_utilities)
    mean_oracle = _mean(oracle_utilities)
    promotion = {
        "macro_f1_not_below_direct": policy_metrics["operation_metrics"]["macro_f1"]
        >= direct_metrics["operation_metrics"]["macro_f1"],
        "utility_above_direct": mean_policy > mean_direct,
        "false_edit_within_delta": policy_metrics["false_edit_rate"]
        <= direct_metrics["false_edit_rate"] + args.false_edit_delta,
        "nonzero_calls": predicted_calls > 0,
        "tool_recall_at_least_0_10": recall >= 0.10,
        "call_rate_at_most_0_50": predicted_calls / len(rows) <= 0.50,
    }
    summary = {
        "schema_version": "cached-policy-relative-gate-evaluation-v1",
        "sample_count": len(rows),
        "policy": policy_metrics,
        "direct_vlm": direct_metrics,
        "policy_relative_oracle": oracle_metrics,
        "static_label_oracle": static_metrics,
        "mean_policy_utility": mean_policy,
        "mean_direct_utility": mean_direct,
        "mean_policy_relative_oracle_utility": mean_oracle,
        "mean_static_label_utility": _mean(static_utilities),
        "policy_minus_direct_utility": mean_policy - mean_direct,
        "policy_minus_direct_macro_f1": policy_metrics["operation_metrics"]["macro_f1"]
        - direct_metrics["operation_metrics"]["macro_f1"],
        "acquisition_regret_to_policy_relative_oracle": mean_oracle - mean_policy,
        "tool_metrics": {
            "target_calls": target_calls,
            "predicted_calls": predicted_calls,
            "true_calls": true_calls,
            "false_calls": false_calls,
            "call_rate": predicted_calls / len(rows),
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        },
        "gate": {
            "threshold": threshold,
            "selection_protocol": gate_summary["selection_protocol"],
            "selection_objective": gate_summary["selection_objective"],
            "fit_weighting": gate_summary.get("fit_weighting"),
            "summary_sha256": _sha256(args.gate_dir / "summary.json"),
        },
        "branch_cache_summary_sha256": _sha256(args.branch_cache / "summary.json"),
        "promotion_gate": {**promotion, "passed": all(promotion.values())},
        "utility_protocol": "cached-realized-post-tool-minus-direct-with-executed-cost",
        "validation_role": "frozen-evaluation-only",
        "test_assets_read": False,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
