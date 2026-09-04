#!/usr/bin/env python3
"""Aggregate frozen visual-policy validation results across model seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _metric(summary: dict[str, Any], dotted: str) -> float:
    value: Any = summary
    for key in dotted.split("."):
        value = value[key]
    return float(value)


METRICS = [
    "operation_metrics.macro_f1",
    "baseline_operation_metrics.macro_f1",
    "forced_operation_metrics.macro_f1",
    "false_edit_rate",
    "baseline_false_edit_rate",
    "forced_false_edit_rate",
    "missed_edit_rate",
    "baseline_missed_edit_rate",
    "forced_missed_edit_rate",
    "mean_policy_utility",
    "mean_baseline_utility",
    "mean_forced_utility",
    "mean_policy_gain",
    "tool_metrics.call_rate",
    "tool_metrics.precision",
    "tool_metrics.recall",
    "tool_metrics.f1",
    "tool_metrics.false_call_rate",
    "tool_metrics.mean_cost",
]


def assess_seed(summary: dict[str, Any]) -> dict[str, Any]:
    gates = {
        "test_isolation": summary.get("test_assets_read") is False,
        "schema_valid": float(summary["schema_valid_rate"]) >= 0.99,
        "executable_valid": float(summary["executable_valid_rate"]) >= 0.99,
        "operation_quality": _metric(summary, "operation_metrics.macro_f1")
        >= _metric(summary, "baseline_operation_metrics.macro_f1"),
        "cost_adjusted_utility": float(summary["mean_policy_utility"])
        > float(summary["mean_baseline_utility"]),
        "false_edit_safety": float(summary["false_edit_rate"])
        <= float(summary["baseline_false_edit_rate"]) + 0.02,
        "sparse_nonzero_calls": 0.0 < _metric(summary, "tool_metrics.call_rate") <= 0.5,
        "tool_opportunity_recall": _metric(summary, "tool_metrics.recall") >= 0.10,
    }
    return {
        "model_training_seed": int(summary["model_training_seed"]),
        "passed": all(gates.values()),
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "metrics": {name: _metric(summary, name) for name in METRICS},
        "source": summary.get("adapter"),
    }


def aggregate(
    summaries: list[dict[str, Any]], expected_seeds: list[int]
) -> dict[str, Any]:
    by_seed = {int(summary["model_training_seed"]): summary for summary in summaries}
    if sorted(by_seed) != sorted(expected_seeds) or len(by_seed) != len(summaries):
        raise ValueError("evaluation results do not match the expected unique seeds")
    sample_counts = {int(summary["sample_count"]) for summary in summaries}
    validation_paths = {str(summary["validation_jsonl"]) for summary in summaries}
    if len(sample_counts) != 1 or len(validation_paths) != 1:
        raise ValueError("seed evaluations do not share one frozen validation protocol")
    assessed = [assess_seed(by_seed[seed]) for seed in expected_seeds]
    metrics = {}
    for name in METRICS:
        values = [_metric(by_seed[seed], name) for seed in expected_seeds]
        metrics[name] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    passed = all(row["passed"] for row in assessed)
    return {
        "schema_version": "semantic-vlm-action-seed-aggregate-v1",
        "model_training_seeds": expected_seeds,
        "seed_count": len(expected_seeds),
        "sample_count_per_seed": sample_counts.pop(),
        "validation_jsonl": validation_paths.pop(),
        "per_seed": assessed,
        "aggregate_metrics": metrics,
        "all_static_gates_passed": passed,
        "closed_loop_evaluation_ready": passed,
        "paper_claim_ready": False,
        "test_assets_read": False,
        "protocol": {
            "seed_selection": "none-all-fixed-seeds-required",
            "max_false_edit_delta": 0.02,
            "max_tool_call_rate": 0.5,
            "min_tool_recall": 0.10,
            "utility_includes_tool_cost": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--expected-seeds", required=True)
    args = parser.parse_args()
    expected = [int(value) for value in args.expected_seeds.split(",")]
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.result]
    result = aggregate(summaries, expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
