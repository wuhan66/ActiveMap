#!/usr/bin/env python3
"""Aggregate fixed-seed hierarchical visual-agent validation results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("aggregate values must be finite and nonempty")
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate(rows: list[dict[str, Any]], expected_seeds: list[int]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one hierarchical result is required")
    by_seed = {int(row["model_training_seed"]): row for row in rows}
    if len(by_seed) != len(rows):
        raise ValueError("duplicate model training seed")
    if sorted(by_seed) != sorted(expected_seeds):
        raise ValueError("results do not match expected seeds")
    ordered = [by_seed[seed] for seed in expected_seeds]
    if any(row.get("test_assets_read") is not False for row in ordered):
        raise ValueError("one or more results violate the frozen-test protocol")

    def values(path: tuple[str, ...]) -> list[float]:
        result = []
        for row in ordered:
            value: Any = row
            for key in path:
                value = value[key]
            result.append(float(value))
        return result

    metrics = {
        "hierarchical_macro_f1": _mean_std(
            values(("hierarchical", "operation_metrics", "macro_f1"))
        ),
        "direct_macro_f1": _mean_std(
            values(("direct_vlm", "operation_metrics", "macro_f1"))
        ),
        "macro_f1_delta": _mean_std(values(("hierarchical_minus_direct_macro_f1",))),
        "hierarchical_utility": _mean_std(values(("mean_hierarchical_utility",))),
        "direct_utility": _mean_std(values(("mean_direct_utility",))),
        "utility_delta": _mean_std(values(("hierarchical_minus_direct_utility",))),
        "hierarchical_false_edit": _mean_std(
            values(("hierarchical", "false_edit_rate"))
        ),
        "direct_false_edit": _mean_std(values(("direct_vlm", "false_edit_rate"))),
        "false_edit_delta": _mean_std(
            [
                float(row["hierarchical"]["false_edit_rate"])
                - float(row["direct_vlm"]["false_edit_rate"])
                for row in ordered
            ]
        ),
        "hierarchical_missed_edit": _mean_std(
            values(("hierarchical", "missed_edit_rate"))
        ),
        "direct_missed_edit": _mean_std(values(("direct_vlm", "missed_edit_rate"))),
        "tool_call_rate": _mean_std(values(("tool_metrics", "call_rate"))),
        "tool_precision": _mean_std(values(("tool_metrics", "precision"))),
        "tool_recall": _mean_std(values(("tool_metrics", "recall"))),
    }
    passed = [bool(row["promotion_gate"]["passed"]) for row in ordered]
    return {
        "schema_version": "hierarchical-semantic-vlm-three-seed-aggregate-v1",
        "expected_seeds": expected_seeds,
        "result_count": len(ordered),
        "per_seed": [
            {
                "seed": seed,
                "promotion_passed": bool(row["promotion_gate"]["passed"]),
                "macro_f1_delta": float(row["hierarchical_minus_direct_macro_f1"]),
                "utility_delta": float(row["hierarchical_minus_direct_utility"]),
                "false_edit_delta": float(row["hierarchical"]["false_edit_rate"])
                - float(row["direct_vlm"]["false_edit_rate"]),
                "call_rate": float(row["tool_metrics"]["call_rate"]),
                "precision": float(row["tool_metrics"]["precision"]),
                "recall": float(row["tool_metrics"]["recall"]),
            }
            for seed, row in zip(expected_seeds, ordered, strict=True)
        ],
        "metrics": metrics,
        "all_seed_gates_passed": all(passed),
        "paper_claim_ready": False,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--expected-seeds", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.result]
    expected = [int(value) for value in args.expected_seeds.split(",")]
    payload = aggregate(rows, expected)
    payload["sources"] = [str(path.resolve()) for path in args.result]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
