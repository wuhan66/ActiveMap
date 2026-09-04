#!/usr/bin/env python3
"""Aggregate seed-matched GRPO-vs-SFT paired bootstrap reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("protocol", {}).get("test_assets_read") is not False:
        raise ValueError(f"test isolation is not proven: {path}")
    if report.get("protocol", {}).get("paired") is not True:
        raise ValueError(f"report is not paired: {path}")
    return report


def aggregate(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != 3:
        raise ValueError("exactly three paired reports are required")
    reports = [_load(path) for path in paths]
    seeds = [
        int(report["protocol"].get("model_seed", report["protocol"]["seed"]))
        for report in reports
    ]
    if len(set(seeds)) != 3:
        raise ValueError("paired reports must have three unique seeds")
    reference = reports[0]
    for report in reports[1:]:
        if report["budgets"] != reference["budgets"]:
            raise ValueError("budget coverage differs across seeds")
        if report["task_count"] != reference["task_count"]:
            raise ValueError("task count differs across seeds")
        if report["sample_count"] != reference["sample_count"]:
            raise ValueError("sample count differs across seeds")

    metric_names = sorted(reports[0]["paired_delta"])
    seed_metrics: dict[str, list[dict[str, float]]] = {
        metric: [report["paired_delta"][metric] for report in reports]
        for metric in metric_names
    }
    aggregated: dict[str, dict[str, Any]] = {}
    for metric, values in seed_metrics.items():
        deltas = [float(value["delta"]) for value in values]
        seed_mean = mean(deltas)
        seed_std = stdev(deltas) if len(deltas) > 1 else 0.0
        half_width = 1.96 * seed_std / math.sqrt(len(deltas))
        aggregated[metric] = {
            "seed_deltas": deltas,
            "seed_mean_delta": seed_mean,
            "seed_std": seed_std,
            "seed_mean_ci95_low": seed_mean - half_width,
            "seed_mean_ci95_high": seed_mean + half_width,
            "bootstrap_ci95_low_range": min(float(value["ci95_low"]) for value in values),
            "bootstrap_ci95_high_range": max(float(value["ci95_high"]) for value in values),
            "positive_seed_count": sum(delta > 0.0 for delta in deltas),
            "nonnegative_seed_count": sum(delta >= 0.0 for delta in deltas),
        }

    return {
        "schema_version": "activemap-reachable-tool-grpo-three-seed-aggregate-v1",
        "claim_boundary": "seed-matched validation only; no test access and no promotion",
        "protocol": {
            "paired": True,
            "split": "val",
            "model_seeds": seeds,
            "task_count": reference["task_count"],
            "sample_count": reference["sample_count"],
            "budgets": reference["budgets"],
            "budget_coverage": reference["protocol"]["budget_coverage"],
            "bootstrap_per_seed": reference["protocol"]["bootstrap"],
            "bootstrap_seeds": [
                int(report["protocol"].get("bootstrap_seed", report["protocol"]["seed"]))
                for report in reports
            ],
            "test_assets_read": False,
        },
        "per_seed_reports": [str(path) for path in paths],
        "candidate_minus_seed_matched_sft": aggregated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs=3, type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = aggregate(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
