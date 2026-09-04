#!/usr/bin/env python3
"""Make the final RL-vs-SFT decision from three-seed executable writebacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_METRICS = (
    "episode_utility_v2_safety_auc",
    "raster_iou_gain_auc",
    "false_edit_auc",
    "vector_delta_topology_valid_auc",
)


def assess(
    aggregate: dict[str, Any],
    *,
    min_safety_utility_ci_low: float = 0.0,
    min_raster_gain_ci_low: float = 0.0,
    max_false_edit_ci_high: float = 0.0,
    min_topology_observed: float = 0.0,
) -> dict[str, Any]:
    if aggregate.get("schema_version") != "agent-writeback-seed-matched-aggregate-v1":
        raise ValueError("unexpected writeback aggregate schema")
    if (
        aggregate.get("split") != "val"
        or aggregate.get("test_assets_read") is not False
        or aggregate.get("seed_matched_references") is not True
        or int(aggregate.get("seed_count", 0)) < 3
    ):
        raise ValueError("final writeback assessment requires three-seed matched val data")
    intervals = aggregate.get("candidate_minus_seed_matched_sft", {})
    missing = [name for name in REQUIRED_METRICS if name not in intervals]
    if missing:
        raise ValueError(f"writeback aggregate lacks required metrics: {missing}")

    checks = {
        "safety_utility_ci_positive": (
            float(intervals["episode_utility_v2_safety_auc"]["ci95_low"])
            > min_safety_utility_ci_low
        ),
        "raster_gain_ci_nonnegative": (
            float(intervals["raster_iou_gain_auc"]["ci95_low"])
            >= min_raster_gain_ci_low
        ),
        "false_edit_ci_noninferior": (
            float(intervals["false_edit_auc"]["ci95_high"])
            <= max_false_edit_ci_high
        ),
        "topology_observed_nonnegative": (
            float(
                intervals["vector_delta_topology_valid_auc"]["observed_delta"]
            )
            >= min_topology_observed
        ),
    }
    promote = all(checks.values())
    return {
        "schema_version": "promoted-rl-writeback-final-assessment-v1",
        "decision": "promote_rl" if promote else "retain_sft",
        "checks": checks,
        "thresholds": {
            "min_safety_utility_ci_low_exclusive": min_safety_utility_ci_low,
            "min_raster_gain_ci_low_inclusive": min_raster_gain_ci_low,
            "max_false_edit_ci_high_inclusive": max_false_edit_ci_high,
            "min_topology_observed_inclusive": min_topology_observed,
        },
        "evidence": {
            name: intervals[name] for name in REQUIRED_METRICS
        },
        "seed_count": int(aggregate["seed_count"]),
        "split": "val",
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-safety-utility-ci-low", type=float, default=0.0)
    parser.add_argument("--min-raster-gain-ci-low", type=float, default=0.0)
    parser.add_argument("--max-false-edit-ci-high", type=float, default=0.0)
    parser.add_argument("--min-topology-observed", type=float, default=0.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    result = assess(
        aggregate,
        min_safety_utility_ci_low=args.min_safety_utility_ci_low,
        min_raster_gain_ci_low=args.min_raster_gain_ci_low,
        max_false_edit_ci_high=args.max_false_edit_ci_high,
        min_topology_observed=args.min_topology_observed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
