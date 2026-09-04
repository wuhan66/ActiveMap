#!/usr/bin/env python3
"""Aggregate paired reliability-gate evidence across independent seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _named_path(value: str) -> tuple[int, Path]:
    seed, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("expected SEED=PATH")
    return int(seed), Path(path)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(
    component: dict[str, Any], closed_loop: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    if component.get("schema_version") != "tool-belief-reliability-gate-ablation-v1":
        raise ValueError("unexpected component ablation schema")
    component_seeds = [int(seed) for seed in component.get("paired_seeds", [])]
    if len(component_seeds) < 3 or set(component_seeds) != set(closed_loop):
        raise ValueError("component and closed-loop comparisons require the same 3+ seeds")

    rows = []
    for seed in sorted(closed_loop):
        report = closed_loop[seed]
        if report.get("schema_version") != "active-catalog-closed-loop-reliability-gate-ablation-v1":
            raise ValueError(f"unexpected closed-loop schema for seed {seed}")
        intervals = report["paired_aoi_bootstrap"]["intervals"]
        checks = report["promotion_checks"]
        rows.append(
            {
                "seed": seed,
                "quality_cost_utility_delta": float(
                    intervals["mean_quality_cost_utility"]["observed_delta"]
                ),
                "terminal_accuracy_delta": float(
                    intervals["terminal_accuracy"]["observed_delta"]
                ),
                "false_edit_rate_delta": float(
                    intervals["false_edit_rate"]["observed_delta"]
                ),
                "utility_ci95_low": float(
                    intervals["mean_quality_cost_utility"]["ci95_low"]
                ),
                "tool_calls_nonzero": bool(checks["tool_calls_nonzero"]),
                "accuracy_noninferior": bool(checks["accuracy_noninferior"]),
                "false_edit_noninferior": bool(checks["false_edit_noninferior"]),
                "passed": bool(checks["passed"]),
            }
        )

    mean = lambda key: statistics.fmean(float(row[key]) for row in rows)
    majority = len(rows) // 2 + 1
    checks = {
        "component_promoted": bool(component.get("promotion_passed")),
        "nonzero_calls_all_seeds": all(row["tool_calls_nonzero"] for row in rows),
        "mean_utility_improved": mean("quality_cost_utility_delta") > 0.0,
        "utility_ci_positive_majority": sum(
            row["utility_ci95_low"] > 0.0 for row in rows
        )
        >= majority,
        "accuracy_noninferior_all_seeds": all(
            row["accuracy_noninferior"] for row in rows
        ),
        "false_edit_noninferior_all_seeds": all(
            row["false_edit_noninferior"] for row in rows
        ),
        "per_seed_promotion_majority": sum(row["passed"] for row in rows) >= majority,
    }
    checks["passed"] = all(checks.values())
    return {
        "schema_version": "active-catalog-reliability-gate-three-seed-promotion-v1",
        "paired_seeds": sorted(closed_loop),
        "controlled_difference": "tool_belief_reliability_gate",
        "aggregate_observed_deltas": {
            "quality_cost_utility": mean("quality_cost_utility_delta"),
            "terminal_accuracy": mean("terminal_accuracy_delta"),
            "false_edit_rate": mean("false_edit_rate_delta"),
        },
        "checks": checks,
        "promotion_passed": checks["passed"],
        "per_seed": rows,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--component", type=Path, required=True)
    parser.add_argument("--closed-loop", action="append", type=_named_path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    paths = dict(args.closed_loop)
    if len(paths) != len(args.closed_loop):
        raise ValueError("duplicate closed-loop seed")
    report = aggregate(_read(args.component), {seed: _read(path) for seed, path in paths.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
