#!/usr/bin/env python3
"""Predeclared promotion gate for selective recurrent tool use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def assess(payload: dict[str, Any], *, false_margin: float = 0.02,
           accuracy_margin: float = 0.01, min_seeds: int = 1) -> dict[str, Any]:
    if payload.get("schema_version") != "active-catalog-closed-loop-paired-comparison-v1":
        raise ValueError("unexpected comparison schema")
    orientation = payload.get("comparison_orientation", {})
    references = set(orientation.get("references", []))
    if orientation.get("candidate") != "selective" or not {
        "forced",
        "no_tool",
    }.issubset(references):
        raise ValueError("promotion requires selective versus forced and no_tool")
    comparisons = payload["paired_aoi_comparisons"]
    selective = payload["metrics"]["selective"]
    forced = payload["metrics"]["forced"]
    seed_count = int(payload.get("seed_count", 1))
    per_seed = payload.get("per_seed_metrics", {})

    def interval(reference: str, metric: str) -> dict[str, float]:
        return comparisons[f"selective_minus_{reference}"]["intervals"][metric]

    checks = {
        "required_seed_count": seed_count >= min_seeds,
        "nonzero_tool_calls": selective["mean_tool_calls"] > 0.0,
        "nonzero_tool_belief_change": selective["mean_tool_belief_l1_delta"] > 0.0,
        "sparser_than_forced": selective["mean_tool_calls"] < forced["mean_tool_calls"],
        "utility_gain_vs_no_tool": interval("no_tool", "mean_quality_cost_utility")["ci95_low"] > 0.0,
        "utility_gain_vs_forced": interval("forced", "mean_quality_cost_utility")["ci95_low"] > 0.0,
        "cost_reduction_vs_forced": interval("forced", "mean_cost")["ci95_high"] < 0.0,
        "false_edit_noninferior_vs_no_tool": interval("no_tool", "false_edit_rate")["ci95_high"] <= false_margin,
        "false_edit_noninferior_vs_forced": interval("forced", "false_edit_rate")["ci95_high"] <= false_margin,
        "accuracy_noninferior_vs_no_tool": interval("no_tool", "terminal_accuracy")["ci95_low"] >= -accuracy_margin,
        "accuracy_noninferior_vs_forced": interval("forced", "terminal_accuracy")["ci95_low"] >= -accuracy_margin,
    }
    if min_seeds > 1:
        checks["every_seed_has_tool_calls"] = len(per_seed) == seed_count and all(
            values["selective"]["mean_tool_calls"] > 0.0 for values in per_seed.values()
        )
        checks["every_seed_changes_belief"] = len(per_seed) == seed_count and all(
            values["selective"]["mean_tool_belief_l1_delta"] > 0.0
            for values in per_seed.values()
        )
        checks["every_seed_is_sparser_than_forced"] = len(per_seed) == seed_count and all(
            values["selective"]["mean_tool_calls"] < values["forced"]["mean_tool_calls"]
            for values in per_seed.values()
        )
    causal_diagnostics = {}
    if "selective_frozen_prior" in references:
        recurrent_interval = interval(
            "selective_frozen_prior", "mean_quality_cost_utility"
        )
        causal_diagnostics = {
            "recurrent_belief_observed_utility_gain": recurrent_interval[
                "observed_delta"
            ],
            "recurrent_belief_utility_ci95_low": recurrent_interval["ci95_low"],
            "recurrent_belief_utility_ci95_high": recurrent_interval["ci95_high"],
            "recurrent_belief_strict_gain_supported": (
                recurrent_interval["ci95_low"] > 0.0
            ),
        }
    return {
        "schema_version": "active-catalog-tool-branch-promotion-v1",
        "promote": all(checks.values()),
        "checks": checks,
        "margins": {"false_edit": false_margin, "terminal_accuracy": accuracy_margin},
        "minimum_seed_count": min_seeds,
        "causal_diagnostics": causal_diagnostics,
        "test_assets_read": payload.get("test_assets_read") is True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--false-edit-margin", type=float, default=0.02)
    parser.add_argument("--accuracy-margin", type=float, default=0.01)
    parser.add_argument("--min-seeds", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = assess(
        json.loads(args.comparison.read_text(encoding="utf-8")),
        false_margin=args.false_edit_margin,
        accuracy_margin=args.accuracy_margin,
        min_seeds=args.min_seeds,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
