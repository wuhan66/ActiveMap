#!/usr/bin/env python3
"""Promote the frozen SN7 Step-0 benefit gate from its three-seed frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def assess(
    payload: dict[str, Any],
    *,
    min_seeds: int = 3,
    expected_split: str = "val",
) -> dict[str, Any]:
    if expected_split not in {"val", "test"}:
        raise ValueError("expected_split must be val or test")
    if payload.get("schema_version") != "sn7-selective-tool-frontier-three-seed-v1":
        raise ValueError("unexpected Step-0 frontier schema")
    seeds = list(payload.get("seeds", []))
    rows = payload.get("per_seed", [])
    by_seed = {
        (int(row["seed"]), str(row["variant"])): row
        for row in rows
    }
    benefit_no_tool = payload["comparisons"]["benefit_vs_notool"][
        "hierarchical_seed_aoi_bootstrap"
    ]
    benefit_forced = payload["comparisons"]["benefit_vs_forced"][
        "hierarchical_seed_aoi_bootstrap"
    ]
    protocol = payload.get("protocol", {})

    checks = {
        "required_seed_count": len(seeds) >= min_seeds,
        "all_seed_rows_present": all(
            (int(seed), variant) in by_seed
            for seed in seeds
            for variant in ("notool", "benefit", "forced")
        ),
        "every_seed_has_tool_calls": all(
            by_seed[(int(seed), "benefit")]["mean_tool_calls"] > 0.0
            for seed in seeds
        ),
        "every_seed_changes_belief": all(
            by_seed[(int(seed), "benefit")]["mean_tool_belief_l1_delta"] > 0.0
            for seed in seeds
        ),
        "every_seed_is_sparser_than_forced": all(
            by_seed[(int(seed), "benefit")]["mean_tool_calls"]
            < by_seed[(int(seed), "forced")]["mean_tool_calls"]
            for seed in seeds
        ),
        "accuracy_gain_vs_no_tool": (
            benefit_no_tool["terminal_correct"]["ci95_low"] > 0.0
        ),
        "false_edit_reduction_vs_no_tool": (
            benefit_no_tool["false_edit"]["ci95_high"] < 0.0
        ),
        "terminal_parity_vs_forced": (
            benefit_forced["terminal_correct"]["ci95_low"] >= 0.0
            and benefit_forced["terminal_correct"]["ci95_high"] <= 0.0
        ),
        "utility_gain_vs_forced": (
            benefit_forced["quality_cost_utility"]["ci95_low"] > 0.0
        ),
        "cost_reduction_vs_forced": benefit_forced["tool_calls"]["ci95_high"] < 0.0,
        "train_only_gate_calibration": (
            protocol.get("gate_calibration_split") == "train"
            and protocol.get("validation_used_for_gate_calibration") is False
        ),
        "matched_samples_and_budgets": (
            protocol.get("matched_samples_and_budgets") is True
        ),
        "split_provenance_matches": (
            protocol.get("split") == expected_split
            and bool(protocol.get("test_assets_read")) == (expected_split == "test")
        ),
    }
    return {
        "schema_version": "active-catalog-tool-branch-promotion-v1",
        "promote": all(checks.values()),
        "checks": checks,
        "minimum_seed_count": min_seeds,
        "seeds": seeds,
        "source_schema": payload["schema_version"],
        "split": expected_split,
        "test_assets_read": bool(protocol.get("test_assets_read")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frontier", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--expected-split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = assess(
        json.loads(args.frontier.read_text(encoding="utf-8")),
        min_seeds=args.min_seeds,
        expected_split=args.expected_split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
