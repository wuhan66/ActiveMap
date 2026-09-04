#!/usr/bin/env python3
"""Assess whether balanced tool supervision fixes single-seed reachability."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


CANDIDATE = "qwen3_4b_sft_tool_to_belief"


def _mean_metrics(summary: dict[str, Any], method: str) -> dict[str, float]:
    rows = [row for row in summary["results"] if row["method"] == method]
    if not rows:
        raise ValueError(f"rollout summary lacks method={method}")
    keys = (
        "mean_quality_cost_utility",
        "false_edit_rate",
        "mean_tool_calls",
        "mean_tool_belief_l1_delta",
    )
    return {key: statistics.fmean(float(row[key]) for row in rows) for key in keys}


def assess(
    static: dict[str, Any], rollout: dict[str, Any], reachability: dict[str, Any]
) -> dict[str, Any]:
    if static.get("protocol", {}).get("test_assets_read") is not False:
        raise ValueError("static decision does not prove test isolation")
    if rollout.get("protocol", {}).get("test_assets_read") is not False:
        raise ValueError("rollout does not prove test isolation")
    if reachability.get("test_assets_read") is not False:
        raise ValueError("reachability audit does not prove test isolation")
    if reachability.get("summary", {}).get("single_seed_diagnostic") is not True:
        raise ValueError("v10 diagnostic requires a single-seed reachability audit")

    selected = static.get("selected_checkpoint")
    selected_row = next(
        (row for row in static.get("checkpoints", []) if row.get("label") == selected),
        None,
    )
    if selected_row is None:
        raise ValueError("static decision lacks its selected checkpoint")

    candidate = _mean_metrics(rollout, CANDIDATE)
    edit = _mean_metrics(rollout, "edit_conditioned_selector")
    no_belief = _mean_metrics(rollout, "qwen3_4b_sft_tools_no_belief")
    forced = _mean_metrics(rollout, "forced_tools")
    recurrent_calls = int(
        rollout.get("action_counts", {}).get(CANDIDATE, {}).get("USE_TOOL", 0)
    )
    checks = {
        "static_selection_passed": bool(static.get("selection_passed")),
        "static_tool_calls_nonzero": int(selected_row.get("predicted_tool_calls", 0)) > 0,
        "recurrent_tool_calls_nonzero": recurrent_calls > 0,
        "recurrent_belief_active": candidate["mean_tool_belief_l1_delta"] > 0.0,
        "utility_above_edit_selector": candidate["mean_quality_cost_utility"]
        > edit["mean_quality_cost_utility"],
        "utility_above_no_belief": candidate["mean_quality_cost_utility"]
        > no_belief["mean_quality_cost_utility"],
        "utility_above_forced": candidate["mean_quality_cost_utility"]
        > forced["mean_quality_cost_utility"],
        "false_edit_absolute_safe": candidate["false_edit_rate"] <= 0.05,
        "false_edit_noninferior_to_edit": candidate["false_edit_rate"]
        <= edit["false_edit_rate"] + 0.01,
    }
    checks["passed"] = all(checks.values())
    return {
        "schema_version": "muno21-balanced-tool-single-seed-diagnostic-v1",
        "selected_checkpoint": selected,
        "recurrent_tool_calls": recurrent_calls,
        "metrics": {
            "candidate": candidate,
            "edit_conditioned_selector": edit,
            "tools_no_belief": no_belief,
            "forced_tools": forced,
        },
        "checks": checks,
        "diagnostic_passed": checks["passed"],
        "next_stage": "replicate_balanced_tool_sft" if checks["passed"] else None,
        "claim_boundary": "Single-seed validation diagnostic; not paper claim evidence.",
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("static_decision", type=Path)
    parser.add_argument("rollout_summary", type=Path)
    parser.add_argument("reachability_audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = assess(
        json.loads(args.static_decision.read_text(encoding="utf-8")),
        json.loads(args.rollout_summary.read_text(encoding="utf-8")),
        json.loads(args.reachability_audit.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
