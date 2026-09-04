#!/usr/bin/env python3
"""Assess the single-seed closed-loop value of calibrated tool admission."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


CANDIDATE = "qwen3_4b_sft_calibrated_tool_to_belief"


def _mean(summary: dict[str, Any], method: str) -> dict[str, float]:
    rows = [row for row in summary["results"] if row["method"] == method]
    if not rows:
        raise ValueError(f"missing rollout method={method}")
    return {
        key: statistics.fmean(float(row[key]) for row in rows)
        for key in (
            "mean_quality_cost_utility",
            "false_edit_rate",
            "mean_tool_calls",
            "mean_tool_belief_l1_delta",
        )
    }


def assess(summary: dict[str, Any]) -> dict[str, Any]:
    protocol = summary.get("protocol", {})
    if protocol.get("test_assets_read") is not False:
        raise ValueError("v11 diagnostic must use validation only")
    gate = protocol.get("tool_need_gate") or {}
    candidate = _mean(summary, CANDIDATE)
    uncalibrated = _mean(summary, "qwen3_4b_sft_tool_to_belief")
    edit = _mean(summary, "edit_conditioned_selector")
    no_belief = _mean(summary, "qwen3_4b_sft_tools_no_belief")
    forced = _mean(summary, "forced_tools")
    calls = int(summary.get("action_counts", {}).get(CANDIDATE, {}).get("USE_TOOL", 0))
    checks = {
        "gate_proposals_nonzero": int(gate.get("proposals", 0)) > 0,
        "gate_admits_nonzero": int(gate.get("admitted", 0)) > 0,
        "recurrent_calls_nonzero": calls > 0,
        "belief_update_active": candidate["mean_tool_belief_l1_delta"] > 0.0,
        "fewer_calls_than_uncalibrated": candidate["mean_tool_calls"]
        < uncalibrated["mean_tool_calls"],
        "utility_above_uncalibrated": candidate["mean_quality_cost_utility"]
        > uncalibrated["mean_quality_cost_utility"],
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
        "schema_version": "muno21-calibrated-tool-need-single-seed-diagnostic-v1",
        "metrics": {
            "candidate": candidate,
            "uncalibrated": uncalibrated,
            "edit_conditioned_selector": edit,
            "tools_no_belief": no_belief,
            "forced_tools": forced,
        },
        "gate": gate,
        "recurrent_tool_calls": calls,
        "checks": checks,
        "diagnostic_passed": checks["passed"],
        "claim_boundary": "Single-seed validation diagnostic; not paper evidence.",
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = assess(json.loads(args.rollout_summary.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
