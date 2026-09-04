#!/usr/bin/env python3
"""Assess proactive paired tools against the frozen v11 controls."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


QWEN = "qwen3_4b_sft_calibrated_tool_to_belief"
HYBRID = "edit_conditioned_proactive_tools"


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


def assess(v11: dict[str, Any], v12: dict[str, Any]) -> dict[str, Any]:
    for summary in (v11, v12):
        if summary.get("protocol", {}).get("test_assets_read") is not False:
            raise ValueError("v12 diagnostic must use validation only")
    qwen = _mean(v12, QWEN)
    hybrid = _mean(v12, HYBRID)
    old_qwen = _mean(v11, QWEN)
    edit = _mean(v11, "edit_conditioned_selector")
    gate = v12["protocol"].get("tool_need_gate") or {}
    gate_methods = gate.get("methods", {})
    qwen_calls = int(v12["action_counts"].get(QWEN, {}).get("USE_TOOL", 0))
    hybrid_calls = int(v12["action_counts"].get(HYBRID, {}).get("USE_TOOL", 0))
    checks = {
        "qwen_gate_admits_nonzero": gate_methods.get(QWEN, {}).get("admitted", 0) > 0,
        "hybrid_gate_admits_nonzero": gate_methods.get(HYBRID, {}).get("admitted", 0) > 0,
        "qwen_paired_calls_nonzero": qwen_calls >= 2,
        "hybrid_paired_calls_nonzero": hybrid_calls >= 2,
        "qwen_belief_update_active": qwen["mean_tool_belief_l1_delta"] > 0.0,
        "hybrid_belief_update_active": hybrid["mean_tool_belief_l1_delta"] > 0.0,
        "qwen_utility_above_v11": qwen["mean_quality_cost_utility"]
        > old_qwen["mean_quality_cost_utility"],
        "hybrid_utility_above_edit": hybrid["mean_quality_cost_utility"]
        > edit["mean_quality_cost_utility"],
        "hybrid_false_edit_noninferior": hybrid["false_edit_rate"]
        <= edit["false_edit_rate"] + 0.01,
        "hybrid_sparse_calls": hybrid["mean_tool_calls"] <= 0.10,
    }
    checks["passed"] = all(checks.values())
    return {
        "schema_version": "muno21-proactive-paired-tool-single-seed-v1",
        "metrics": {
            "qwen_proactive": qwen,
            "edit_hybrid_proactive": hybrid,
            "qwen_v11_veto_only": old_qwen,
            "edit_conditioned_selector": edit,
        },
        "gate": gate,
        "tool_calls": {"qwen": qwen_calls, "hybrid": hybrid_calls},
        "checks": checks,
        "diagnostic_passed": checks["passed"],
        "claim_boundary": "Single-seed validation diagnostic; not paper evidence.",
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("v11_summary", type=Path)
    parser.add_argument("v12_summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = assess(
        json.loads(args.v11_summary.read_text(encoding="utf-8")),
        json.loads(args.v12_summary.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
