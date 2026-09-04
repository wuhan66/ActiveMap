#!/usr/bin/env python3
"""Gate single-seed RL on SFT safety and closed-loop executability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FULL_METHOD = "qwen3_4b_sft_tool_to_belief"


def assess_rl_readiness(
    static_decision: dict[str, Any],
    rollout_summary: dict[str, Any],
    reachability_audit: dict[str, Any] | None = None,
    *,
    max_false_edit: float = 0.05,
    max_fallback: float = 0.01,
    min_validity: float = 0.99,
) -> dict[str, Any]:
    if static_decision.get("protocol", {}).get("test_assets_read") is not False:
        raise ValueError("static decision does not prove test isolation")
    validity = rollout_summary.get("llm_validity_by_method", {}).get(FULL_METHOD)
    if not isinstance(validity, dict):
        raise ValueError("rollout summary lacks full-method validity")
    rows = [row for row in rollout_summary.get("results", []) if row.get("method") == FULL_METHOD]
    if not rows:
        raise ValueError(f"rollout summary lacks {FULL_METHOD}")
    total = sum(int(row["sample_count"]) for row in rows)

    def weighted(key: str) -> float:
        return sum(float(row[key]) * int(row["sample_count"]) for row in rows) / total

    metrics = {
        "false_edit_rate": weighted("false_edit_rate"),
        "mean_tool_calls": weighted("mean_tool_calls"),
        "tool_success_rate": weighted("tool_success_rate"),
        "mean_tool_belief_l1_delta": weighted("mean_tool_belief_l1_delta"),
        "tool_action_flip_rate": weighted("tool_action_flip_rate"),
        "terminal_edit_flip_rate": weighted("terminal_edit_flip_rate"),
    }
    gates = {
        "trajectory_reachability": bool(
            reachability_audit
            and reachability_audit.get("passed") is True
            and float(reachability_audit.get("reachable_tool_state_rate", 0.0)) == 1.0
            and float(reachability_audit.get("fully_bridged_tool_task_rate", 0.0)) == 1.0
        ),
        "static_selection_passed": bool(static_decision.get("selection_passed")),
        "schema_valid": float(validity["schema_valid_rate"]) >= min_validity,
        "executable_valid": float(validity["executable_valid_rate"]) >= min_validity,
        "fallback": float(validity["fallback_rate"]) <= max_fallback,
        "absolute_false_edit": metrics["false_edit_rate"] <= max_false_edit,
        "tool_success": metrics["tool_success_rate"] >= min_validity,
        "nonzero_sparse_tool_use": metrics["mean_tool_calls"] > 0.0,
        "recurrent_belief_active": metrics["mean_tool_belief_l1_delta"] > 0.0,
    }
    return {
        "protocol": {
            "scope": "single_seed_rl_readiness",
            "test_assets_read": False,
            "permits": "one contextual constrained GRPO warm-start seed",
            "does_not_establish": "Agent gain or paper promotion",
            "max_false_edit": max_false_edit,
            "max_fallback": max_fallback,
            "min_validity": min_validity,
        },
        "selected_checkpoint": static_decision.get("selected_checkpoint"),
        "trajectory_reachability": reachability_audit,
        "closed_loop_metrics": metrics,
        "gates": gates,
        "ready_for_single_seed_rl": all(gates.values()),
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("static_decision", type=Path)
    parser.add_argument("rollout_summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reachability-audit", type=Path)
    args = parser.parse_args()
    decision = assess_rl_readiness(
        json.loads(args.static_decision.read_text(encoding="utf-8")),
        json.loads(args.rollout_summary.read_text(encoding="utf-8")),
        (
            json.loads(args.reachability_audit.read_text(encoding="utf-8"))
            if args.reachability_audit is not None
            else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
