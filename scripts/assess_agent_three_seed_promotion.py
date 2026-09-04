#!/usr/bin/env python3
"""Gate three-seed Agent claims on utility, safety, and recurrent tool use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def assess(report: dict[str, Any]) -> dict[str, Any]:
    protocol = report.get("protocol", {})
    if protocol.get("test_assets_read") is not False:
        raise ValueError("aggregate does not prove test isolation")
    seeds = protocol.get("model_seeds", [])
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("promotion requires exactly three unique model seeds")
    comparisons = report["comparisons"]
    required = {
        "edit_conditioned_selector",
        "generic_selector",
        "qwen3_4b_sft_tools_no_belief",
        "forced_tools",
    }
    missing = sorted(required - set(comparisons))
    if missing:
        raise ValueError(f"missing three-seed comparisons: {missing}")

    edit = comparisons["edit_conditioned_selector"]
    no_belief = comparisons["qwen3_4b_sft_tools_no_belief"]
    forced = comparisons["forced_tools"]
    generic = comparisons["generic_selector"]
    candidate_rows = edit["per_seed"]
    candidate_mean = {
        name: sum(float(row["candidate"][name]) for row in candidate_rows)
        / len(candidate_rows)
        for name in candidate_rows[0]["candidate"]
    }

    def interval(comparison: dict[str, Any], metric: str) -> dict[str, float]:
        return comparison["paired_delta"][metric]

    primary_metric = str(
        protocol.get("primary_metric", "quality_cost_utility_auc")
    )
    if any(primary_metric not in comparison["paired_delta"] for comparison in comparisons.values()):
        raise ValueError(f"aggregate is missing primary metric: {primary_metric}")

    gates = {
        "absolute_false_edit": candidate_mean["false_edit_rate"] <= 0.05,
        "nonzero_tool_use": candidate_mean["mean_tool_calls"] > 0.0,
        "recurrent_belief_active": candidate_mean["mean_tool_belief_l1_delta"] > 0.0,
        "vs_edit_utility_positive_ci": interval(edit, primary_metric)["ci95_low"] > 0.0,
        "vs_edit_false_edit_noninferior": interval(edit, "false_edit_rate")["ci95_high"] <= 0.01,
        "vs_generic_utility_positive_ci": interval(generic, primary_metric)["ci95_low"] > 0.0,
        "vs_no_belief_utility_positive_ci": interval(no_belief, primary_metric)["ci95_low"] > 0.0,
        "vs_forced_utility_positive_ci": interval(forced, primary_metric)["ci95_low"] > 0.0,
    }
    return {
        "schema_version": "activemap-agent-three-seed-promotion-v1",
        "protocol": {
            "split": "val",
            "test_assets_read": False,
            "model_seeds": seeds,
            "primary_metric": primary_metric,
        },
        "candidate_mean": candidate_mean,
        "gates": gates,
        "promote": all(gates.values()),
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    decision = assess(json.loads(args.aggregate.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
