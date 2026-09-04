#!/usr/bin/env python3
"""Apply frozen single-seed gates to the sparse Tool-to-Belief Agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FULL_METHOD = "qwen3_4b_sft_tool_to_belief"


def assess_tool_agent(
    rollout_summary: dict[str, Any],
    comparisons: dict[str, dict[str, Any]],
    *,
    writeback_comparison: dict[str, Any] | None = None,
    max_false_edit: float = 0.05,
    max_fallback: float = 0.01,
    min_validity: float = 0.99,
    max_false_edit_delta: float = 0.01,
) -> dict[str, Any]:
    full_rows = [
        row for row in rollout_summary["results"] if row["method"] == FULL_METHOD
    ]
    if not full_rows:
        raise ValueError(f"rollout summary lacks {FULL_METHOD}")
    validity = rollout_summary.get("llm_validity_by_method", {}).get(FULL_METHOD)
    if not isinstance(validity, dict):
        raise ValueError("rollout summary lacks full-method validity")

    weighted = lambda key: sum(  # noqa: E731
        float(row[key]) * int(row["sample_count"]) for row in full_rows
    ) / sum(int(row["sample_count"]) for row in full_rows)
    full_metrics = {
        "false_edit_rate": weighted("false_edit_rate"),
        "mean_tool_calls": weighted("mean_tool_calls"),
        "tool_success_rate": weighted("tool_success_rate"),
        "mean_tool_belief_l1_delta": weighted("mean_tool_belief_l1_delta"),
        "tool_action_flip_rate": weighted("tool_action_flip_rate"),
        "terminal_edit_flip_rate": weighted("terminal_edit_flip_rate"),
    }

    required = {"selector", "no_belief", "forced"}
    missing = sorted(required - set(comparisons))
    if missing:
        raise ValueError(f"missing paired comparisons: {missing}")

    def interval(name: str, metric: str) -> dict[str, float]:
        return comparisons[name]["paired_delta"][metric]

    selector_utility = interval("selector", "quality_cost_utility_auc")
    selector_false_edit = interval("selector", "false_edit_rate")
    recurrent_utility = interval("no_belief", "quality_cost_utility_auc")
    sparse_utility = interval("forced", "quality_cost_utility_auc")
    gates = {
        "schema_valid": float(validity["schema_valid_rate"]) >= min_validity,
        "executable_valid": float(validity["executable_valid_rate"]) >= min_validity,
        "fallback": float(validity["fallback_rate"]) <= max_fallback,
        "absolute_false_edit": full_metrics["false_edit_rate"] <= max_false_edit,
        "tool_success": full_metrics["tool_success_rate"] >= min_validity,
        "nonzero_sparse_tool_use": full_metrics["mean_tool_calls"] > 0.0,
        "recurrent_belief_active": full_metrics["mean_tool_belief_l1_delta"] > 0.0,
        "selector_utility_positive_ci": selector_utility["ci95_low"] > 0.0,
        "selector_false_edit_noninferior": (
            selector_false_edit["ci95_high"] <= max_false_edit_delta
        ),
        "recurrent_belief_utility_positive_ci": recurrent_utility["ci95_low"] > 0.0,
        "sparse_stopping_utility_positive_ci": sparse_utility["ci95_low"] > 0.0,
    }
    writeback_evidence = None
    if writeback_comparison is not None:
        writeback = writeback_comparison["paired_delta"]
        raster = writeback["raster_iou_auc"]
        topology = writeback["vector_delta_topology_valid_auc"]
        map_benefits = [
            writeback[name]["ci95_low"] > 0.0
            for name in (
                "raster_iou_auc",
                "added_polygon_iou_auc",
                "removed_polygon_iou_auc",
            )
        ]
        map_benefits.append(
            writeback["component_count_absolute_error_auc"]["ci95_high"] < 0.0
        )
        gates.update(
            {
                "writeback_raster_noninferior": raster["ci95_low"] >= -0.005,
                "writeback_topology_noninferior": topology["ci95_low"] >= -0.001,
                "writeback_benefit_positive_ci": any(map_benefits),
            }
        )
        writeback_evidence = {
            "raster_iou": raster,
            "topology_valid": topology,
            "component_error": writeback["component_count_absolute_error_auc"],
            "positive_ci_benefit": any(map_benefits),
        }
    return {
        "protocol": {
            "scope": "single_seed_validation_tool_agent",
            "test_assets_read": False,
            "paper_claim_requires_three_seeds": True,
            "max_false_edit": max_false_edit,
            "max_fallback": max_fallback,
            "min_validity": min_validity,
            "max_false_edit_delta": max_false_edit_delta,
            "primary_metric": "quality_cost_utility_auc",
        },
        "full_method": FULL_METHOD,
        "full_metrics": full_metrics,
        "paired_evidence": {
            "vs_selector": {
                "utility": selector_utility,
                "false_edit": selector_false_edit,
            },
            "vs_no_belief": {"utility": recurrent_utility},
            "vs_forced": {"utility": sparse_utility},
            "writeback_vs_selector": writeback_evidence,
        },
        "gates": gates,
        "single_seed_intervention_passed": all(gates.values()),
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--selector-comparison", type=Path, required=True)
    parser.add_argument("--no-belief-comparison", type=Path, required=True)
    parser.add_argument("--forced-comparison", type=Path, required=True)
    parser.add_argument("--writeback-comparison", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.rollout_summary.read_text(encoding="utf-8"))
    comparisons = {
        "selector": json.loads(args.selector_comparison.read_text(encoding="utf-8")),
        "no_belief": json.loads(args.no_belief_comparison.read_text(encoding="utf-8")),
        "forced": json.loads(args.forced_comparison.read_text(encoding="utf-8")),
    }
    writeback = json.loads(args.writeback_comparison.read_text(encoding="utf-8"))
    decision = assess_tool_agent(
        summary,
        comparisons,
        writeback_comparison=writeback,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
