#!/usr/bin/env python3
"""Apply frozen paired non-inferiority and benefit gates to a DPO intervention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def assess(
    action_comparison: dict[str, Any],
    rollout_comparison: dict[str, Any],
    absolute_decision: dict[str, Any],
    writeback_comparison: dict[str, Any] | None = None,
    *,
    utility_margin: float = 0.01,
    accuracy_margin: float = 0.01,
    false_edit_margin: float = 0.01,
    acquire_recall_margin: float = 0.10,
) -> dict[str, Any]:
    action_delta = action_comparison["paired_delta"]
    rollout_delta = rollout_comparison["paired_delta"]
    utility_metric = (
        "quality_cost_utility_auc"
        if "quality_cost_utility_auc" in rollout_delta
        else "joint_utility_auc"
    )
    gates = {
        "absolute_validation_gates": bool(absolute_decision["promotion_passed"]),
        "joint_utility_noninferior": (
            float(rollout_delta[utility_metric]["ci95_low"]) >= -utility_margin
        ),
        "terminal_accuracy_noninferior": (
            float(rollout_delta["terminal_accuracy"]["ci95_low"]) >= -accuracy_margin
        ),
        "false_edit_noninferior": (
            float(rollout_delta["false_edit_rate"]["ci95_high"]) <= false_edit_margin
        ),
        "acquire_recall_noninferior": (
            float(action_delta["acquire_recall"]["ci95_low"]) >= -acquire_recall_margin
        ),
    }
    if writeback_comparison is not None:
        writeback_delta = writeback_comparison["paired_delta"]
        gates.update(
            {
                "map_raster_iou_noninferior": (
                    float(writeback_delta["raster_iou_auc"]["ci95_low"]) >= -utility_margin
                ),
                "vector_topology_noninferior": (
                    float(writeback_delta["vector_delta_topology_valid_auc"]["ci95_low"])
                    >= -accuracy_margin
                ),
            }
        )
    benefit_evidence = {
        "joint_utility_improved": (float(rollout_delta[utility_metric]["ci95_low"]) > 0.0),
        "false_edit_reduced": (float(rollout_delta["false_edit_rate"]["ci95_high"]) < 0.0),
        "missed_edit_reduced": (float(rollout_delta["missed_edit_rate"]["ci95_high"]) < 0.0),
        "cost_reduced": float(rollout_delta["mean_cost"]["ci95_high"]) < 0.0,
        "macro_f1_improved": float(action_delta["macro_f1"]["ci95_low"]) > 0.0,
    }
    if writeback_comparison is not None:
        writeback_delta = writeback_comparison["paired_delta"]
        benefit_evidence.update(
            {
                "map_raster_iou_improved": (
                    float(writeback_delta["raster_iou_auc"]["ci95_low"]) > 0.0
                ),
                "map_component_error_reduced": (
                    float(writeback_delta["component_count_absolute_error_auc"]["ci95_high"]) < 0.0
                ),
            }
        )
    failures = [name for name, passed in gates.items() if not passed]
    any_benefit = any(benefit_evidence.values())
    if not any_benefit:
        failures.append("no_primary_metric_has_positive_paired_evidence")
    return {
        "protocol": {
            "scope": "single_seed_validation_intervention",
            "paired_confidence": 0.95,
            "test_assets_read": False,
            "noninferiority_margins": {
                utility_metric: utility_margin,
                "terminal_accuracy": accuracy_margin,
                "false_edit_rate": false_edit_margin,
                "acquire_recall": acquire_recall_margin,
            },
            "paper_promotion_requires_three_seeds": True,
            "primary_utility_metric": utility_metric,
            "primary_map_quality_metric": (
                "raster_iou_auc" if writeback_comparison is not None else None
            ),
        },
        "gates": gates,
        "benefit_evidence": benefit_evidence,
        "single_seed_intervention_passed": not failures,
        "failed_gates": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action_comparison", type=Path)
    parser.add_argument("rollout_comparison", type=Path)
    parser.add_argument("absolute_decision", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--writeback-comparison", type=Path)
    parser.add_argument("--utility-margin", type=float, default=0.01)
    parser.add_argument("--accuracy-margin", type=float, default=0.01)
    parser.add_argument("--false-edit-margin", type=float, default=0.01)
    parser.add_argument("--acquire-recall-margin", type=float, default=0.10)
    args = parser.parse_args()
    result = assess(
        json.loads(args.action_comparison.read_text(encoding="utf-8")),
        json.loads(args.rollout_comparison.read_text(encoding="utf-8")),
        json.loads(args.absolute_decision.read_text(encoding="utf-8")),
        (
            json.loads(args.writeback_comparison.read_text(encoding="utf-8"))
            if args.writeback_comparison is not None
            else None
        ),
        utility_margin=args.utility_margin,
        accuracy_margin=args.accuracy_margin,
        false_edit_margin=args.false_edit_margin,
        acquire_recall_margin=args.acquire_recall_margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
