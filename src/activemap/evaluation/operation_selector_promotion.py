"""Validation-only checkpoint promotion and seed aggregation for operation selectors."""

from __future__ import annotations

from typing import Any

import numpy as np


def select_operation_selector_checkpoint(
    calibrations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not calibrations:
        raise ValueError("at least one operation-selector calibration is required")
    candidates = []
    for name, calibration in calibrations.items():
        if calibration.get("test_evaluation") is not None:
            raise ValueError("checkpoint promotion must not include test evaluation")
        selected = calibration["selected"]
        candidates.append(
            {
                "checkpoint_name": name,
                "constraint_satisfied": bool(calibration["constraint_satisfied"]),
                "sample_count": int(calibration["sample_count"]),
                "selected": selected,
            }
        )
    feasible = [candidate for candidate in candidates if candidate["constraint_satisfied"]]
    pool = feasible or candidates
    promoted = max(
        pool,
        key=lambda candidate: (
            candidate["selected"]["macro_f1"],
            -candidate["selected"]["missed_update_rate"],
            -candidate["selected"]["false_edit_rate"],
        ),
    )
    return {
        "status": "promoted_validation_only",
        "selection_objective": "maximize calibrated macro F1 under false-edit constraint",
        "promoted_checkpoint": promoted["checkpoint_name"],
        "promoted": promoted,
        "candidates": sorted(candidates, key=lambda value: value["checkpoint_name"]),
        "test_evaluation": None,
    }


def aggregate_operation_selector_seeds(
    promotions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if len(promotions) < 2:
        raise ValueError("seed aggregation requires at least two promotions")
    rows = []
    for seed, promotion in sorted(promotions.items()):
        if promotion.get("test_evaluation") is not None:
            raise ValueError("validation aggregation must not include test evaluation")
        rows.append(
            {
                "seed": seed,
                "checkpoint": promotion["promoted_checkpoint"],
                **promotion["promoted"]["selected"],
            }
        )
    metric_names = [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "false_edit_rate",
        "missed_update_rate",
        "f1_keep",
        "f1_add",
        "f1_delete",
        "f1_reshape",
        "update_threshold",
    ]
    aggregate = {}
    for metric in metric_names:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "values": values.tolist(),
        }
    return {
        "status": "validation_only",
        "seed_count": len(rows),
        "seeds": rows,
        "aggregate": aggregate,
        "all_constraints_satisfied": all(
            promotion["promoted"]["constraint_satisfied"] for promotion in promotions.values()
        ),
        "test_evaluation": None,
    }
