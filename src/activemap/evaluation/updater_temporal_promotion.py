"""Validation-only promotion and cross-seed summaries for temporal updater heads."""

from __future__ import annotations

from statistics import fmean, stdev
from typing import Any


def select_temporal_checkpoint(
    calibrations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Promote the strongest safety-feasible checkpoint at its calibrated thresholds."""

    if not calibrations:
        raise ValueError("at least one checkpoint calibration is required")
    candidates = []
    reference: tuple[str, str, int] | None = None
    for checkpoint_name, calibration in sorted(calibrations.items()):
        if calibration.get("split") != "val":
            raise ValueError("temporal checkpoint promotion requires validation calibration")
        identity = (
            str(calibration.get("samples")),
            str(calibration.get("split")),
            int(calibration.get("sample_count", 0)),
        )
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError("candidate calibrations must use the same validation samples")
        selected = calibration["selected"]
        candidates.append(
            {
                "checkpoint_name": checkpoint_name,
                "checkpoint": calibration["checkpoint"],
                "checkpoint_epoch": int(calibration["checkpoint_epoch"]),
                "constraint_satisfied": bool(calibration["constraint_satisfied"]),
                "selected": selected,
            }
        )
    feasible = [row for row in candidates if row["constraint_satisfied"]]
    if not feasible:
        raise ValueError("no candidate satisfies stable-map false-positive constraints")

    def rank(row: dict[str, Any]) -> tuple[float, float, float, str]:
        selected = row["selected"]
        add_iou = float(selected["add"]["mean_positive_iou"])
        remove_iou = float(selected["remove"]["mean_positive_iou"])
        stable_fp = float(selected["add"]["stable_false_positive_fraction"]) + float(
            selected["remove"]["stable_false_positive_fraction"]
        )
        return (
            float(selected["harmonic_mean_positive_iou"]),
            min(add_iou, remove_iou),
            -stable_fp,
            str(row["checkpoint_name"]),
        )

    promoted = max(feasible, key=rank)
    assert reference is not None
    return {
        "status": "promoted_validation_only",
        "selection_objective": (
            "maximize calibrated ADD/REMOVE harmonic mean under per-channel "
            "stable-map false-positive constraints"
        ),
        "samples": reference[0],
        "split": reference[1],
        "sample_count": reference[2],
        "promoted_checkpoint": promoted["checkpoint_name"],
        "promoted": promoted,
        "candidates": candidates,
        "test_evaluation": None,
    }


def aggregate_temporal_seed_decisions(
    decisions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Summarize promoted validation operating points across random seeds."""

    if not decisions:
        raise ValueError("at least one seed decision is required")
    if any(decision.get("test_evaluation") is not None for decision in decisions.values()):
        raise ValueError("validation seed aggregation must not contain test evaluations")
    rows = []
    for seed, decision in sorted(decisions.items()):
        promoted = decision["promoted"]
        selected = promoted["selected"]
        rows.append(
            {
                "seed": seed,
                "checkpoint_name": promoted["checkpoint_name"],
                "checkpoint_epoch": int(promoted["checkpoint_epoch"]),
                "add_threshold": float(selected["add_threshold"]),
                "remove_threshold": float(selected["remove_threshold"]),
                "added_change_iou": float(selected["add"]["mean_positive_iou"]),
                "removed_change_iou": float(selected["remove"]["mean_positive_iou"]),
                "temporal_change_harmonic_iou": float(selected["harmonic_mean_positive_iou"]),
                "stable_add_false_positive": float(
                    selected["add"]["stable_false_positive_fraction"]
                ),
                "stable_remove_false_positive": float(
                    selected["remove"]["stable_false_positive_fraction"]
                ),
            }
        )
    metric_names = (
        "added_change_iou",
        "removed_change_iou",
        "temporal_change_harmonic_iou",
        "stable_add_false_positive",
        "stable_remove_false_positive",
    )
    aggregate = {}
    for metric in metric_names:
        values = [float(row[metric]) for row in rows]
        aggregate[metric] = {
            "mean": fmean(values),
            "sample_std": stdev(values) if len(values) > 1 else 0.0,
            "minimum": min(values),
            "maximum": max(values),
        }
    return {
        "seed_count": len(rows),
        "selection_scope": "validation_only",
        "test_evaluation_count": 0,
        "seeds": rows,
        "aggregate": aggregate,
    }


def aggregate_temporal_validation_evaluations(
    evaluations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate standard validation summaries at frozen per-seed operating points."""

    if not evaluations:
        raise ValueError("at least one validation evaluation is required")
    rows = []
    sample_count: int | None = None
    for seed, evaluation in sorted(evaluations.items()):
        if evaluation.get("split") != "val":
            raise ValueError("temporal aggregation accepts validation summaries only")
        count = int(evaluation["sample_count"])
        if sample_count is None:
            sample_count = count
        elif count != sample_count:
            raise ValueError("validation summaries must contain the same sample count")
        add_iou = float(evaluation["temporal_change"]["mean_added_iou"])
        remove_iou = float(evaluation["temporal_change"]["mean_removed_iou"])
        harmonic = (
            2.0 * add_iou * remove_iou / (add_iou + remove_iou) if add_iou + remove_iou else 0.0
        )
        rows.append(
            {
                "seed": seed,
                "checkpoint": str(evaluation["checkpoint"]),
                "add_threshold": float(evaluation["add_threshold"]),
                "remove_threshold": float(evaluation["remove_threshold"]),
                "mean_raster_iou": float(evaluation["mean_raster_iou"]),
                "added_change_iou": add_iou,
                "removed_change_iou": remove_iou,
                "temporal_change_harmonic_iou": harmonic,
                "edit_accuracy": float(evaluation["edit_accuracy"]),
                "macro_f1": float(evaluation["macro_f1"]),
                "false_edit_rate": float(evaluation["false_edit_rate"]),
                "missed_update_rate": float(evaluation["missed_update_rate"]),
            }
        )
    metric_names = (
        "mean_raster_iou",
        "added_change_iou",
        "removed_change_iou",
        "temporal_change_harmonic_iou",
        "edit_accuracy",
        "macro_f1",
        "false_edit_rate",
        "missed_update_rate",
    )
    aggregate = {}
    for metric in metric_names:
        values = [float(row[metric]) for row in rows]
        aggregate[metric] = {
            "mean": fmean(values),
            "sample_std": stdev(values) if len(values) > 1 else 0.0,
            "minimum": min(values),
            "maximum": max(values),
        }
    return {
        "seed_count": len(rows),
        "sample_count_per_seed": sample_count,
        "selection_scope": "validation_only",
        "test_evaluation_count": 0,
        "seeds": rows,
        "aggregate": aggregate,
    }
