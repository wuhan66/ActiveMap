#!/usr/bin/env python3
"""Select a validation-only writeback confidence veto from cached executable rows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from activemap.evaluation.episode_utility import score_episode_profiles
from activemap.models import EditOperation


def _operation(value: str) -> EditOperation:
    if value == "REJECT":
        return EditOperation.KEEP
    if value.startswith("COMMIT:"):
        return EditOperation(value.split(":", 1)[1])
    return EditOperation(value)


def _cached_effective_operation(row: dict[str, Any]) -> EditOperation:
    if row.get("effective_operation") is not None:
        return EditOperation(str(row["effective_operation"]))
    added = row.get("predicted_add_geometry") is not None
    removed = row.get("predicted_remove_geometry") is not None
    if added and removed:
        return EditOperation.RESHAPE
    if added:
        return EditOperation.ADD
    if removed:
        return EditOperation.DELETE
    return EditOperation.KEEP


def _fused_confidence(row: dict[str, Any]) -> float:
    if row.get("fused_confidence") is not None:
        return float(row["fused_confidence"])
    weights = np.asarray(row.get("fusion_weights", []), dtype=np.float64)
    if not len(weights):
        return 1.0
    return float(np.sum(weights * weights) / np.sum(weights))


def _errors(
    target: EditOperation, prediction: EditOperation
) -> tuple[bool, bool, bool]:
    return (
        target == EditOperation.KEEP and prediction != target,
        target != EditOperation.KEEP and prediction == EditOperation.KEEP,
        target != EditOperation.KEEP
        and prediction != EditOperation.KEEP
        and prediction != target,
    )


def _auc(rows: list[dict[str, float]], metric: str) -> float:
    by_budget: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        by_budget[float(row["budget"])].append(float(row[metric]))
    points = [(budget, float(np.mean(values))) for budget, values in sorted(by_budget.items())]
    if len(points) == 1:
        return points[0][1]
    area = sum(
        (right[0] - left[0]) * (left[1] + right[1]) / 2.0
        for left, right in pairwise(points)
    )
    return float(area / (points[-1][0] - points[0][0]))


def _evaluate_floor(source: list[dict[str, Any]], floor: float) -> dict[str, Any]:
    evaluated = []
    vetoed = 0
    for row in source:
        target = _operation(str(row["target"]))
        cached_operation = _cached_effective_operation(row)
        veto = _fused_confidence(row) < floor and cached_operation != EditOperation.KEEP
        effective = EditOperation.KEEP if veto else cached_operation
        final_quality = (
            float(row["prior_raster_iou"]) if veto else float(row["raster_iou"])
        )
        false_edit, missed_edit, wrong_edit = _errors(target, effective)
        utility = score_episode_profiles(
            final_map_quality=final_quality,
            prior_map_quality=float(row["prior_raster_iou"]),
            spent_cost=float(row.get("spent_cost", 0.0)),
            budget=float(row["budget"]),
            false_edit=false_edit,
            missed_edit=missed_edit,
            wrong_edit=wrong_edit,
            topology_valid=True if veto else bool(row["vector_delta_topology_valid"]),
        )
        vetoed += int(veto)
        evaluated.append(
            {
                "budget": float(row["budget"]),
                "raster_iou": final_quality,
                "raster_iou_gain": final_quality - float(row["prior_raster_iou"]),
                "false_edit": float(false_edit),
                "missed_edit": float(missed_edit),
                "wrong_edit": float(wrong_edit),
                "balanced": utility["balanced"]["value"],
                "safety": utility["safety"]["value"],
                "cost_aware": utility["cost_aware"]["value"],
            }
        )
    metrics = {
        name: _auc(evaluated, name)
        for name in (
            "raster_iou",
            "raster_iou_gain",
            "false_edit",
            "missed_edit",
            "wrong_edit",
            "balanced",
            "safety",
            "cost_aware",
        )
    }
    return {
        "confidence_floor": floor,
        "vetoed_rows": vetoed,
        "veto_rate": vetoed / len(source),
        **{f"{name}_auc": value for name, value in metrics.items()},
    }


def sweep(
    path: Path, floors: list[float], *, min_retained_change_rate: float = 0.5
) -> dict[str, Any]:
    if not floors or any(not 0.0 <= value <= 1.0 for value in floors):
        raise ValueError("confidence floors must be between zero and one")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(bool(row.get("test_assets_read")) for row in rows):
        raise ValueError("sweep requires non-empty validation-only writeback rows")
    if not 0.0 < min_retained_change_rate <= 1.0:
        raise ValueError("min_retained_change_rate must be in (0, 1]")
    baseline_changed = sum(
        _cached_effective_operation(row) != EditOperation.KEEP for row in rows
    )
    if not baseline_changed:
        raise ValueError("confidence sweep requires at least one executable map change")
    candidates = [_evaluate_floor(rows, floor) for floor in sorted(set(floors))]
    baseline = next((row for row in candidates if row["confidence_floor"] == 0.0), None)
    if baseline is None:
        raise ValueError("confidence floors must include zero")
    for row in candidates:
        row["retained_changed_rows"] = baseline_changed - row["vetoed_rows"]
        row["retained_change_rate"] = row["retained_changed_rows"] / baseline_changed
        row["feasible"] = row["retained_change_rate"] >= min_retained_change_rate
    feasible = [row for row in candidates if row["feasible"]]
    selected = max(
        feasible,
        key=lambda row: (
            row["balanced_auc"],
            row["raster_iou_auc"],
            -row["false_edit_auc"],
            -row["confidence_floor"],
        ),
    )
    return {
        "schema_version": "writeback-confidence-gate-sweep-v1",
        "selection_split": "validation",
        "source_writeback": str(path.resolve()),
        "sample_count": len(rows),
        "selection_rule": "maximize balanced Utility v2 AUC, then map IoU and safety",
        "baseline_floor": 0.0,
        "baseline_changed_rows": baseline_changed,
        "minimum_retained_change_rate": min_retained_change_rate,
        "candidates": candidates,
        "selected_confidence_floor": selected["confidence_floor"],
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("writeback", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--floor", type=float, action="append", default=[0.0, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    parser.add_argument("--min-retained-change-rate", type=float, default=0.5)
    args = parser.parse_args()
    result = sweep(
        args.writeback,
        args.floor,
        min_retained_change_rate=args.min_retained_change_rate,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
