"""Map-update metrics for typed edits, geometry quality, and calibration."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from activemap.models import EditOperation, GeoJSONGeometry, StrictModel

EDIT_LABELS = tuple(EditOperation)
OBJECT_SCALE_BINS = (
    ("tiny", 0.0, 0.001),
    ("small", 0.001, 0.005),
    ("medium", 0.005, 0.02),
    ("large", 0.02, float("inf")),
)


class UpdatePrediction(StrictModel):
    """One auditable updater prediction used by every evaluation backend."""

    sample_id: str
    aoi_id: str
    target_edit: EditOperation
    predicted_edit: EditOperation
    confidence: float = Field(ge=0.0, le=1.0)
    committed: bool = True
    raster_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    polygon_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    topology_valid: bool | None = None
    object_id: str | None = None
    predicted_geometry: GeoJSONGeometry | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def segmentation_strata_metrics(records: list[UpdatePrediction]) -> dict[str, Any]:
    foreground = [
        record
        for record in records
        if bool(record.metadata.get("target_foreground", False))
    ]
    empty = [
        record
        for record in records
        if record.metadata.get("target_foreground") is False
    ]
    foreground_ious = [
        float(record.raster_iou)
        for record in foreground
        if record.raster_iou is not None
    ]
    empty_false_positive = [
        float(record.metadata["empty_scene_false_positive_fraction"])
        for record in empty
        if record.metadata.get("empty_scene_false_positive_fraction") is not None
    ]
    return {
        "foreground_sample_count": len(foreground),
        "empty_sample_count": len(empty),
        "mean_foreground_iou": (
            sum(foreground_ious) / len(foreground_ious) if foreground_ious else None
        ),
        "mean_empty_scene_false_positive_fraction": (
            sum(empty_false_positive) / len(empty_false_positive)
            if empty_false_positive
            else None
        ),
    }


def object_scale_strata_metrics(records: list[UpdatePrediction]) -> dict[str, Any]:
    """Report update quality by target area fraction without resolution-specific bins."""

    available = [
        record
        for record in records
        if record.metadata.get("target_foreground_fraction") is not None
    ]
    strata: dict[str, Any] = {}
    for name, lower, upper in OBJECT_SCALE_BINS:
        selected = [
            record
            for record in available
            if float(record.metadata["target_foreground_fraction"]) > lower
            and float(record.metadata["target_foreground_fraction"]) <= upper
        ]
        raster_ious = [
            float(record.raster_iou)
            for record in selected
            if record.raster_iou is not None
        ]
        polygon_ious = [
            float(record.polygon_iou)
            for record in selected
            if record.polygon_iou is not None
        ]
        correct = [
            (
                record.predicted_edit if record.committed else EditOperation.KEEP
            )
            == record.target_edit
            for record in selected
        ]
        strata[name] = {
            "target_fraction_gt": lower,
            "target_fraction_lte": None if not np.isfinite(upper) else upper,
            "sample_count": len(selected),
            "mean_raster_iou": float(np.mean(raster_ious)) if raster_ious else None,
            "mean_polygon_iou": float(np.mean(polygon_ious)) if polygon_ious else None,
            "typed_edit_accuracy": float(np.mean(correct)) if correct else None,
        }
    return {
        "binning": "target_foreground_pixels / valid_pixels",
        "available_sample_count": len(available),
        "missing_fraction_sample_count": len(records) - len(available),
        "strata": strata,
    }


def load_update_predictions(path: Path) -> list[UpdatePrediction]:
    records: list[UpdatePrediction] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(UpdatePrediction.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid update prediction at line {line_number}") from exc
    if not records:
        raise ValueError(f"no update predictions found in {path}")
    return records


def write_update_predictions(records: Iterable[UpdatePrediction], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
            count += 1
    return count


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _confusion(records: list[UpdatePrediction]) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(EDIT_LABELS)}
    matrix = np.zeros((len(EDIT_LABELS), len(EDIT_LABELS)), dtype=np.int64)
    for record in records:
        predicted = record.predicted_edit if record.committed else EditOperation.KEEP
        matrix[label_to_index[record.target_edit], label_to_index[predicted]] += 1
    return matrix


def calibration_metrics(
    confidence: np.ndarray,
    correctness: np.ndarray,
    *,
    bins: int = 15,
) -> dict[str, float]:
    """Return ECE, Brier score, and negative log likelihood."""

    confidence = np.asarray(confidence, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.float64)
    if confidence.shape != correctness.shape or confidence.ndim != 1:
        raise ValueError("confidence and correctness must be one-dimensional and aligned")
    if not len(confidence):
        raise ValueError("calibration requires at least one prediction")
    confidence = np.clip(confidence, 1e-7, 1.0 - 1e-7)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(correctness[selected].mean()) - float(confidence[selected].mean())
            )
    brier = float(np.mean((confidence - correctness) ** 2))
    nll = float(
        -np.mean(
            correctness * np.log(confidence) + (1.0 - correctness) * np.log(1.0 - confidence)
        )
    )
    return {"ece": ece, "brier": brier, "nll": nll}


def risk_coverage_curve(records: Iterable[UpdatePrediction]) -> list[dict[str, float]]:
    values = list(records)
    if not values:
        raise ValueError("risk-coverage evaluation requires at least one prediction")
    ordered = sorted(values, key=lambda item: item.confidence, reverse=True)
    errors = np.asarray(
        [
            float(
                item.target_edit
                != (item.predicted_edit if item.committed else EditOperation.KEEP)
            )
            for item in ordered
        ]
    )
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return [
        {
            "coverage": float((index + 1) / len(ordered)),
            "risk": float(cumulative_risk[index]),
            "threshold": float(ordered[index].confidence),
        }
        for index in range(len(ordered))
    ]


def _optional_mean(records: list[UpdatePrediction], field: str) -> float | None:
    values = [getattr(record, field) for record in records if getattr(record, field) is not None]
    return float(np.mean(values)) if values else None


def evaluate_updates(
    records: Iterable[UpdatePrediction],
    *,
    calibration_bins: int = 15,
) -> dict[str, Any]:
    """Evaluate stable-map preservation and positive update quality together."""

    values = list(records)
    if not values:
        raise ValueError("update evaluation requires at least one prediction")
    confusion = _confusion(values)
    per_edit: dict[str, dict[str, float | int]] = {}
    class_f1: list[float] = []
    for index, label in enumerate(EDIT_LABELS):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        class_f1.append(f1)
        per_edit[label.value] = {
            "support": int(confusion[index, :].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    true_update = np.asarray([record.target_edit != EditOperation.KEEP for record in values])
    predicted_update = np.asarray(
        [record.committed and record.predicted_edit != EditOperation.KEEP for record in values]
    )
    update_tp = int(np.sum(true_update & predicted_update))
    update_fp = int(np.sum(~true_update & predicted_update))
    update_fn = int(np.sum(true_update & ~predicted_update))
    update_precision = _safe_divide(update_tp, update_tp + update_fp)
    update_recall = _safe_divide(update_tp, update_tp + update_fn)
    update_f1 = _safe_divide(
        2.0 * update_precision * update_recall, update_precision + update_recall
    )
    exact = np.asarray(
        [
            record.target_edit
            == (record.predicted_edit if record.committed else EditOperation.KEEP)
            for record in values
        ],
        dtype=np.float64,
    )
    confidence = np.asarray([record.confidence for record in values], dtype=np.float64)
    calibration = calibration_metrics(confidence, exact, bins=calibration_bins)
    curve = risk_coverage_curve(values)
    coverages = np.asarray([point["coverage"] for point in curve])
    risks = np.asarray([point["risk"] for point in curve])
    aurc = (
        float(np.sum((risks[1:] + risks[:-1]) * np.diff(coverages) * 0.5))
        if len(curve) > 1
        else float(risks[0])
    )

    committed_updates = [
        record
        for record in values
        if record.committed and record.predicted_edit != EditOperation.KEEP
    ]
    topology = [
        float(record.topology_valid)
        for record in committed_updates
        if record.topology_valid is not None
    ]
    return {
        "sample_count": len(values),
        "aoi_count": len({record.aoi_id for record in values}),
        "edit_accuracy": float(exact.mean()),
        "macro_f1": float(np.mean(class_f1)),
        "stable_f1": float(per_edit[EditOperation.KEEP.value]["f1"]),
        "update_precision": update_precision,
        "update_recall": update_recall,
        "update_f1": update_f1,
        "false_edit_rate": _safe_divide(update_fp, int(np.sum(~true_update))),
        "missed_update_rate": _safe_divide(update_fn, int(np.sum(true_update))),
        "commit_precision": _safe_divide(
            sum(record.target_edit == record.predicted_edit for record in committed_updates),
            len(committed_updates),
        ),
        "mean_raster_iou": _optional_mean(values, "raster_iou"),
        "mean_polygon_iou": _optional_mean(values, "polygon_iou"),
        "topology_valid_rate": float(np.mean(topology)) if topology else None,
        "ece": calibration["ece"],
        "brier": calibration["brier"],
        "nll": calibration["nll"],
        "aurc": aurc,
        "per_edit": per_edit,
        "confusion": {
            target.value: {
                predicted.value: int(confusion[row, column])
                for column, predicted in enumerate(EDIT_LABELS)
            }
            for row, target in enumerate(EDIT_LABELS)
        },
        "risk_coverage": curve,
    }


def select_commit_threshold(
    records: Iterable[UpdatePrediction],
    *,
    thresholds: Iterable[float] | None = None,
    false_edit_weight: float = 1.0,
) -> dict[str, Any]:
    """Choose a threshold on validation data, balancing updates and map stability."""

    values = list(records)
    if not values:
        raise ValueError("threshold selection requires at least one prediction")
    candidates = (
        [float(value) for value in thresholds]
        if thresholds is not None
        else np.linspace(0.0, 1.0, 101).tolist()
    )
    curve: list[dict[str, float]] = []
    for threshold in candidates:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("thresholds must be between zero and one")
        thresholded = [
            record.model_copy(update={"committed": record.confidence >= threshold})
            for record in values
        ]
        metrics = evaluate_updates(thresholded)
        objective = (
            float(metrics["update_f1"])
            + float(metrics["stable_f1"])
            - false_edit_weight * float(metrics["false_edit_rate"])
        )
        curve.append(
            {
                "threshold": threshold,
                "objective": objective,
                "update_f1": float(metrics["update_f1"]),
                "stable_f1": float(metrics["stable_f1"]),
                "false_edit_rate": float(metrics["false_edit_rate"]),
            }
        )
    best = max(curve, key=lambda row: (row["objective"], row["threshold"]))
    return {"best_threshold": best["threshold"], "best": best, "curve": curve}


def save_update_evaluation(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
