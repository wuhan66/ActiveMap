"""Rank auditable updater failures without rerunning model inference."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from activemap.evaluation.update import UpdatePrediction
from activemap.models import EditOperation


def select_failure_cases(
    records: list[UpdatePrediction], *, per_category: int = 12
) -> list[dict[str, Any]]:
    """Select severe, deterministic examples for each updater failure mode."""

    if per_category < 1:
        raise ValueError("per_category must be positive")
    categories: tuple[
        tuple[str, Callable[[UpdatePrediction], bool], Callable[[UpdatePrediction], float]],
        ...,
    ] = (
        (
            "false_edit",
            lambda record: record.target_edit == EditOperation.KEEP
            and record.predicted_edit != EditOperation.KEEP,
            lambda record: record.confidence,
        ),
        (
            "missed_update",
            lambda record: record.target_edit != EditOperation.KEEP
            and record.predicted_edit == EditOperation.KEEP,
            lambda record: record.confidence,
        ),
        (
            "wrong_update_type",
            lambda record: record.target_edit != EditOperation.KEEP
            and record.predicted_edit != EditOperation.KEEP
            and record.target_edit != record.predicted_edit,
            lambda record: record.confidence,
        ),
        (
            "low_geometry_iou",
            lambda record: record.target_edit in {EditOperation.ADD, EditOperation.RESHAPE}
            and record.predicted_edit == record.target_edit
            and record.polygon_iou is not None,
            lambda record: 1.0 - float(record.polygon_iou or 0.0),
        ),
        (
            "topology_invalid",
            lambda record: record.topology_valid is False,
            lambda record: record.confidence,
        ),
    )
    selected = []
    for category, predicate, severity in categories:
        candidates = [record for record in records if predicate(record)]
        candidates.sort(key=lambda record: (-severity(record), record.sample_id))
        for record in candidates[:per_category]:
            selected.append(
                {
                    "failure_category": category,
                    "severity": severity(record),
                    "prediction": record.model_dump(mode="json"),
                }
            )
    return selected
