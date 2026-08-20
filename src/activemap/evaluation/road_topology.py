"""Tolerance-aware road connectivity metrics for raster map updates."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage


def _component_consistency(
    source_labels: np.ndarray,
    destination_mask: np.ndarray,
    *,
    tolerance_pixels: int,
) -> float:
    source_count = int(source_labels.max(initial=0))
    source_foreground = source_labels > 0
    total = int(source_foreground.sum())
    if total == 0:
        return 1.0
    if not bool(destination_mask.any()):
        return 0.0
    destination_labels, _ = ndimage.label(
        destination_mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    distance, nearest = ndimage.distance_transform_edt(
        ~destination_mask, return_indices=True
    )
    nearest_labels = destination_labels[tuple(nearest)]
    matched = distance <= tolerance_pixels
    preserved = 0
    for component in range(1, source_count + 1):
        selected = source_labels == component
        labels = nearest_labels[selected & matched]
        labels = labels[labels > 0]
        if labels.size:
            preserved += int(np.bincount(labels).max())
    return preserved / total


def road_connectivity_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    tolerance_pixels: int = 2,
) -> dict[str, Any]:
    """Measure fragmentation and merging after tolerance-aware pixel matching."""

    if tolerance_pixels < 0:
        raise ValueError("tolerance_pixels must be non-negative")
    predicted = np.asarray(prediction, dtype=bool).squeeze()
    expected = np.asarray(target, dtype=bool).squeeze()
    if predicted.shape != expected.shape or predicted.ndim != 2:
        raise ValueError("prediction and target must be aligned two-dimensional masks")
    if valid is not None:
        valid_mask = np.asarray(valid, dtype=bool).squeeze()
        if valid_mask.shape != expected.shape:
            raise ValueError("valid mask must align with prediction and target")
        predicted &= valid_mask
        expected &= valid_mask
    structure = np.ones((3, 3), dtype=np.uint8)
    predicted_labels, predicted_components = ndimage.label(predicted, structure=structure)
    target_labels, target_components = ndimage.label(expected, structure=structure)
    recall = _component_consistency(
        target_labels, predicted, tolerance_pixels=tolerance_pixels
    )
    precision = _component_consistency(
        predicted_labels, expected, tolerance_pixels=tolerance_pixels
    )
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "connectivity_precision": precision,
        "connectivity_recall": recall,
        "connectivity_f1": f1,
        "predicted_components": int(predicted_components),
        "target_components": int(target_components),
        "component_count_error": int(predicted_components - target_components),
        "tolerance_pixels": tolerance_pixels,
    }
