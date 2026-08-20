"""Safety calibration helpers for terminal editable-map decisions."""

from __future__ import annotations

from typing import Any

import numpy as np


def apply_commit_threshold(
    probabilities: np.ndarray, *, keep_index: int, threshold: float
) -> np.ndarray:
    """Suppress update predictions whose winning update probability is too low."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("probabilities must be a two-dimensional class matrix")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if keep_index < 0 or keep_index >= values.shape[1]:
        raise ValueError("keep_index is out of range")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be finite")

    prediction = np.argmax(values, axis=1).astype(np.int64)
    update_values = values.copy()
    update_values[:, keep_index] = -np.inf
    update_confidence = np.max(update_values, axis=1)
    suppress = (prediction != keep_index) & (update_confidence < threshold)
    prediction[suppress] = keep_index
    return prediction


def apply_keep_preserving_guard(
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    *,
    keep_index: int,
) -> np.ndarray:
    """Prevent a candidate from changing baseline KEEP decisions into edits."""
    baseline = np.asarray(baseline_prediction, dtype=np.int64)
    candidate = np.asarray(candidate_prediction, dtype=np.int64)
    if baseline.ndim != 1 or candidate.ndim != 1 or baseline.shape != candidate.shape:
        raise ValueError("baseline and candidate predictions must be matching vectors")
    guarded = candidate.copy()
    guarded[baseline == keep_index] = keep_index
    return guarded


def select_safety_candidate(
    candidates: list[dict[str, Any]], *, false_edit_cap: float
) -> dict[str, Any]:
    """Select the best train-only candidate satisfying a false-edit cap."""
    if not 0.0 <= false_edit_cap <= 1.0:
        raise ValueError("false_edit_cap must be in [0, 1]")
    eligible = [
        row for row in candidates if float(row["metrics"]["false_edit_rate"]) <= false_edit_cap
    ]
    if not eligible:
        raise ValueError("no candidate satisfies the train false-edit cap")
    return max(
        eligible,
        key=lambda row: (
            float(row["metrics"]["macro_f1"]),
            -float(row["metrics"]["missed_edit_rate"]),
            -float(row["metrics"]["false_edit_rate"]),
            -float(row["commit_threshold"]),
            -float(row["C"]),
        ),
    )
