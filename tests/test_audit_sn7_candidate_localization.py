import numpy as np
import pytest

from scripts.audit_sn7_candidate_localization import (
    _fit_geometry_classifier,
    _predict_geometry,
    mask_features,
)


def test_mask_features_capture_empty_and_centered_geometry() -> None:
    valid = np.ones((8, 8), dtype=bool)
    empty = mask_features(np.zeros((8, 8), dtype=bool), valid)
    centered_mask = np.zeros((8, 8), dtype=bool)
    centered_mask[3:5, 3:5] = True
    centered = mask_features(centered_mask, valid)
    assert empty.tolist() == [0.0] * len(empty)
    assert centered[0] == pytest.approx(4 / 64)
    assert centered[1] == pytest.approx(0.5)
    assert centered[2] == pytest.approx(0.5)


def test_nearest_centroid_geometry_classifier() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [1.0, 1.0],
            [1.1, 1.1],
            [2.0, 2.0],
            [2.1, 2.1],
            [3.0, 3.0],
            [3.1, 3.1],
        ]
    )
    labels = [edit for edit in ("KEEP", "ADD", "DELETE", "RESHAPE") for _ in range(2)]
    mean, scale, centroids = _fit_geometry_classifier(features, labels)
    assert _predict_geometry(
        np.asarray([2.05, 2.05]), mean, scale, centroids
    ) == "DELETE"
