from __future__ import annotations

from pathlib import Path

import numpy as np

from activemap.models import EditOperation
from activemap.training.visualization import _select_visualization_samples, _zoom_box
from activemap.updater_records import UpdaterSample


def _sample(index: int, operation: EditOperation) -> UpdaterSample:
    return UpdaterSample(
        sample_id=f"{operation.value.lower()}-{index:03d}",
        split="val",
        image_path=str(Path("image.png")),
        prior_mask_path=str(Path("prior.png")),
        target_mask_path=str(Path("target.png")),
        edit_type=operation,
        geometry_delta=[0.0] * 8,
    )


def test_visualization_selection_is_balanced_and_deterministic() -> None:
    samples = [
        _sample(index, operation)
        for operation in EditOperation
        for index in range(5)
    ]
    selected = _select_visualization_samples(list(reversed(samples)), 8)
    assert selected == _select_visualization_samples(samples, 8)
    counts = {
        operation: sum(item.edit_type == operation for item in selected)
        for operation in EditOperation
    }
    assert counts == {
        operation: 2 for operation in EditOperation
    }


def test_zoom_box_uses_prior_or_target_and_keeps_context() -> None:
    prior = np.zeros((128, 128), dtype=np.float32)
    target = np.zeros_like(prior)
    prediction = np.zeros_like(prior)
    prior[60:64, 70:74] = 1.0
    target[58:66, 68:76] = 1.0
    prediction[0, 0] = 1.0
    left, top, right, bottom = _zoom_box(prior, target, prediction)
    assert right - left == 48
    assert bottom - top == 48
    assert left <= 68 and right >= 76
    assert top <= 58 and bottom >= 66
