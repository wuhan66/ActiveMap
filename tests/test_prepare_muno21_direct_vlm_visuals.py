import numpy as np
import pytest
from types import SimpleNamespace

from scripts.prepare_muno21_direct_vlm_visuals import (
    _assert_no_leak,
    _composite,
    _one_per_operation,
    _visuals,
)


def test_visuals_preserve_shapes_and_mark_prior() -> None:
    image = np.zeros((3, 4, 4), dtype=np.float32)
    prior = np.zeros((4, 4), dtype=np.float32)
    prior[1, 2] = 1.0

    rgb, mask, overlay = _visuals(image, prior)

    assert rgb.size == mask.size == overlay.size == (4, 4)
    assert np.asarray(mask)[1, 2] == 255
    assert tuple(np.asarray(overlay)[1, 2]) == (191, 147, 0)
    assert tuple(np.asarray(overlay)[0, 0]) == (0, 0, 0)


def test_no_leak_audit_rejects_target_fields() -> None:
    _assert_no_leak({"example_id": "x", "observation": {"aoi_id": "a"}})
    with pytest.raises(ValueError, match="forbidden target keys"):
        _assert_no_leak({"example_id": "x", "target_geometry": {}})


def test_composite_has_stable_three_panel_geometry() -> None:
    image = np.zeros((3, 4, 4), dtype=np.float32)
    prior = np.zeros((4, 4), dtype=np.float32)
    rgb, mask, overlay = _visuals(image, prior)
    composite = _composite(rgb, mask, overlay, panel_size=8)
    assert composite.mode == "RGB"
    assert composite.size == (24, 8)


def test_one_per_operation_is_balanced_and_deterministic() -> None:
    def episode(name: str, operation: str) -> SimpleNamespace:
        return SimpleNamespace(
            name=name,
            gt_edit=SimpleNamespace(op=SimpleNamespace(value=operation)),
        )

    selected = _one_per_operation(
        [
            episode("add-first", "ADD"),
            episode("keep", "KEEP"),
            episode("delete", "DELETE"),
            episode("reshape", "RESHAPE"),
            episode("add-second", "ADD"),
        ]
    )
    assert [item.gt_edit.op.value for item in selected] == [
        "KEEP",
        "ADD",
        "DELETE",
        "RESHAPE",
    ]
    assert selected[1].name == "add-first"
