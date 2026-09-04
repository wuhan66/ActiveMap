from pathlib import Path

import numpy as np
import pytest

from activemap.integrations.sam_road import SAMRoadPredictor


def test_sam_road_rgb_conversion_preserves_byte_scale() -> None:
    byte_image = np.full((3, 4, 5), 127.0, dtype=np.float32)
    unit_image = byte_image / 255.0

    assert np.allclose(SAMRoadPredictor._rgb_255(byte_image), byte_image)
    assert np.allclose(SAMRoadPredictor._rgb_255(unit_image), byte_image)


def test_sam_road_rgb_conversion_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="three-band"):
        SAMRoadPredictor._rgb_255(np.zeros((4, 5), dtype=np.float32))
    invalid = np.zeros((3, 4, 5), dtype=np.float32)
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        SAMRoadPredictor._rgb_255(invalid)


def test_sam_road_constructor_validates_assets_before_optional_import(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        SAMRoadPredictor(
            tmp_path / "repo",
            tmp_path / "config.yml",
            tmp_path / "model.ckpt",
            tmp_path / "sam.pth",
        )


def test_sam_road_threshold_prefers_explicit_then_adapter_metadata() -> None:
    payload = {"activemap_adapter": {"road_threshold": 0.6}}

    assert SAMRoadPredictor.resolve_road_threshold(
        payload, config_threshold=0.341, explicit_threshold=None
    ) == pytest.approx(0.6)
    assert SAMRoadPredictor.resolve_road_threshold(
        payload, config_threshold=0.341, explicit_threshold=0.4
    ) == pytest.approx(0.4)
    assert SAMRoadPredictor.resolve_road_threshold(
        {}, config_threshold=0.341, explicit_threshold=None
    ) == pytest.approx(0.341)


def test_sam_road_threshold_rejects_invalid_checkpoint_metadata() -> None:
    with pytest.raises(ValueError, match="resolved road threshold"):
        SAMRoadPredictor.resolve_road_threshold(
            {"activemap_adapter": {"road_threshold": 1.2}},
            config_threshold=0.341,
            explicit_threshold=None,
        )
