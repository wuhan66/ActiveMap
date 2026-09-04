import json
from pathlib import Path

import numpy as np
import pytest

from activemap.integrations.sam_road_training import (
    ThresholdMetrics,
    decode_uncompressed_rle,
    load_road_records,
    road_bce_dice_loss,
    split_support,
    union_segmentations,
)


def _encode_uncompressed(mask: np.ndarray) -> dict[str, object]:
    flattened = mask.astype(np.uint8).reshape(-1, order="F")
    counts = []
    value = 0
    run = 0
    for pixel in flattened:
        if int(pixel) == value:
            run += 1
        else:
            counts.append(run)
            run = 1
            value = int(pixel)
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


def test_uncompressed_coco_rle_round_trip_and_union() -> None:
    first = np.zeros((3, 4), dtype=np.uint8)
    first[0:2, 1] = 1
    second = np.zeros((3, 4), dtype=np.uint8)
    second[2, 3] = 1

    assert np.array_equal(decode_uncompressed_rle(_encode_uncompressed(first)), first)
    assert np.array_equal(
        union_segmentations(
            [_encode_uncompressed(first), _encode_uncompressed(second)],
            height=3,
            width=4,
        ),
        first | second,
    )


def test_uncompressed_rle_rejects_invalid_count_sum() -> None:
    with pytest.raises(ValueError, match="counts sum"):
        decode_uncompressed_rle({"size": [2, 2], "counts": [1, 1]})
    with pytest.raises(ValueError, match="uncompressed"):
        decode_uncompressed_rle({"size": [2, 2], "counts": "encoded"})


def test_load_records_enforces_aoi_protocol(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    from PIL import Image

    for name in ("a.png", "b.png"):
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image_root / name)
    payload = {
        "images": [
            {
                "id": 1,
                "file_name": "a.png",
                "width": 2,
                "height": 2,
                "aoi_id": "atlanta",
                "edit_type": "ADD",
                "activemap_sample_id": "sample-a",
            },
            {
                "id": 2,
                "file_name": "b.png",
                "width": 2,
                "height": 2,
                "aoi_id": "austin",
                "edit_type": "KEEP",
                "activemap_sample_id": "sample-b",
            },
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "segmentation": _encode_uncompressed(np.eye(2, dtype=np.uint8)),
            }
        ],
    }
    annotation_path = tmp_path / "train.json"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    train = load_road_records(annotation_path, image_root, exclude_aois={"atlanta"})
    calibration = load_road_records(
        annotation_path, image_root, include_aois={"atlanta"}
    )

    assert [row.aoi_id for row in train] == ["austin"]
    assert [row.aoi_id for row in calibration] == ["atlanta"]
    assert split_support(train) == {
        "images": 1,
        "aois": {"austin": 1},
        "edit_types": {"KEEP": 1},
        "images_without_annotations": 1,
    }


def test_threshold_metrics_selects_best_f1() -> None:
    metrics = ThresholdMetrics((0.3, 0.7))
    metrics.update(
        np.array([[0.9, 0.6], [0.4, 0.1]], dtype=np.float32),
        np.array([[1, 1], [0, 0]], dtype=np.uint8),
    )

    assert metrics.best()["threshold"] == 0.3
    assert metrics.best()["f1"] == pytest.approx(0.8)
    assert metrics.best()["mean_image_iou"] == pytest.approx(2 / 3)


def test_road_loss_is_finite_and_backpropagates() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros((2, 3, 3), requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, 1, 1] = 1

    loss, components = road_bce_dice_loss(logits, target, positive_weight=4.0)
    loss.backward()

    assert torch.isfinite(loss)
    assert set(components) == {"bce", "dice"}
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())
