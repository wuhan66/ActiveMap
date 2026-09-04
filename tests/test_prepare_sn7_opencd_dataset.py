import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.prepare_sn7_opencd_dataset import export_dataset


def _write_sample(root: Path, name: str, split: str) -> dict:
    image = np.full((3, 4, 4), 0.5, dtype=np.float32)
    prior = np.zeros((4, 4), dtype=np.float32)
    target = prior.copy()
    target[1:3, 1:3] = 1.0
    valid = np.ones((4, 4), dtype=np.float32)
    valid[0, 0] = 0.0
    paths = {}
    for key, value in {
        "image": image,
        "prior": prior,
        "target": target,
        "valid": valid,
    }.items():
        path = root / f"{name}_{key}.npy"
        np.save(path, value)
        paths[key] = path.name
    return {
        "sample_id": name,
        "aoi_id": f"aoi-{split}",
        "split": split,
        "edit_type": "ADD",
        "image_path": paths["image"],
        "prior_mask_path": paths["prior"],
        "target_mask_path": paths["target"],
        "valid_mask_path": paths["valid"],
    }


def test_export_preserves_xor_and_ignore_labels(tmp_path: Path) -> None:
    rows = [
        _write_sample(tmp_path, "train-sample", "train"),
        _write_sample(tmp_path, "val-sample", "val"),
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "opencd"
    summary = export_dataset(manifest, output)
    assert summary["sample_count"] == 2
    assert summary["test_assets_read"] is False
    identity = [
        json.loads(line)
        for line in (output / "samples.jsonl").read_text().splitlines()
    ]
    train_stem = identity[0]["stem"]
    label = np.asarray(Image.open(output / "train" / "label" / f"{train_stem}.png"))
    assert set(np.unique(label)) == {0, 1, 255}
    assert label[0, 0] == 255
    assert label[1, 1] == 1
    prior = np.asarray(Image.open(output / "train" / "A" / f"{train_stem}.png"))
    image = np.asarray(Image.open(output / "train" / "B" / f"{train_stem}.png"))
    assert prior.shape == image.shape == (4, 4, 3)
    assert np.all(prior == 0)
    assert np.all(image == 128)


def test_export_rejects_test_split(tmp_path: Path) -> None:
    row = _write_sample(tmp_path, "test-sample", "test")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden splits"):
        export_dataset(manifest, tmp_path / "output")
