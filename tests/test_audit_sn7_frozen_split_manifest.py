import json
from pathlib import Path

import pandas as pd

from scripts.audit_sn7_frozen_split_manifest import audit


def _episodes(path: Path, split: str, aoi: str):
    path.write_text(
        json.dumps({"aoi_id": aoi, "split": split}) + "\n",
        encoding="utf-8",
    )


def test_materializes_path_only_test_manifest_without_overlap(tmp_path: Path):
    manifest = tmp_path / "split.parquet"
    pd.DataFrame(
        [
            {"aoi_id": "train-aoi", "split": "train", "label_path": "train.json"},
            {"aoi_id": "val-aoi", "split": "val", "label_path": "val.json"},
            {"aoi_id": "test-aoi", "split": "test", "label_path": "test.json"},
        ]
    ).to_parquet(manifest, index=False)
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _episodes(train, "train", "train-aoi")
    _episodes(val, "val", "val-aoi")
    result = audit(manifest, train, val, tmp_path / "output")
    assert result["passed"] is True
    assert result["test_overlap_with_existing_train_val"] == 0
    assert result["label_geometries_read"] is False
