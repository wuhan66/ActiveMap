from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from activemap.models import EditOperation
from activemap.updater_records import UpdaterSample


def _load_auditor():
    script_path = Path(__file__).parents[1] / "scripts" / "audit_temporal_supervision_support.py"
    spec = importlib.util.spec_from_file_location("temporal_support_auditor", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample(tmp_path: Path, *, split: str, operation: EditOperation) -> UpdaterSample:
    prior = np.zeros((1, 8, 8), dtype=np.float32)
    target = prior.copy()
    if operation == EditOperation.ADD:
        target[:, 2:5, 2:5] = 1.0
    elif operation == EditOperation.DELETE:
        prior[:, 2:5, 2:5] = 1.0
    elif operation == EditOperation.RESHAPE:
        prior[:, 1:4, 1:4] = 1.0
        target[:, 4:7, 4:7] = 1.0
    stem = f"{split}-{operation.value.lower()}"
    np.save(tmp_path / f"{stem}-prior.npy", prior)
    np.save(tmp_path / f"{stem}-target.npy", target)
    return UpdaterSample(
        sample_id=stem,
        aoi_id="aoi-1",
        split=split,
        image_path=str(tmp_path / f"{stem}-image.npy"),
        prior_mask_path=str(tmp_path / f"{stem}-prior.npy"),
        target_mask_path=str(tmp_path / f"{stem}-target.npy"),
        edit_type=operation,
        geometry_delta=[0.0] * 8,
        supervision_type="real_temporal",
    )


def test_temporal_support_audit_records_real_temporal_change_pixels(tmp_path: Path) -> None:
    samples = [
        _sample(tmp_path, split="train", operation=EditOperation.ADD),
        _sample(tmp_path, split="val", operation=EditOperation.DELETE),
        _sample(tmp_path, split="val", operation=EditOperation.RESHAPE),
    ]
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(
        "".join(sample.model_dump_json() + "\n" for sample in samples), encoding="utf-8"
    )
    output_path = tmp_path / "report.json"

    report = _load_auditor().audit_temporal_supervision_support(
        samples_path,
        output_path,
        allowed_splits={"train", "val"},
    )

    assert report["status"] == "passed"
    assert report["test_assets_read"] is False
    rows = {(row["split"], row["operation"]): row for row in report["rows"]}
    assert rows[("train", "ADD")]["added_positive_samples"] == 1
    assert rows[("val", "DELETE")]["removed_positive_samples"] == 1
    assert rows[("val", "RESHAPE")]["both_change_positive_samples"] == 1
    assert json.loads(output_path.read_text(encoding="utf-8"))["sample_count"] == 3


def test_temporal_support_audit_refuses_test_split(tmp_path: Path) -> None:
    sample = _sample(tmp_path, split="test", operation=EditOperation.ADD)
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(sample.model_dump_json() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="disallowed splits"):
        _load_auditor().audit_temporal_supervision_support(
            samples_path,
            tmp_path / "report.json",
            allowed_splits={"train", "val"},
        )
