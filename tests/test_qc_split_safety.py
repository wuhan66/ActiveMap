import json
from pathlib import Path

import numpy as np

from activemap.data.qc import render_updater_qc
from activemap.models import EditOperation
from activemap.updater_records import UpdaterSample
from scripts.approve_dataset_qc import QC_DIRECTORY_NAME, approve_qc


def _write_sample(root: Path, split: str) -> UpdaterSample:
    image = root / f"{split}-image.npy"
    prior = root / f"{split}-prior.npy"
    target = root / f"{split}-target.npy"
    np.save(image, np.zeros((3, 8, 8), dtype=np.uint8))
    np.save(prior, np.zeros((8, 8), dtype=np.uint8))
    np.save(target, np.ones((8, 8), dtype=np.uint8))
    return UpdaterSample(
        sample_id=f"sample-{split}",
        split=split,
        image_path=str(image),
        prior_mask_path=str(prior),
        target_mask_path=str(target),
        edit_type=EditOperation.ADD,
        geometry_delta=[0.0] * 8,
    )


def test_qc_split_filter_never_renders_test(tmp_path: Path) -> None:
    manifest = tmp_path / "samples.jsonl"
    samples = [_write_sample(tmp_path, split) for split in ("train", "val", "test")]
    manifest.write_text(
        "\n".join(sample.model_dump_json() for sample in samples) + "\n",
        encoding="utf-8",
    )
    summary = render_updater_qc(
        manifest, tmp_path / "qc", count=3, splits={"train", "val"}
    )
    assert summary["rendered"] == 2
    assert summary["rendered_split_counts"] == {"train": 1, "val": 1}
    assert summary["test_assets_rendered"] is False
    assert not (tmp_path / "qc/sample-test.png").exists()


def test_qc_approval_rejects_test_rendering(tmp_path: Path) -> None:
    roots = {
        "sn7": tmp_path / "processed/sn7_v1/updater_v4_cap20",
        "muno21": tmp_path / "processed/muno21_v2/updater",
        "inria": tmp_path / "processed/inria_v1/segmentation",
    }
    for root in roots.values():
        qc = root / QC_DIRECTORY_NAME
        qc.mkdir(parents=True)
        (root / "audit.json").write_text('{"passed": true}\n', encoding="utf-8")
        (qc / "index.json").write_text(
            json.dumps(
                {
                    "test_assets_rendered": True,
                    "rendered_split_counts": {"test": 1},
                }
            ),
            encoding="utf-8",
        )
        (qc / "example.png").write_bytes(b"not-an-image")
    try:
        approve_qc(
            tmp_path,
            tmp_path / "approval.json",
            reviewer="reviewer",
            notes="reviewed",
            minimum_counts={"sn7": 1, "muno21": 1, "inria": 1},
        )
    except ValueError as exc:
        assert "test_assets_rendered=false" in str(exc)
    else:
        raise AssertionError("approval unexpectedly accepted test QC")
