from pathlib import Path

import numpy as np
from shapely.geometry import box, mapping

from activemap.data.updater_audit import audit_updater_dataset
from activemap.models import EditOperation, GeoJSONGeometry
from activemap.synthetic_updater import generate_updater_smoke_dataset
from activemap.updater_records import UpdaterSample


def test_synthetic_updater_dataset_passes_structural_audit(tmp_path: Path) -> None:
    manifest = generate_updater_smoke_dataset(
        tmp_path / "data", sample_count=40, image_size=16, seed=8
    )
    summary = audit_updater_dataset(manifest)
    assert summary["passed"]
    assert summary["sample_count"] == 40
    assert summary["aoi_count"] == 10
    assert summary["empty_contract_violations"] == 0
    assert summary["low_valid_count"] == 0


def test_audit_rejects_nonlocal_reshape(tmp_path: Path) -> None:
    image = np.ones((3, 16, 16), dtype=np.float32)
    prior = np.zeros((16, 16), dtype=np.float32)
    target = np.zeros((16, 16), dtype=np.float32)
    prior[4:8, 2:6] = 1.0
    target[4:8, 10:14] = 1.0
    np.save(tmp_path / "image.npy", image)
    np.save(tmp_path / "prior.npy", prior)
    np.save(tmp_path / "target.npy", target)
    sample = UpdaterSample(
        sample_id="far-reshape",
        aoi_id="aoi-1",
        split="train",
        image_path="image.npy",
        prior_mask_path="prior.npy",
        target_mask_path="target.npy",
        edit_type=EditOperation.RESHAPE,
        geometry_delta=[0.0] * 8,
        prior_geometry=GeoJSONGeometry.model_validate(mapping(box(0, 0, 4, 4))),
        target_geometry=GeoJSONGeometry.model_validate(mapping(box(100, 0, 104, 4))),
    )
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(sample.model_dump_json() + "\n", encoding="utf-8")
    summary = audit_updater_dataset(samples_path, max_reshape_centroid_distance=20.0)
    assert not summary["passed"]
    assert summary["reshape_distance_violations"] == 1

    sample.geometry_family = "polyline"
    samples_path.write_text(sample.model_dump_json() + "\n", encoding="utf-8")
    allowed = audit_updater_dataset(
        samples_path,
        max_reshape_centroid_distance=20.0,
        allow_nonlocal_polyline_reshape=True,
    )
    assert allowed["passed"]
    assert allowed["reshape_distance_violations"] == 0
    assert allowed["nonlocal_polyline_reshape_count"] == 1


def test_audit_allows_paired_empty_keep_only_when_explicit(tmp_path: Path) -> None:
    np.save(tmp_path / "image.npy", np.ones((3, 16, 16), dtype=np.float32))
    np.save(tmp_path / "empty.npy", np.zeros((16, 16), dtype=np.float32))
    sample = UpdaterSample(
        sample_id="empty-keep",
        aoi_id="aoi-1",
        split="train",
        image_path="image.npy",
        prior_mask_path="empty.npy",
        target_mask_path="empty.npy",
        edit_type=EditOperation.KEEP,
        geometry_delta=[0.0] * 8,
    )
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(sample.model_dump_json() + "\n", encoding="utf-8")

    assert not audit_updater_dataset(samples_path)["passed"]
    allowed = audit_updater_dataset(samples_path, allow_empty_keep=True)
    assert allowed["passed"]
    assert allowed["allow_empty_keep"] is True
    assert allowed["empty_keep_count"] == 1


def test_full_scene_add_requires_added_pixels_not_empty_prior(tmp_path: Path) -> None:
    image = np.ones((3, 16, 16), dtype=np.float32)
    prior = np.zeros((16, 16), dtype=np.float32)
    prior[4:8, 2:6] = 1.0
    target = prior.copy()
    target[4:8, 10:14] = 1.0
    np.save(tmp_path / "image.npy", image)
    np.save(tmp_path / "prior.npy", prior)
    np.save(tmp_path / "target.npy", target)
    sample = UpdaterSample(
        sample_id="full-scene-add",
        aoi_id="aoi-1",
        split="train",
        image_path="image.npy",
        prior_mask_path="prior.npy",
        target_mask_path="target.npy",
        edit_type=EditOperation.ADD,
        geometry_delta=[0.0] * 8,
        supervision_type="full_scene_temporal",
    )
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(sample.model_dump_json() + "\n", encoding="utf-8")

    summary = audit_updater_dataset(samples_path)
    assert summary["passed"]
    assert summary["empty_contract_violations"] == 0
    assert summary["temporal_contract_violations"] == 0

    np.save(tmp_path / "target.npy", prior)
    missing = audit_updater_dataset(samples_path)
    assert not missing["passed"]
    assert missing["temporal_contract_violations"] == 1
