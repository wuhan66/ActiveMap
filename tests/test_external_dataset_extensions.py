import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from activemap.data.disaster_map import (
    DisasterEvidence,
    DisasterMapEpisode,
    deterministic_disaster_split,
    match_spacenet8_assets,
    spacenet8_target_counts,
    spacenet8_tile_id,
    validate_disaster_map_jsonl,
    write_disaster_map_jsonl,
    write_spacenet8_prior,
)
from activemap.data.external_datasets import (
    audit_external_dataset_registry,
    build_external_dataset_scaffold,
)
from activemap.data.mapex_kth import (
    OCCUPANCY_UNKNOWN,
    build_mapex_kth_navigation_episodes,
    deterministic_mapex_kth_split,
)
from activemap.data.navigation_map import (
    NavigationEvidence,
    NavigationMapEpisode,
    NavigationPose,
    validate_navigation_map_jsonl,
    write_navigation_map_jsonl,
)
from activemap.data.structured_map import (
    StructuredAtomicEdit,
    StructuredMapObservation,
    StructuredMapSample,
    convert_structured_map_scenes,
    derive_structured_atomic_edits,
    validate_structured_map_jsonl,
    write_structured_map_jsonl,
)
from activemap.integrations.baselines.contracts import (
    ExternalBaselinePrediction,
    ExternalBaselineResult,
    validate_prediction_jsonl,
    write_prediction_jsonl,
)

REGISTRY = Path("configs/data/external_datasets.yaml")


def test_external_dataset_registry_is_valid() -> None:
    report = audit_external_dataset_registry(REGISTRY)
    assert report["valid"] is True
    assert report["summary"] == {
        "datasets": 10,
        "prepared": 2,
        "terms_required": 4,
    }
    assert {row["id"] for row in report["datasets"]} >= {
        "argotweak",
        "tbv",
        "nuscenes",
        "argoverse2",
        "hrdx",
        "mapex_kth",
        "cubicasa5k_pilot",
        "spacenet8",
    }


def _disaster_episode(*, split: str = "train") -> DisasterMapEpisode:
    return DisasterMapEpisode(
        schema_version="activemap-disaster-map-episode-v1",
        episode_id=f"sn8-{split}-1",
        dataset="spacenet8",
        split=split,
        aoi_id="germany",
        prior_map_path="maps/prior.geojson",
        target_map_path="maps/target.geojson",
        evidence=[
            DisasterEvidence(
                evidence_id="post-1",
                modality="post_event_rgb",
                path="images/post-1.tif",
                cost=1.0,
                timestamp=1,
            )
        ],
        target_layers=["building", "road", "flooded_building", "flooded_road"],
        budget=2.0,
        test_assets_read=split == "test",
    )


def test_disaster_map_contract_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "train.jsonl"
    write_disaster_map_jsonl([_disaster_episode()], output)
    assert validate_disaster_map_jsonl(output) == (1, [])


def test_disaster_map_test_split_is_locked_by_default(tmp_path: Path) -> None:
    output = tmp_path / "test.jsonl"
    write_disaster_map_jsonl([_disaster_episode(split="test")], output)
    count, errors = validate_disaster_map_jsonl(output)
    assert count == 1
    assert any("test disaster episode is locked" in error for error in errors)
    assert validate_disaster_map_jsonl(output, allow_test=True) == (1, [])


def test_spacenet8_asset_matching_keeps_multiple_post_observations(tmp_path: Path) -> None:
    for directory in ("annotations", "PRE-event", "POST-event"):
        (tmp_path / directory).mkdir()
    annotation = tmp_path / "annotations" / "0_24_66.geojson"
    _write_feature_collection(annotation, [])
    (tmp_path / "PRE-event" / "pre_scene_0_24_66.tif").write_bytes(b"pre")
    for scene in ("post_a", "post_b"):
        (tmp_path / "POST-event" / f"{scene}_0_24_66.tif").write_bytes(b"post")

    rows = match_spacenet8_assets(tmp_path)
    assert spacenet8_tile_id(annotation) == "0_24_66"
    assert len(rows) == 1
    assert len(rows[0]["post"]) == 2


def test_spacenet8_prior_hides_flood_state(tmp_path: Path) -> None:
    target = tmp_path / "target.geojson"
    prior = tmp_path / "prior.geojson"
    _write_feature_collection(
        target,
        [
            {
                "type": "Feature",
                "properties": {"highway": "primary", "flooded": "yes"},
                "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
            },
            {
                "type": "Feature",
                "properties": {"building": "yes", "flooded": "no"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                },
            },
        ],
    )
    counts = write_spacenet8_prior(target, prior)
    payload = json.loads(prior.read_text(encoding="utf-8"))
    assert counts == {"features": 2, "roads": 1, "buildings": 1, "flooded": 1}
    assert spacenet8_target_counts(target) == counts
    assert all(feature["properties"]["flooded"] is None for feature in payload["features"])
    assert deterministic_disaster_split("0_24_66") == deterministic_disaster_split("0_24_66")


def _navigation_episode(*, split: str = "train") -> NavigationMapEpisode:
    return NavigationMapEpisode(
        schema_version="activemap-navigation-map-episode-v1",
        episode_id=f"mapex-{split}-1",
        dataset="mapex_kth",
        domain="indoor",
        split=split,
        map_representation="occupancy_grid",
        initial_map_path="maps/initial.png",
        target_map_path="maps/target.png",
        start_pose=NavigationPose(x=1.0, y=2.0, yaw=0.0),
        evidence=[
            NavigationEvidence(
                evidence_id="crop-1",
                modality="occupancy_crop",
                path="evidence/crop-1.png",
                cost=1.0,
                timestamp=0,
            )
        ],
        budget=3.0,
        test_assets_read=split == "test",
    )


def test_navigation_map_contract_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "train.jsonl"
    write_navigation_map_jsonl([_navigation_episode()], output)
    assert validate_navigation_map_jsonl(output) == (1, [])


def test_navigation_map_test_split_is_locked_by_default(tmp_path: Path) -> None:
    output = tmp_path / "test.jsonl"
    write_navigation_map_jsonl([_navigation_episode(split="test")], output)
    count, errors = validate_navigation_map_jsonl(output)
    assert count == 1
    assert any("test navigation episode is locked" in error for error in errors)
    assert validate_navigation_map_jsonl(output, allow_test=True) == (1, [])


def test_mapex_kth_adapter_builds_local_observation_development_episodes(tmp_path: Path) -> None:
    source = tmp_path / "kth_test_maps"
    for map_id in ("room-a", "room-b"):
        map_dir = source / map_id
        map_dir.mkdir(parents=True)
        raw = np.full((48, 64), 254, dtype=np.uint8)
        raw[:3, :] = 0
        raw[:, :3] = 0
        valid = np.ones((48, 64), dtype=np.uint8)
        valid[:3, :] = 0
        np.save(map_dir / "occ_map.npy", raw)
        np.save(map_dir / "valid_space.npy", valid)

    output = tmp_path / "derived"
    summary = build_mapex_kth_navigation_episodes(
        source,
        output,
        candidate_count=2,
        observation_radius=4,
        seed=11,
    )
    assert summary["map_count"] == 2
    assert summary["test_assets_read"] is False
    assert summary["split_counts"]["test"] == 0
    manifests = list((output / "manifests").glob("*.jsonl"))
    assert manifests
    rows = [json.loads(line) for path in manifests for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["metadata"]["development_only"] is True for row in rows)
    assert all(len(row["evidence"]) == 2 for row in rows)
    assert all(row["evidence"][0]["pose"] is not None for row in rows)
    assert all(row["evidence"][0]["footprint_radius_pixels"] == 4 for row in rows)
    initial = np.load(rows[0]["initial_map_path"])
    target = np.load(rows[0]["target_map_path"])
    assert np.any(initial == OCCUPANCY_UNKNOWN)
    assert np.any(initial != OCCUPANCY_UNKNOWN)
    assert set(np.unique(target)) == {0.0, 1.0}


def test_mapex_kth_adapter_records_reproducible_synthetic_sensor_error(tmp_path: Path) -> None:
    source = tmp_path / "kth_test_maps"
    map_dir = source / "room-a"
    map_dir.mkdir(parents=True)
    raw = np.full((48, 64), 254, dtype=np.uint8)
    raw[:3, :] = 0
    valid = np.ones((48, 64), dtype=np.uint8)
    valid[:3, :] = 0
    np.save(map_dir / "occ_map.npy", raw)
    np.save(map_dir / "valid_space.npy", valid)

    with pytest.raises(ValueError, match="sensor_error_rate"):
        build_mapex_kth_navigation_episodes(source, tmp_path / "invalid", sensor_error_rate=1.0)

    output = tmp_path / "noisy"
    summary = build_mapex_kth_navigation_episodes(
        source,
        output,
        candidate_count=2,
        observation_radius=4,
        sensor_error_rate=0.5,
        seed=11,
    )
    assert summary["observation_model"] == "square_local_noisy_reveal_v1"
    row = json.loads(next((output / "manifests").glob("*.jsonl")).read_text().splitlines()[0])
    assert row["metadata"]["candidate_sensor_error_rate"] == 0.5


def test_mapex_kth_split_is_stable_and_validates_fraction() -> None:
    assert deterministic_mapex_kth_split("room-a") == deterministic_mapex_kth_split("room-a")
    with pytest.raises(ValueError, match="val_fraction"):
        deterministic_mapex_kth_split("room-a", 1.0)


def test_dataset_scaffold_records_license_boundary(tmp_path: Path) -> None:
    plan = build_external_dataset_scaffold(REGISTRY, "argotweak", tmp_path)
    assert plan["automatic_download_permitted"] is False
    assert plan["license_acceptance_required"] is True
    assert plan["test_labels_locked"] is True
    root = tmp_path / "argotweak"
    assert (root / "raw").is_dir()
    assert (root / "manifests").is_dir()
    assert json.loads((root / "dataset_plan.json").read_text())["dataset"]["id"] == "argotweak"
    with pytest.raises(FileExistsError):
        build_external_dataset_scaffold(REGISTRY, "argotweak", tmp_path)


def test_prediction_contract_round_trip(tmp_path: Path) -> None:
    prediction = ExternalBaselinePrediction(
        schema_version="activemap-external-baseline-prediction-v1",
        sample_id="sample-1",
        split="validation",
        dataset="argotweak",
        baseline="rtmap",
        operation="ADD",
        confidence=0.8,
        geometry={"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]},
        source_artifact="predictions.pkl",
        test_assets_read=False,
    )
    output = tmp_path / "predictions.jsonl"
    write_prediction_jsonl([prediction], output)
    assert validate_prediction_jsonl(output) == (1, [])


def test_prediction_contract_rejects_invalid_edit_payload() -> None:
    with pytest.raises(ValidationError, match="ADD requires geometry"):
        ExternalBaselinePrediction(
            schema_version="activemap-external-baseline-prediction-v1",
            sample_id="sample-1",
            split="validation",
            dataset="argotweak",
            baseline="rtmap",
            operation="ADD",
            source_artifact="predictions.pkl",
        )


def test_result_contract_enforces_test_provenance() -> None:
    with pytest.raises(ValidationError, match="test_assets_read"):
        ExternalBaselineResult(
            schema_version="activemap-external-baseline-result-v1",
            run_id="run-1",
            dataset="argotweak",
            baseline="rtmap",
            split="validation",
            seed=1,
            source_commit="deadbeef",
            predictions_path="predictions.jsonl",
            sample_count=1,
            metrics={"atomic_edit_f1": 0.5},
            test_assets_read=True,
        )


def _structured_sample(*, split: str = "train") -> StructuredMapSample:
    return StructuredMapSample(
        schema_version="activemap-structured-map-sample-v1",
        sample_id=f"argotweak-{split}-1",
        dataset="argotweak",
        split=split,
        aoi_id="log-1",
        prior_map_path="maps/prior.json",
        target_map_path="maps/target.json",
        observations=[
            StructuredMapObservation(
                observation_id="camera-front-1",
                timestamp="2025-01-01T00:00:00Z",
                modality="camera",
                path="images/front.jpg",
                cost=1.0,
            )
        ],
        atomic_edits=[
            StructuredAtomicEdit(
                operation="ADD",
                object_id="lane-1",
                target_geometry={
                    "type": "LineString",
                    "coordinates": [[0.0, 0.0], [1.0, 1.0]],
                },
            )
        ],
        native_sample_id="native-1",
        test_assets_read=split == "test",
    )


def test_structured_map_contract_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "train.jsonl"
    write_structured_map_jsonl([_structured_sample()], output)
    assert validate_structured_map_jsonl(output) == (1, [])


def test_structured_map_test_split_is_locked_by_default(tmp_path: Path) -> None:
    output = tmp_path / "test.jsonl"
    write_structured_map_jsonl([_structured_sample(split="test")], output)
    count, errors = validate_structured_map_jsonl(output)
    assert count == 1
    assert any("test sample is locked" in error for error in errors)
    assert validate_structured_map_jsonl(output, allow_test=True) == (1, [])


def test_structured_atomic_edit_rejects_invalid_add_transition() -> None:
    with pytest.raises(ValidationError, match="ADD requires only target_geometry"):
        StructuredAtomicEdit(
            operation="ADD",
            object_id="lane-1",
            prior_geometry={
                "type": "LineString",
                "coordinates": [[0.0, 0.0], [1.0, 1.0]],
            },
            target_geometry={
                "type": "LineString",
                "coordinates": [[0.0, 0.0], [2.0, 2.0]],
            },
        )


def _write_feature_collection(path: Path, features: list[dict]) -> None:
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )


def _line_feature(object_id: str, end: float, *, lane_type: str = "vehicle") -> dict:
    return {
        "type": "Feature",
        "id": object_id,
        "properties": {"lane_type": lane_type},
        "geometry": {
            "type": "LineString",
            "coordinates": [[0.0, 0.0], [end, end]],
        },
    }


def test_derive_structured_atomic_edits_from_geojson(tmp_path: Path) -> None:
    prior = tmp_path / "prior.geojson"
    target = tmp_path / "target.geojson"
    _write_feature_collection(
        prior,
        [_line_feature("keep", 1.0), _line_feature("reshape", 1.0), _line_feature("delete", 1.0)],
    )
    _write_feature_collection(
        target,
        [_line_feature("keep", 1.0), _line_feature("reshape", 2.0), _line_feature("add", 1.0)],
    )
    edits, counts = derive_structured_atomic_edits(prior, target)
    assert {(edit.object_id, edit.operation.value) for edit in edits} == {
        ("add", "ADD"),
        ("delete", "DELETE"),
        ("reshape", "RESHAPE"),
    }
    assert counts == {"KEEP": 1, "ADD": 1, "DELETE": 1, "RESHAPE": 1}


def test_convert_structured_map_scene_and_lock_test(tmp_path: Path) -> None:
    prior = tmp_path / "prior.geojson"
    target = tmp_path / "target.geojson"
    image = tmp_path / "front.jpg"
    image.write_bytes(b"fixture")
    _write_feature_collection(prior, [_line_feature("keep", 1.0)])
    _write_feature_collection(target, [_line_feature("keep", 1.0)])
    manifest = tmp_path / "scenes.jsonl"
    scene = {
        "schema_version": "activemap-structured-map-scene-v1",
        "sample_id": "argotweak-val-1",
        "dataset": "argotweak",
        "split": "val",
        "aoi_id": "log-1",
        "native_sample_id": "native-1",
        "prior_map_path": prior.name,
        "target_map_path": target.name,
        "observations": [
            {
                "observation_id": "front-1",
                "timestamp": "1",
                "modality": "camera",
                "path": image.name,
                "cost": 1.0,
            }
        ],
    }
    manifest.write_text(json.dumps(scene) + "\n", encoding="utf-8")
    output = tmp_path / "structured.jsonl"
    summary = convert_structured_map_scenes(manifest, output)
    assert summary["sample_count"] == 1
    sample = StructuredMapSample.model_validate_json(output.read_text(encoding="utf-8"))
    assert [(edit.object_id, edit.operation.value) for edit in sample.atomic_edits] == [
        ("keep", "KEEP")
    ]
    assert Path(sample.observations[0].path).is_absolute()

    locked = dict(scene, sample_id="argotweak-test-1", split="test")
    locked_manifest = tmp_path / "test_scenes.jsonl"
    locked_manifest.write_text(json.dumps(locked) + "\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="allow_test"):
        convert_structured_map_scenes(locked_manifest, tmp_path / "test.jsonl")
