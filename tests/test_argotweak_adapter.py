from __future__ import annotations

import json
from pathlib import Path

from activemap.data.argotweak import (
    CAMERAS,
    argotweak_annotation_to_geojson,
    build_argotweak_segment_episode,
    build_argotweak_segment_scenes,
    build_argotweak_tbv_subset_manifest,
    convert_argotweak_annotation,
)
from activemap.data.structured_map import derive_structured_atomic_edits


def _lane(lane_id: str, offset: float = 0.0) -> dict:
    numeric = int(lane_id.split("-")[-1])
    return {
        "id": lane_id,
        "is_intersection": False,
        "lane_type": "VEHICLE",
        "left_lane_boundary": [{"x": 0, "y": 1 + offset}, {"x": 2, "y": 1 + offset}],
        "right_lane_boundary": [{"x": 0, "y": offset}, {"x": 2, "y": offset}],
        "left_lane_mark_type": "SOLID_WHITE",
        "right_lane_mark_type": "DASHED_WHITE",
        "predecessors": [numeric - 1],
        "successors": [numeric + 1],
    }


def _annotation(path: Path) -> None:
    payload = {
        "laneSegments": {
            "ls-10": {"old": _lane("ls-10"), "new": None, "changes": [0]},
            "ls-20": {"old": _lane("ls-20"), "new": _lane("ls-20", 0.5), "changes": [3]},
            "ls-30": {"old": _lane("ls-30"), "new": None, "changes": [2]},
            "ls-40": {"old": None, "new": _lane("ls-40"), "changes": [1]},
        },
        "pedCrossings": {},
        "drivableAreas": {},
        "filename": "segment.json",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_argotweak_old_is_target_and_new_is_stale_prior(tmp_path: Path) -> None:
    annotation = tmp_path / "segment.json"
    _annotation(annotation)
    prior, target, counts = argotweak_annotation_to_geojson(annotation)

    assert {row["id"] for row in prior["features"]} == {"ls-10", "ls-20", "ls-40"}
    assert {row["id"] for row in target["features"]} == {"ls-10", "ls-20", "ls-30"}
    assert counts == {"KEEP": 1, "ADD": 1, "DELETE": 1, "RESHAPE": 1}
    lane = next(row for row in target["features"] if row["id"] == "ls-20")
    assert lane["properties"]["predecessors"] == "ls-19"
    assert lane["geometry"]["coordinates"][0][0] == lane["geometry"]["coordinates"][0][-1]


def test_argotweak_conversion_derives_atomic_edits(tmp_path: Path) -> None:
    annotation = tmp_path / "segment.json"
    _annotation(annotation)
    summary = convert_argotweak_annotation(annotation, tmp_path / "maps")
    edits, counts = derive_structured_atomic_edits(
        Path(summary["prior"]), Path(summary["target"]), include_keep=True
    )

    assert counts == {"KEEP": 1, "ADD": 1, "DELETE": 1, "RESHAPE": 1}
    assert {(row.object_id, row.operation.value) for row in edits} == {
        ("ls-10", "KEEP"),
        ("ls-20", "RESHAPE"),
        ("ls-30", "ADD"),
        ("ls-40", "DELETE"),
    }


def test_build_tbv_manifest_locks_test_and_lists_seven_cameras(tmp_path: Path) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"train": {"0": "train-log"}, "val": {"1": "val-log"}, "test": {"2": "test-log"}}),
        encoding="utf-8",
    )
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for name in ("train-log", "val-log"):
        (annotations / f"{name}.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "tbv_dependencies.jsonl"

    summary = build_argotweak_tbv_subset_manifest(splits, annotations, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert summary["split_counts"] == {"train": 1, "val": 1}
    assert summary["test_assets_read"] is False
    assert len(rows) == 2
    assert len(rows[0]["camera_globs"]) == len(CAMERAS) == 7
    assert all(row["split"] != "test" for row in rows)


def test_build_segment_scenes_synchronizes_seven_cameras(tmp_path: Path) -> None:
    annotation = tmp_path / "annotation.json"
    _annotation(annotation)
    maps = convert_argotweak_annotation(annotation, tmp_path / "maps")
    segment = tmp_path / "segment-1"
    (segment / "calibration").mkdir(parents=True)
    for relative in (
        "city_SE3_egovehicle.feather",
        "calibration/egovehicle_SE3_sensor.feather",
        "calibration/intrinsics.feather",
    ):
        (segment / relative).write_bytes(b"fixture")
    for camera in CAMERAS:
        root = segment / "sensors" / "cameras" / camera
        root.mkdir(parents=True)
        for index in range(21):
            (root / f"{1000 + index}.jpg").write_bytes(b"image")
    output = tmp_path / "scenes.jsonl"

    summary = build_argotweak_segment_scenes(
        segment,
        Path(maps["prior"]),
        Path(maps["target"]),
        output,
        split="val",
        stride=10,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert summary["scene_count"] == 2
    assert summary["observations_per_scene"] == 7
    assert len(rows) == 2
    assert {row["split"] for row in rows} == {"val"}
    assert all(len(row["observations"]) == 7 for row in rows)
    assert summary["test_assets_read"] is False


def test_build_segment_episode_uses_camera_bundles(tmp_path: Path) -> None:
    annotation = tmp_path / "annotation.json"
    _annotation(annotation)
    maps = convert_argotweak_annotation(annotation, tmp_path / "maps")
    segment = tmp_path / "segment-episode"
    (segment / "calibration").mkdir(parents=True)
    for relative in (
        "city_SE3_egovehicle.feather",
        "calibration/egovehicle_SE3_sensor.feather",
        "calibration/intrinsics.feather",
    ):
        (segment / relative).write_bytes(b"fixture")
    for camera_index, camera in enumerate(CAMERAS):
        root = segment / "sensors" / "cameras" / camera
        root.mkdir(parents=True)
        for index in range(21 + int(camera_index == 0)):
            (root / f"{1000 + index}.jpg").write_bytes(b"image")

    scene, summary = build_argotweak_segment_episode(
        segment,
        Path(maps["prior"]),
        Path(maps["target"]),
        tmp_path / "bundles",
        split="train",
        stride=10,
    )

    assert scene.sample_id == "argotweak:segment-episode"
    assert len(scene.observations) == 2
    assert {row.modality for row in scene.observations} == {"camera_bundle"}
    assert all(Path(row.path).is_file() for row in scene.observations)
    bundle = json.loads(Path(scene.observations[0].path).read_text())
    assert len(bundle["images"]) == 7
    assert summary["evidence_bundle_count"] == 2
    assert summary["test_assets_read"] is False
