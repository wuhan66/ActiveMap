"""Adapters for ArgoTweak annotations and matched TbV sensor subsets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from activemap.data.structured_map import StructuredMapObservation, StructuredMapScene


CAMERAS = (
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_rear_left",
    "ring_rear_right",
    "ring_side_left",
    "ring_side_right",
)


def _points(rows: Any, *, minimum: int, name: str) -> list[list[float]]:
    if not isinstance(rows, list) or len(rows) < minimum:
        raise ValueError(f"ArgoTweak {name} requires at least {minimum} points")
    try:
        return [[float(row["x"]), float(row["y"])] for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid ArgoTweak {name}") from exc


def _polygon(points: list[list[float]]) -> dict[str, Any]:
    ring = [*points]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


def _lane_feature(object_id: str, row: dict[str, Any]) -> dict[str, Any]:
    left = _points(row.get("left_lane_boundary"), minimum=2, name="left lane boundary")
    right = _points(row.get("right_lane_boundary"), minimum=2, name="right lane boundary")
    properties = {
        "object_id": object_id,
        "feature_class": "lane_segment",
        "lane_type": str(row.get("lane_type", "UNKNOWN")),
        "is_intersection": bool(row.get("is_intersection", False)),
        "left_lane_mark_type": str(row.get("left_lane_mark_type", "UNKNOWN")),
        "right_lane_mark_type": str(row.get("right_lane_mark_type", "UNKNOWN")),
        "predecessors": ",".join(f"ls-{value}" for value in row.get("predecessors") or []),
        "successors": ",".join(f"ls-{value}" for value in row.get("successors") or []),
    }
    return {
        "type": "Feature",
        "id": object_id,
        "properties": properties,
        "geometry": _polygon([*left, *reversed(right)]),
    }


def _crossing_feature(object_id: str, row: dict[str, Any]) -> dict[str, Any]:
    points = _points(row.get("edges"), minimum=3, name="pedestrian crossing")
    return {
        "type": "Feature",
        "id": object_id,
        "properties": {"object_id": object_id, "feature_class": "pedestrian_crossing"},
        "geometry": _polygon(points),
    }


def _drivable_feature(object_id: str, row: dict[str, Any]) -> dict[str, Any]:
    points = _points(row.get("area_boundary"), minimum=3, name="drivable area")
    return {
        "type": "Feature",
        "id": object_id,
        "properties": {"object_id": object_id, "feature_class": "drivable_area"},
        "geometry": _polygon(points),
    }


def _feature(section: str, object_id: str, row: dict[str, Any]) -> dict[str, Any]:
    if section == "laneSegments":
        return _lane_feature(object_id, row)
    if section == "pedCrossings":
        return _crossing_feature(object_id, row)
    if section == "drivableAreas":
        return _drivable_feature(object_id, row)
    raise ValueError(f"unsupported ArgoTweak section: {section}")


def argotweak_annotation_to_geojson(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    """Return stale-prior and current-target maps from one annotation.

    ArgoTweak's official loader defines ``old`` as the current ground-truth
    object and ``new`` as the modified stale-map object. Unchanged objects use
    ``old`` on both sides. This direction is intentionally not inferred from
    the field names.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ArgoTweak annotation must contain an object")
    prior: list[dict[str, Any]] = []
    target: list[dict[str, Any]] = []
    counts = {"KEEP": 0, "ADD": 0, "DELETE": 0, "RESHAPE": 0}
    for section in ("laneSegments", "pedCrossings", "drivableAreas"):
        objects = payload.get(section, {})
        if not isinstance(objects, dict):
            raise ValueError(f"ArgoTweak {section} must contain an object")
        for raw_id, change in sorted(objects.items()):
            if not isinstance(change, dict):
                raise ValueError(f"invalid ArgoTweak object {raw_id}")
            object_id = str(raw_id)
            codes = change.get("changes")
            unchanged = codes == [0]
            current = change.get("old")
            stale = current if unchanged else change.get("new")
            if stale is not None:
                prior.append(_feature(section, object_id, stale))
            if current is not None:
                target.append(_feature(section, object_id, current))
            if stale is None and current is not None:
                counts["ADD"] += 1
            elif stale is not None and current is None:
                counts["DELETE"] += 1
            elif stale is not None and current is not None:
                counts["KEEP" if unchanged else "RESHAPE"] += 1

    if not prior or not target:
        raise ValueError(f"ArgoTweak annotation lacks prior or target objects: {path}")

    def collection(features: list[dict[str, Any]], role: str) -> dict[str, Any]:
        return {
            "type": "FeatureCollection",
            "features": sorted(features, key=lambda row: str(row["id"])),
            "properties": {
                "source_format": "argotweak-v1-annotation",
                "source_path": str(path.resolve()),
                "map_role": role,
            },
        }

    return collection(prior, "stale_prior"), collection(target, "current_target"), counts


def convert_argotweak_annotation(path: Path, output_dir: Path) -> dict[str, Any]:
    """Write stale-prior and current-target GeoJSON without overwriting assets."""

    prior_path = output_dir / "prior.geojson"
    target_path = output_dir / "target.geojson"
    if prior_path.exists() or target_path.exists():
        raise FileExistsError(f"refusing to overwrite ArgoTweak conversion: {output_dir}")
    prior, target, counts = argotweak_annotation_to_geojson(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_path.write_text(json.dumps(prior, separators=(",", ":")) + "\n", encoding="utf-8")
    target_path.write_text(json.dumps(target, separators=(",", ":")) + "\n", encoding="utf-8")
    return {
        "schema_version": "activemap-argotweak-map-conversion-v1",
        "source": str(path.resolve()),
        "prior": str(prior_path.resolve()),
        "target": str(target_path.resolve()),
        "operation_counts": counts,
        "test_assets_read": False,
    }


def _split_rows(payload: dict[str, Any], splits: Iterable[str]) -> Iterable[tuple[str, str]]:
    for split in splits:
        rows = payload.get(split)
        if not isinstance(rows, dict):
            raise ValueError(f"ArgoTweak official split {split!r} is missing")
        for _, segment_id in sorted(rows.items()):
            yield split, str(segment_id)


def build_argotweak_tbv_subset_manifest(
    splits_path: Path,
    annotation_root: Path,
    output_path: Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
) -> dict[str, Any]:
    """Build an exact train/val TbV dependency list from official split IDs."""

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite TbV subset manifest: {output_path}")
    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    rows = []
    missing_annotations = []
    split_counts = {split: 0 for split in splits}
    for split, segment_id in _split_rows(payload, splits):
        annotation = annotation_root / f"{segment_id}.json"
        if not annotation.is_file():
            missing_annotations.append(segment_id)
            continue
        rows.append(
            {
                "schema_version": "activemap-argotweak-tbv-dependency-v1",
                "split": split,
                "segment_id": segment_id,
                "annotation_path": str(annotation.resolve()),
                "required_files": [
                    f"{segment_id}/city_SE3_egovehicle.feather",
                    f"{segment_id}/calibration/egovehicle_SE3_sensor.feather",
                    f"{segment_id}/calibration/intrinsics.feather",
                ],
                "camera_globs": [
                    f"{segment_id}/sensors/cameras/{camera}/*.jpg" for camera in CAMERAS
                ],
                "test_assets_read": False,
            }
        )
        split_counts[split] += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "activemap-argotweak-tbv-subset-manifest-v1",
        "splits": list(splits),
        "rows": len(rows),
        "split_counts": split_counts,
        "missing_annotation_count": len(missing_annotations),
        "missing_annotation_ids": missing_annotations,
        "cameras": list(CAMERAS),
        "test_assets_read": False,
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_argotweak_segment_scenes(
    segment_root: Path,
    prior_map: Path,
    target_map: Path,
    output_path: Path,
    *,
    split: str,
    stride: int = 10,
    camera_cost: float = 1.0,
) -> dict[str, Any]:
    """Create synchronized seven-camera scenes for one downloaded TbV segment."""

    if split not in {"train", "val"}:
        raise ValueError("ArgoTweak scene construction only permits train or val")
    if stride < 1:
        raise ValueError("stride must be positive")
    if camera_cost <= 0:
        raise ValueError("camera_cost must be positive")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite ArgoTweak scenes: {output_path}")
    required = (
        segment_root / "city_SE3_egovehicle.feather",
        segment_root / "calibration" / "egovehicle_SE3_sensor.feather",
        segment_root / "calibration" / "intrinsics.feather",
        prior_map,
        target_map,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    camera_files = {
        camera: sorted((segment_root / "sensors" / "cameras" / camera).glob("*.jpg"))
        for camera in CAMERAS
    }
    if any(not rows for rows in camera_files.values()):
        missing = [camera for camera, rows in camera_files.items() if not rows]
        raise FileNotFoundError(f"TbV segment lacks camera frames: {missing}")
    frame_count = len(camera_files["ring_front_center"])
    selected = list(range(0, frame_count, stride))[:-1]
    if any(selected and len(rows) <= selected[-1] for rows in camera_files.values()):
        raise ValueError(f"TbV camera is shorter than the official front-center iteration: {segment_root}")
    scenes = []
    segment_id = segment_root.name
    for index in selected:
        reference = camera_files["ring_front_center"][index]
        timestamp = reference.stem
        observations = []
        for camera in CAMERAS:
            image = camera_files[camera][index]
            observations.append(
                StructuredMapObservation(
                    observation_id=f"{camera}:{image.stem}",
                    timestamp=image.stem,
                    modality="camera",
                    path=str(image.resolve()),
                    cost=camera_cost,
                    metadata={"camera": camera, "frame_index": index},
                )
            )
        scenes.append(
            StructuredMapScene(
                schema_version="activemap-structured-map-scene-v1",
                sample_id=f"argotweak:{segment_id}:{timestamp}",
                dataset="argotweak",
                split=split,
                aoi_id=segment_id,
                native_sample_id=f"{segment_id}:{timestamp}",
                prior_map_path=str(prior_map.resolve()),
                target_map_path=str(target_map.resolve()),
                observations=observations,
                metadata={
                    "pose_path": str(required[0].resolve()),
                    "extrinsics_path": str(required[1].resolve()),
                    "intrinsics_path": str(required[2].resolve()),
                    "official_frame_stride": stride,
                },
            )
        )
    if not scenes:
        raise ValueError(f"TbV segment has no selectable frames: {segment_root}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(row.model_dump_json() + "\n" for row in scenes), encoding="utf-8"
    )
    summary = {
        "schema_version": "activemap-argotweak-segment-scenes-v1",
        "segment_id": segment_id,
        "split": split,
        "source_frame_count": frame_count,
        "stride": stride,
        "scene_count": len(scenes),
        "observations_per_scene": len(CAMERAS),
        "camera_frame_counts": {camera: len(rows) for camera, rows in camera_files.items()},
        "test_assets_read": False,
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_argotweak_segment_episode(
    segment_root: Path,
    prior_map: Path,
    target_map: Path,
    bundle_root: Path,
    *,
    split: str,
    stride: int = 10,
    bundle_cost: float = 1.0,
) -> tuple[StructuredMapScene, dict[str, Any]]:
    """Build one segment-level episode with selectable seven-camera bundles."""

    if split not in {"train", "val"}:
        raise ValueError("ArgoTweak episode construction only permits train or val")
    if stride < 1:
        raise ValueError("stride must be positive")
    if bundle_cost <= 0:
        raise ValueError("bundle_cost must be positive")
    required = (
        segment_root / "city_SE3_egovehicle.feather",
        segment_root / "calibration" / "egovehicle_SE3_sensor.feather",
        segment_root / "calibration" / "intrinsics.feather",
        prior_map,
        target_map,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    camera_files = {
        camera: sorted((segment_root / "sensors" / "cameras" / camera).glob("*.jpg"))
        for camera in CAMERAS
    }
    if any(not rows for rows in camera_files.values()):
        missing = [camera for camera, rows in camera_files.items() if not rows]
        raise FileNotFoundError(f"TbV segment lacks camera frames: {missing}")
    frame_count = len(camera_files["ring_front_center"])
    selected = list(range(0, frame_count, stride))[:-1]
    if any(selected and len(rows) <= selected[-1] for rows in camera_files.values()):
        raise ValueError(f"TbV camera is shorter than the official front-center iteration: {segment_root}")
    if not selected:
        raise ValueError(f"TbV segment has no selectable frames: {segment_root}")
    bundle_root.mkdir(parents=True, exist_ok=True)
    observations = []
    for index in selected:
        timestamp = camera_files["ring_front_center"][index].stem
        bundle_path = bundle_root / f"{timestamp}.json"
        bundle = {
            "schema_version": "activemap-seven-camera-bundle-v1",
            "segment_id": segment_root.name,
            "timestamp": timestamp,
            "frame_index": index,
            "images": {
                camera: str(camera_files[camera][index].resolve()) for camera in CAMERAS
            },
            "pose_path": str(required[0].resolve()),
            "extrinsics_path": str(required[1].resolve()),
            "intrinsics_path": str(required[2].resolve()),
            "test_assets_read": False,
        }
        if bundle_path.exists():
            existing = json.loads(bundle_path.read_text(encoding="utf-8"))
            if existing != bundle:
                raise FileExistsError(f"camera bundle differs from frozen artifact: {bundle_path}")
        else:
            bundle_path.write_text(json.dumps(bundle, separators=(",", ":")) + "\n", encoding="utf-8")
        observations.append(
            StructuredMapObservation(
                observation_id=f"bundle:{timestamp}",
                timestamp=timestamp,
                modality="camera_bundle",
                path=str(bundle_path.resolve()),
                cost=bundle_cost,
                metadata={"frame_index": index, "camera_count": len(CAMERAS)},
            )
        )
    scene = StructuredMapScene(
        schema_version="activemap-structured-map-scene-v1",
        sample_id=f"argotweak:{segment_root.name}",
        dataset="argotweak",
        split=split,
        aoi_id=segment_root.name,
        native_sample_id=segment_root.name,
        prior_map_path=str(prior_map.resolve()),
        target_map_path=str(target_map.resolve()),
        observations=observations,
        metadata={
            "official_frame_stride": stride,
            "evidence_unit": "synchronized_seven_camera_frame_bundle",
            "source_frame_count": frame_count,
        },
    )
    summary = {
        "schema_version": "activemap-argotweak-segment-episode-v1",
        "segment_id": segment_root.name,
        "split": split,
        "source_frame_count": frame_count,
        "stride": stride,
        "evidence_bundle_count": len(observations),
        "cameras_per_bundle": len(CAMERAS),
        "camera_frame_counts": {camera: len(rows) for camera, rows in camera_files.items()},
        "test_assets_read": False,
    }
    return scene, summary
