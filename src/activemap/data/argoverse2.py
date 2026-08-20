"""Dependency-free adapter for native Argoverse 2 static vector-map JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _entities(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key, {})
    rows = list(value.values()) if isinstance(value, dict) else value
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"AV2 {key} must be an object or list of objects")
    return rows


def _point(value: dict[str, Any]) -> list[float]:
    try:
        return [float(value["x"]), float(value["y"])]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid AV2 map point: {value!r}") from exc


def _polyline(value: Any, *, name: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"AV2 {name} requires at least two points")
    return [_point(point) for point in value]


def _closed_polygon(points: list[list[float]], *, name: str) -> dict[str, Any]:
    if len(points) < 3:
        raise ValueError(f"AV2 {name} requires at least three polygon vertices")
    ring = [*points]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


def _lane_feature(row: dict[str, Any]) -> dict[str, Any]:
    lane_id = str(row["id"])
    left = _polyline(row.get("left_lane_boundary"), name="left lane boundary")
    right = _polyline(row.get("right_lane_boundary"), name="right lane boundary")
    geometry = _closed_polygon([*left, *reversed(right)], name="lane segment")
    return {
        "type": "Feature",
        "id": f"lane_segment:{lane_id}",
        "properties": {
            "object_id": f"lane_segment:{lane_id}",
            "native_id": lane_id,
            "feature_class": "lane_segment",
            "lane_type": str(row.get("lane_type", "UNKNOWN")),
            "is_intersection": bool(row.get("is_intersection", False)),
            "left_lane_mark_type": str(row.get("left_lane_mark_type", "UNKNOWN")),
            "right_lane_mark_type": str(row.get("right_lane_mark_type", "UNKNOWN")),
            "predecessors": ",".join(map(str, row.get("predecessors") or [])),
            "successors": ",".join(map(str, row.get("successors") or [])),
            "left_neighbor_id": str(row.get("left_neighbor_id") or ""),
            "right_neighbor_id": str(row.get("right_neighbor_id") or ""),
        },
        "geometry": geometry,
    }


def _drivable_area_feature(row: dict[str, Any]) -> dict[str, Any]:
    area_id = str(row["id"])
    boundary = _polyline(row.get("area_boundary"), name="drivable area boundary")
    return {
        "type": "Feature",
        "id": f"drivable_area:{area_id}",
        "properties": {
            "object_id": f"drivable_area:{area_id}",
            "native_id": area_id,
            "feature_class": "drivable_area",
        },
        "geometry": _closed_polygon(boundary, name="drivable area"),
    }


def _crossing_feature(row: dict[str, Any]) -> dict[str, Any]:
    crossing_id = str(row["id"])
    edge1 = _polyline(row.get("edge1"), name="pedestrian crossing edge1")
    edge2 = _polyline(row.get("edge2"), name="pedestrian crossing edge2")
    return {
        "type": "Feature",
        "id": f"pedestrian_crossing:{crossing_id}",
        "properties": {
            "object_id": f"pedestrian_crossing:{crossing_id}",
            "native_id": crossing_id,
            "feature_class": "pedestrian_crossing",
        },
        "geometry": _closed_polygon([*edge1, *reversed(edge2)], name="pedestrian crossing"),
    }


def argoverse2_static_map_to_geojson(path: Path) -> tuple[dict[str, Any], dict[str, int]]:
    """Convert one official AV2 per-scenario vector map to stable GeoJSON objects."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("AV2 static map JSON must contain an object")
    lanes = [_lane_feature(row) for row in _entities(payload, "lane_segments")]
    areas = [_drivable_area_feature(row) for row in _entities(payload, "drivable_areas")]
    crossings = [_crossing_feature(row) for row in _entities(payload, "pedestrian_crossings")]
    features = sorted([*lanes, *areas, *crossings], key=lambda row: str(row["id"]))
    if not features:
        raise ValueError(f"AV2 static map has no vector objects: {path}")
    return (
        {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "source_format": "argoverse2-static-map-json",
                "source_path": str(path.resolve()),
            },
        },
        {
            "lane_segments": len(lanes),
            "drivable_areas": len(areas),
            "pedestrian_crossings": len(crossings),
            "features": len(features),
        },
    )


def convert_argoverse2_static_map(path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite AV2 GeoJSON: {output}")
    collection, counts = argoverse2_static_map_to_geojson(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(collection, separators=(",", ":")) + "\n", encoding="utf-8")
    return {
        "schema_version": "activemap-av2-static-map-conversion-v1",
        "source": str(path.resolve()),
        "output": str(output.resolve()),
        "counts": counts,
    }
