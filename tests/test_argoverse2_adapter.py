from __future__ import annotations

import json
from pathlib import Path

from activemap.data.argoverse2 import convert_argoverse2_static_map
from activemap.data.structured_map import derive_structured_atomic_edits


def _point(x: float, y: float) -> dict[str, float]:
    return {"x": x, "y": y, "z": 0.0}


def _map_payload(*, lane_end: float = 2.0) -> dict:
    return {
        "lane_segments": {
            "1": {
                "id": 1,
                "is_intersection": False,
                "lane_type": "VEHICLE",
                "left_lane_boundary": [_point(0, 1), _point(lane_end, 1)],
                "right_lane_boundary": [_point(0, 0), _point(lane_end, 0)],
                "left_lane_mark_type": "SOLID_WHITE",
                "right_lane_mark_type": "DASHED_WHITE",
                "predecessors": [],
                "successors": [2],
                "left_neighbor_id": None,
                "right_neighbor_id": 2,
            }
        },
        "drivable_areas": {
            "10": {
                "id": 10,
                "area_boundary": [_point(0, 0), _point(2, 0), _point(2, 2)],
            }
        },
        "pedestrian_crossings": {
            "20": {
                "id": 20,
                "edge1": [_point(0.5, 0), _point(0.5, 1)],
                "edge2": [_point(1.0, 0), _point(1.0, 1)],
            }
        },
    }


def test_convert_native_av2_map_and_derive_edit(tmp_path: Path) -> None:
    prior_native = tmp_path / "log_map_archive_prior.json"
    target_native = tmp_path / "log_map_archive_target.json"
    prior_native.write_text(json.dumps(_map_payload()), encoding="utf-8")
    target_native.write_text(json.dumps(_map_payload(lane_end=3.0)), encoding="utf-8")
    prior = tmp_path / "prior.geojson"
    target = tmp_path / "target.geojson"

    summary = convert_argoverse2_static_map(prior_native, prior)
    convert_argoverse2_static_map(target_native, target)
    collection = json.loads(prior.read_text(encoding="utf-8"))

    assert summary["counts"] == {
        "lane_segments": 1,
        "drivable_areas": 1,
        "pedestrian_crossings": 1,
        "features": 3,
    }
    assert {feature["id"] for feature in collection["features"]} == {
        "lane_segment:1",
        "drivable_area:10",
        "pedestrian_crossing:20",
    }
    assert all(
        feature["geometry"]["coordinates"][0][0] == feature["geometry"]["coordinates"][0][-1]
        for feature in collection["features"]
    )
    edits, counts = derive_structured_atomic_edits(prior, target)
    assert [(edit.object_id, edit.operation.value) for edit in edits] == [
        ("lane_segment:1", "RESHAPE")
    ]
    assert counts == {"KEEP": 2, "ADD": 0, "DELETE": 0, "RESHAPE": 1}
