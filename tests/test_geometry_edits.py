import geopandas as gpd
import pytest
from shapely.geometry import box

import activemap.geometry as geometry_module
from activemap.data.edits import derive_edit_events, edit_event_summary
from activemap.data.perturb import generate_synthetic_case
from activemap.data.sn7_pairs import filter_persistent_changes
from activemap.geometry import geometry_features, geometry_iou, match_geometry_sets
from activemap.models import EditOperation


def test_geometry_iou_features_and_matching() -> None:
    left = box(0, 0, 10, 10)
    right = box(5, 0, 15, 10)
    assert geometry_iou(left, right) == pytest.approx(1 / 3)
    assert geometry_features(left).shape == (8,)
    matches = match_geometry_sets([left], [right], min_iou=0.2)
    assert len(matches) == 1
    assert matches[0].iou == pytest.approx(1 / 3)


def test_spatial_matching_only_scores_local_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = [box(index * 10, 0, index * 10 + 2, 2) for index in range(100)]
    new = [box(index * 10 + 0.1, 0, index * 10 + 2.1, 2) for index in range(100)]
    original = geometry_module.geometry_iou
    calls = 0

    def counted_iou(left: object, right: object) -> float:
        nonlocal calls
        calls += 1
        return original(left, right)  # type: ignore[arg-type]

    monkeypatch.setattr(geometry_module, "geometry_iou", counted_iou)
    matches = match_geometry_sets(old, new, min_iou=0.2)
    assert len(matches) == 100
    assert calls == 100


def test_derive_all_edit_operations() -> None:
    old = gpd.GeoDataFrame(
        {
            "object_id": ["keep", "reshape", "delete"],
            "geometry": [box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 10)],
        },
        crs="EPSG:3857",
    )
    new = gpd.GeoDataFrame(
        {
            "object_id": ["keep", "reshape", "add"],
            "geometry": [box(0, 0, 10, 10), box(24, 0, 34, 10), box(60, 0, 70, 10)],
        },
        crs="EPSG:3857",
    )
    events = derive_edit_events(old, new)
    summary = edit_event_summary(events)
    assert summary == {"KEEP": 1, "ADD": 1, "DELETE": 1, "RESHAPE": 1, "total": 4}
    for event in events:
        event.to_edit_record()


def test_change_persistence_rejects_label_flicker() -> None:
    old = gpd.GeoDataFrame(
        {
            "object_id": [
                "keep",
                "stable-delete",
                "returning-delete",
                "stable-reshape",
                "reverting-reshape",
            ],
            "geometry": [
                box(0, 0, 10, 10),
                box(20, 0, 30, 10),
                box(40, 0, 50, 10),
                box(100, 0, 110, 10),
                box(120, 0, 130, 10),
            ],
        },
        crs="EPSG:3857",
    )
    current = gpd.GeoDataFrame(
        {
            "object_id": [
                "keep",
                "stable-add",
                "flash-add",
                "stable-reshape",
                "reverting-reshape",
            ],
            "geometry": [
                box(0, 0, 10, 10),
                box(60, 0, 70, 10),
                box(80, 0, 90, 10),
                box(103, 0, 113, 10),
                box(123, 0, 133, 10),
            ],
        },
        crs="EPSG:3857",
    )
    future = gpd.GeoDataFrame(
        {
            "object_id": [
                "keep",
                "stable-add",
                "returning-delete",
                "stable-reshape",
                "reverting-reshape",
            ],
            "geometry": [
                box(0, 0, 10, 10),
                box(60, 0, 70, 10),
                box(40, 0, 50, 10),
                box(103, 0, 113, 10),
                box(120, 0, 130, 10),
            ],
        },
        crs="EPSG:3857",
    )
    events = derive_edit_events(old, current)
    filtered = filter_persistent_changes(
        events,
        [current, future],
        min_change_persistence=2,
    )
    assert {(event.op, event.object_id) for event in filtered} == {
        (EditOperation.KEEP, "keep"),
        (EditOperation.ADD, "stable-add"),
        (EditOperation.DELETE, "stable-delete"),
        (EditOperation.RESHAPE, "stable-reshape"),
    }


def test_change_persistence_requires_enough_future_frames() -> None:
    old = gpd.GeoDataFrame(
        {"object_id": ["delete"], "geometry": [box(0, 0, 10, 10)]},
        crs="EPSG:3857",
    )
    current = gpd.GeoDataFrame(
        {"object_id": ["add"], "geometry": [box(20, 0, 30, 10)]},
        crs="EPSG:3857",
    )
    events = derive_edit_events(old, current)
    assert not filter_persistent_changes(
        events,
        [current],
        min_change_persistence=2,
    )


def test_far_persistent_id_is_split_into_persistent_add_delete() -> None:
    old = gpd.GeoDataFrame(
        {"object_id": ["reused-id"], "geometry": [box(0, 0, 10, 10)]},
        crs="EPSG:3857",
    )
    current = gpd.GeoDataFrame(
        {"object_id": ["reused-id"], "geometry": [box(100, 0, 110, 10)]},
        crs="EPSG:3857",
    )
    future = current.copy()
    events = derive_edit_events(old, current, max_centroid_distance=20.0)
    assert {(event.op, event.object_id) for event in events} == {
        (EditOperation.ADD, "reused-id"),
        (EditOperation.DELETE, "reused-id"),
    }
    filtered = filter_persistent_changes(
        events,
        [current, future],
        min_change_persistence=2,
    )
    assert {(event.op, event.object_id) for event in filtered} == {
        (EditOperation.ADD, "reused-id"),
        (EditOperation.DELETE, "reused-id"),
    }


@pytest.mark.parametrize(
    "operation",
    [
        EditOperation.KEEP,
        EditOperation.ADD,
        EditOperation.DELETE,
        EditOperation.RESHAPE,
    ],
)
def test_synthetic_prior_operations(operation: EditOperation) -> None:
    target = gpd.GeoDataFrame(
        {"object_id": ["obj-1", "obj-2"], "geometry": [box(0, 0, 10, 10), box(30, 0, 40, 10)]},
        crs="EPSG:3857",
    )
    result = generate_synthetic_case(
        target,
        operation=operation,
        object_id="obj-1",
        seed=7,
    )
    if operation == EditOperation.ADD:
        assert "obj-1" not in set(result.prior_map["object_id"])
    elif operation == EditOperation.DELETE:
        assert len(result.prior_map) == len(target) + 1
        assert result.target_geometry is None
    elif operation == EditOperation.RESHAPE:
        original = target.loc[target["object_id"] == "obj-1"].geometry.iloc[0]
        perturbed = result.prior_map.loc[result.prior_map["object_id"] == "obj-1"].geometry.iloc[0]
        assert not original.equals(perturbed)
    else:
        assert result.prior_map.geometry.equals(target.geometry)
