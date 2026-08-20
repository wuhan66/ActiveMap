import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import LineString, box

from activemap.models import EditOperation, EditRecord, GeoJSONGeometry
from activemap.vector_map import apply_edit, topology_is_valid, vectorize_mask


def _geometry(min_x: float, min_y: float, max_x: float, max_y: float) -> GeoJSONGeometry:
    return GeoJSONGeometry(
        type="Polygon",
        coordinates=[
            [
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
                [min_x, min_y],
            ]
        ],
    )


def test_vectorize_mask_and_topology() -> None:
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[3:9, 4:12] = 1.0
    geometry = vectorize_mask(mask, Affine.identity())
    assert geometry is not None
    assert geometry.area == pytest.approx(48.0)
    assert topology_is_valid(geometry)
    assert not topology_is_valid(None)


def test_apply_typed_edits_without_mutating_input() -> None:
    original = gpd.GeoDataFrame(
        {"object_id": ["one", "two"]},
        geometry=[box(0, 0, 2, 2), box(5, 5, 7, 7)],
        crs="EPSG:3857",
    )
    reshaped = apply_edit(
        original,
        EditRecord(
            op=EditOperation.RESHAPE,
            object_id="one",
            geometry=_geometry(0, 0, 3, 3),
        ),
    )
    assert original.loc[0].geometry.area == pytest.approx(4.0)
    assert reshaped.loc[0].geometry.area == pytest.approx(9.0)
    deleted = apply_edit(
        reshaped,
        EditRecord(op=EditOperation.DELETE, object_id="two"),
    )
    assert list(deleted["object_id"]) == ["one"]
    added = apply_edit(
        deleted,
        EditRecord(
            op=EditOperation.ADD,
            object_id="three",
            geometry=_geometry(10, 10, 12, 12),
        ),
    )
    assert set(added["object_id"]) == {"one", "three"}


def test_apply_line_edits_for_road_maps() -> None:
    original = gpd.GeoDataFrame(
        {"object_id": ["road-1"]},
        geometry=[LineString([(0, 0), (2, 0)])],
        crs="EPSG:3857",
    )
    reshaped_geometry = GeoJSONGeometry(
        type="LineString", coordinates=[[0, 0], [1, 1], [3, 1]]
    )
    added_geometry = GeoJSONGeometry(
        type="LineString", coordinates=[[5, 5], [8, 5]]
    )

    reshaped = apply_edit(
        original,
        EditRecord(
            op=EditOperation.RESHAPE,
            object_id="road-1",
            geometry=reshaped_geometry,
        ),
    )
    added = apply_edit(
        reshaped,
        EditRecord(
            op=EditOperation.ADD,
            object_id="road-2",
            geometry=added_geometry,
        ),
    )

    assert topology_is_valid(reshaped.loc[0].geometry)
    assert reshaped.loc[0].geometry.length == pytest.approx(1.4142 + 2.0, rel=1e-3)
    assert set(added["object_id"]) == {"road-1", "road-2"}
