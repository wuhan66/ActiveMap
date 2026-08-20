"""Vectorize updater masks and apply typed edits to an editable vector map."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from rasterio.features import shapes
from shapely import make_valid
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from activemap.models import EditOperation, EditRecord


def vectorize_mask(
    mask: np.ndarray,
    transform: Affine,
    *,
    threshold: float = 0.5,
    min_area: float = 0.0,
    simplify_tolerance: float = 0.0,
) -> BaseGeometry | None:
    """Convert a probability/binary mask into a repaired polygonal geometry."""

    values = np.asarray(mask)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError("mask must have shape [H,W] or [1,H,W]")
    binary = values >= threshold
    geometries = [
        shape(geometry)
        for geometry, value in shapes(
            binary.astype(np.uint8), mask=binary, transform=transform
        )
        if int(value) == 1
    ]
    if not geometries:
        return None
    merged = make_valid(unary_union(geometries))
    if simplify_tolerance > 0:
        merged = make_valid(merged.simplify(simplify_tolerance, preserve_topology=True))
    polygonal = _polygonal_parts(merged, min_area=min_area)
    if not polygonal:
        return None
    return make_valid(unary_union(polygonal))


def _polygonal_parts(geometry: BaseGeometry, *, min_area: float) -> list[BaseGeometry]:
    if geometry.geom_type == "Polygon":
        return [geometry] if geometry.area >= min_area else []
    if geometry.geom_type == "MultiPolygon":
        return [part for part in geometry.geoms if part.area >= min_area]
    if geometry.geom_type == "GeometryCollection":
        return [
            part
            for item in geometry.geoms
            for part in _polygonal_parts(item, min_area=min_area)
        ]
    return []


def topology_is_valid(
    geometry: BaseGeometry | None,
    *,
    other_geometries: Iterable[BaseGeometry] = (),
    max_overlap_ratio: float = 0.25,
) -> bool:
    if geometry is None or geometry.is_empty or not geometry.is_valid:
        return False
    is_polygonal = geometry.geom_type in {"Polygon", "MultiPolygon"}
    is_linear = geometry.geom_type in {"LineString", "MultiLineString"}
    if not is_polygonal and not is_linear:
        return False
    measure = geometry.area if is_polygonal else geometry.length
    if measure <= 0:
        return False
    for other in other_geometries:
        if other.is_empty or not geometry.intersects(other):
            continue
        overlap = geometry.intersection(other)
        overlap_measure = overlap.area if is_polygonal else overlap.length
        overlap_ratio = overlap_measure / max(measure, 1e-9)
        if overlap_ratio > max_overlap_ratio:
            return False
    return True


def _generated_id(geometry: BaseGeometry) -> str:
    digest = hashlib.sha1(geometry.wkb).hexdigest()[:16]
    return f"activemap-{digest}"


def apply_edit(
    frame: gpd.GeoDataFrame,
    edit: EditRecord,
    *,
    id_column: str = "object_id",
) -> gpd.GeoDataFrame:
    """Apply one typed edit without mutating the caller's GeoDataFrame."""

    if id_column not in frame.columns:
        raise ValueError(f"map is missing ID column: {id_column}")
    output = frame.copy(deep=True)
    if edit.op == EditOperation.KEEP:
        return output
    object_id = edit.object_id
    matching = output[id_column].astype(str) == str(object_id) if object_id is not None else None
    if edit.op == EditOperation.DELETE:
        if matching is None or int(matching.sum()) != 1:
            raise ValueError(f"DELETE requires one existing object: {object_id}")
        return output.loc[~matching].reset_index(drop=True)
    if edit.geometry is None:
        raise ValueError(f"{edit.op.value} requires output geometry")
    geometry = make_valid(shape(edit.geometry.model_dump(mode="json")))
    if not topology_is_valid(geometry):
        raise ValueError(f"{edit.op.value} geometry is not line/polygon valid")
    if edit.op == EditOperation.RESHAPE:
        if matching is None or int(matching.sum()) != 1:
            raise ValueError(f"RESHAPE requires one existing object: {object_id}")
        output.loc[matching, "geometry"] = geometry
        return output
    if edit.op == EditOperation.ADD:
        new_id = object_id or _generated_id(geometry)
        if (output[id_column].astype(str) == str(new_id)).any():
            raise ValueError(f"ADD object ID already exists: {new_id}")
        row: dict[str, Any] = {column: None for column in output.columns}
        row[id_column] = new_id
        row["geometry"] = geometry
        addition = gpd.GeoDataFrame([row], geometry="geometry", crs=output.crs)
        return gpd.GeoDataFrame(
            pd.concat([output, addition], ignore_index=True),
            geometry="geometry",
            crs=output.crs,
        )
    raise ValueError(f"unsupported edit operation: {edit.op}")
