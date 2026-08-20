"""Shared vector-to-mask rasterization for updater training and oracle inference."""

from __future__ import annotations

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry


def rasterize_geometry_mask(
    geometry: BaseGeometry | None,
    transform: Affine,
    size: int,
) -> np.ndarray:
    """Rasterize by pixel center, preserving tiny non-empty geometries as one pixel."""

    if geometry is None or geometry.is_empty:
        return np.zeros((size, size), dtype=np.float32)
    mask = rasterize(
        [(mapping(geometry), 1.0)],
        out_shape=(size, size),
        transform=transform,
        fill=0.0,
        dtype="float32",
        all_touched=False,
    )
    if np.any(mask):
        return mask

    west, south, east, north = array_bounds(size, size, transform)
    visible_geometry = geometry.intersection(box(west, south, east, north))
    if visible_geometry.is_empty:
        return mask
    point = visible_geometry.representative_point()
    column, row = ~transform * (point.x, point.y)
    row_index, column_index = int(np.floor(row)), int(np.floor(column))
    if 0 <= row_index < size and 0 <= column_index < size:
        mask[row_index, column_index] = 1.0
    return mask
