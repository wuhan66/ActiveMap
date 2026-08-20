"""Construct finite region x time x scale evidence catalogs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from rasterio.transform import Affine, rowcol
from shapely.geometry.base import BaseGeometry

from activemap.models import EvidenceItem


def evidence_cost(
    *,
    scale: int,
    temporal_distance: int,
    clear_fraction: float,
    remote_tool_penalty: float = 0.0,
) -> float:
    quality_penalty = 1.0 - clear_fraction
    return float(
        1.0
        + 0.25 * max(scale - 1, 0)
        + 0.05 * temporal_distance
        + 0.25 * quality_penalty
        + remote_tool_penalty
    )


def _timestamp_index(timestamp: str) -> int:
    year, month = (int(part) for part in timestamp.replace("-", "_").split("_")[:2])
    return year * 12 + month


def _pixel_region(
    geometry: BaseGeometry,
    transform_values: list[float] | tuple[float, ...],
    *,
    context_meters: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    transform = Affine(*transform_values[:6])
    min_x, min_y, max_x, max_y = geometry.buffer(context_meters).bounds
    top_row, left_col = rowcol(transform, min_x, max_y)
    bottom_row, right_col = rowcol(transform, max_x, min_y)
    x_min = max(0, min(left_col, right_col))
    x_max = min(width, max(left_col, right_col) + 1)
    y_min = max(0, min(top_row, bottom_row))
    y_max = min(height, max(top_row, bottom_row) + 1)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("geometry does not overlap the raster")
    return int(x_min), int(y_min), int(x_max), int(y_max)


def build_evidence_catalog(
    manifest: pd.DataFrame,
    *,
    aoi_id: str,
    anchor_timestamp: str,
    geometry: BaseGeometry,
    context_meters: tuple[float, ...] = (0.0, 32.0, 96.0),
    scales: tuple[int, ...] = (1, 2, 4),
    default_clear_fraction: float = 1.0,
) -> list[EvidenceItem]:
    required = {"aoi_id", "timestamp", "image_path", "width", "height", "transform"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing evidence columns: {sorted(missing)}")
    rows = manifest[manifest["aoi_id"].astype(str) == str(aoi_id)].sort_values("timestamp")
    if rows.empty:
        raise ValueError(f"AOI not found in manifest: {aoi_id}")

    anchor_index = _timestamp_index(anchor_timestamp)
    items: list[EvidenceItem] = []
    for _, row in rows.iterrows():
        timestamp = str(row["timestamp"])
        temporal_distance = abs(_timestamp_index(timestamp) - anchor_index)
        clear_fraction = float(row.get("clear_fraction", default_clear_fraction))
        for scale_value, context_value in zip(scales, context_meters, strict=True):
            region = _pixel_region(
                geometry,
                row["transform"],
                context_meters=context_value,
                width=int(row["width"]),
                height=int(row["height"]),
            )
            identity = f"{aoi_id}|{timestamp}|{scale_value}|{region}"
            digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
            udm_value = row.get("udm_path")
            udm_path = None if pd.isna(udm_value) else str(udm_value)
            items.append(
                EvidenceItem(
                    evidence_id=f"ev-{digest}",
                    timestamp=timestamp,
                    region=region,
                    scale=scale_value,
                    image_path=str(Path(row["image_path"])),
                    clear_fraction=clear_fraction,
                    cost=evidence_cost(
                        scale=scale_value,
                        temporal_distance=temporal_distance,
                        clear_fraction=clear_fraction,
                    ),
                    udm_path=udm_path,
                )
            )
    return items
