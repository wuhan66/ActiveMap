"""Normalize SpaceNet 7 labels and derive leakage-safe adjacent snapshot pairs."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import rasterio
from pyproj import CRS

from activemap.data.edits import EditEvent, derive_edit_events
from activemap.geometry import geometry_iou
from activemap.models import EditOperation

ID_CANDIDATES = ("object_id", "building_id", "BuildingId", "Id", "id")


@dataclass(frozen=True)
class SnapshotPair:
    aoi_id: str
    split: str
    old_timestamp: str
    new_timestamp: str
    old_label_path: Path
    new_label_path: Path
    old_image_path: Path
    image_path: Path
    old_udm_path: Path | None
    udm_path: Path | None
    month_gap: int
    old_map: gpd.GeoDataFrame
    new_map: gpd.GeoDataFrame
    events: tuple[EditEvent, ...]


@dataclass(frozen=True)
class _LoadedSnapshot:
    timestamp: str
    split: str
    label_path: Path
    image_path: Path
    udm_path: Path | None
    map_frame: gpd.GeoDataFrame


def _path_value(value: Any) -> Path | None:
    if value is None or pd.isna(value):
        return None
    return Path(str(value))


def month_index(timestamp: str) -> int:
    """Convert YYYY_MM or YYYY-MM timestamps to a monotonically increasing month."""

    parts = timestamp.replace("-", "_").split("_")
    if len(parts) < 2:
        raise ValueError(f"timestamp must contain year and month: {timestamp!r}")
    year, month = int(parts[0]), int(parts[1])
    if month < 1 or month > 12:
        raise ValueError(f"timestamp month must be in [1, 12]: {timestamp!r}")
    return year * 12 + month


def load_label_map(
    path: Path,
    *,
    target_crs: CRS | str | None,
    id_column: str | None = None,
    min_area: float = 0.0,
) -> gpd.GeoDataFrame:
    frame = gpd.read_file(path)
    if "geometry" not in frame or frame.empty:
        return gpd.GeoDataFrame({"object_id": []}, geometry=[], crs=target_crs)
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    frame.geometry = frame.geometry.make_valid()
    frame = frame[frame.geometry.geom_type.isin({"Polygon", "MultiPolygon"})].copy()
    if target_crs is not None:
        if frame.crs is None:
            frame = frame.set_crs(target_crs, allow_override=True)
        elif CRS.from_user_input(frame.crs) != CRS.from_user_input(target_crs):
            frame = frame.to_crs(target_crs)

    selected_id = id_column if id_column in frame.columns else None
    if selected_id is None:
        selected_id = next((name for name in ID_CANDIDATES if name in frame.columns), None)
    if selected_id is None:
        frame["object_id"] = [f"{path.stem}-row-{index}" for index in range(len(frame))]
    else:
        frame["object_id"] = frame[selected_id].astype(str)
        if frame["object_id"].duplicated().any():
            duplicates = sorted(frame.loc[frame["object_id"].duplicated(), "object_id"].unique())
            raise ValueError(f"duplicate object IDs in {path}: {duplicates[:5]}")
    if min_area > 0:
        frame = frame[frame.geometry.area >= min_area].copy()
    return frame[["object_id", "geometry"]].reset_index(drop=True)


def _limit_events(
    events: Sequence[EditEvent],
    *,
    max_events_per_operation: int | None,
    sampling_key: str,
    sampling_seed: int,
) -> tuple[EditEvent, ...]:
    if max_events_per_operation is None:
        return tuple(events)
    selected_indices: set[int] = set()
    for operation in EditOperation:
        candidates = [
            (index, event) for index, event in enumerate(events) if event.op == operation
        ]
        ranked = sorted(
            candidates,
            key=lambda item: (
                hashlib.sha1(
                    (
                        f"{sampling_seed}|{sampling_key}|{operation.value}|"
                        f"{item[1].object_id}"
                    ).encode()
                ).hexdigest(),
                item[1].object_id,
            ),
        )
        selected_indices.update(index for index, _ in ranked[:max_events_per_operation])
    return tuple(event for index, event in enumerate(events) if index in selected_indices)


def filter_persistent_changes(
    events: Sequence[EditEvent],
    future_maps: Sequence[gpd.GeoDataFrame],
    *,
    min_change_persistence: int,
    presence_iou_min: float = 0.20,
) -> list[EditEvent]:
    """Reject ADD/DELETE events that do not persist across later label snapshots."""

    if min_change_persistence < 1:
        raise ValueError("min_change_persistence must be at least 1")
    if not 0.0 <= presence_iou_min <= 1.0:
        raise ValueError("presence_iou_min must be between zero and one")
    if min_change_persistence == 1:
        return list(events)

    persistent_maps = future_maps[:min_change_persistence]
    object_maps = [
        {
            str(object_id): geometry
            for object_id, geometry in zip(
                frame["object_id"], frame.geometry, strict=True
            )
        }
        for frame in persistent_maps
    ]
    enough_future = len(object_maps) == min_change_persistence
    selected: list[EditEvent] = []
    for event in events:
        if event.op == EditOperation.ADD:
            if enough_future and all(event.object_id in objects for objects in object_maps):
                selected.append(event)
        elif event.op == EditOperation.DELETE:
            if (
                enough_future
                and event.old_geometry is not None
                and all(
                    event.object_id not in objects
                    or geometry_iou(event.old_geometry, objects[event.object_id])
                    < presence_iou_min
                    for objects in object_maps
                )
            ):
                selected.append(event)
        elif event.op == EditOperation.RESHAPE:
            if (
                enough_future
                and event.old_geometry is not None
                and event.new_geometry is not None
                and all(event.object_id in objects for objects in object_maps)
                and all(
                    geometry_iou(event.new_geometry, objects[event.object_id])
                    >= geometry_iou(event.old_geometry, objects[event.object_id])
                    for objects in object_maps
                )
            ):
                selected.append(event)
        else:
            selected.append(event)
    return selected


def _future_maps_in_sequence(
    snapshots: Sequence[_LoadedSnapshot],
    start_index: int,
    *,
    max_month_gap: int | None,
) -> list[gpd.GeoDataFrame]:
    selected = [snapshots[start_index].map_frame]
    for index in range(start_index + 1, len(snapshots)):
        gap = month_index(snapshots[index].timestamp) - month_index(
            snapshots[index - 1].timestamp
        )
        if max_month_gap is not None and gap > max_month_gap:
            break
        selected.append(snapshots[index].map_frame)
    return selected


def iter_snapshot_pairs(
    manifest: pd.DataFrame,
    *,
    id_column: str | None = None,
    keep_iou_min: float = 0.80,
    fallback_match_iou_min: float = 0.20,
    max_centroid_distance: float | None = None,
    min_area: float = 0.0,
    max_events_per_operation: int | None = None,
    max_month_gap: int | None = None,
    min_change_persistence: int = 1,
    sampling_seed: int = 20260710,
) -> Iterator[SnapshotPair]:
    required = {"aoi_id", "timestamp", "image_path", "label_path", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing snapshot-pair columns: {sorted(missing)}")
    if max_month_gap is not None and max_month_gap < 1:
        raise ValueError("max_month_gap must be at least 1 when provided")
    if min_change_persistence < 1:
        raise ValueError("min_change_persistence must be at least 1")

    for aoi_id, group in manifest.groupby("aoi_id", sort=True):
        labeled = group[group["label_path"].notna()].sort_values("timestamp").reset_index(drop=True)
        snapshots: list[_LoadedSnapshot] = []
        for _, row in labeled.iterrows():
            label_path = _path_value(row["label_path"])
            image_path = _path_value(row["image_path"])
            if label_path is None or image_path is None:
                continue
            with rasterio.open(image_path) as dataset:
                target_crs = dataset.crs
            snapshots.append(
                _LoadedSnapshot(
                    timestamp=str(row["timestamp"]),
                    split=str(row["split"]),
                    label_path=label_path,
                    image_path=image_path,
                    udm_path=_path_value(row.get("udm_path")),
                    map_frame=load_label_map(
                        label_path,
                        target_crs=target_crs,
                        id_column=id_column,
                        min_area=min_area,
                    ),
                )
            )

        for index in range(1, len(snapshots)):
            old_snapshot = snapshots[index - 1]
            new_snapshot = snapshots[index]
            month_gap = month_index(new_snapshot.timestamp) - month_index(
                old_snapshot.timestamp
            )
            if month_gap <= 0:
                raise ValueError(
                    f"non-increasing timestamps in AOI {aoi_id}: "
                    f"{old_snapshot.timestamp} -> {new_snapshot.timestamp}"
                )
            if max_month_gap is not None and month_gap > max_month_gap:
                continue
            events = derive_edit_events(
                old_snapshot.map_frame,
                new_snapshot.map_frame,
                id_column="object_id",
                keep_iou_min=keep_iou_min,
                fallback_match_iou_min=fallback_match_iou_min,
                max_centroid_distance=max_centroid_distance,
            )
            events = filter_persistent_changes(
                events,
                _future_maps_in_sequence(
                    snapshots,
                    index,
                    max_month_gap=max_month_gap,
                ),
                min_change_persistence=min_change_persistence,
                presence_iou_min=fallback_match_iou_min,
            )
            yield SnapshotPair(
                aoi_id=str(aoi_id),
                split=new_snapshot.split,
                old_timestamp=old_snapshot.timestamp,
                new_timestamp=new_snapshot.timestamp,
                old_label_path=old_snapshot.label_path,
                new_label_path=new_snapshot.label_path,
                old_image_path=old_snapshot.image_path,
                image_path=new_snapshot.image_path,
                old_udm_path=old_snapshot.udm_path,
                udm_path=new_snapshot.udm_path,
                month_gap=month_gap,
                old_map=old_snapshot.map_frame,
                new_map=new_snapshot.map_frame,
                events=_limit_events(
                    events,
                    max_events_per_operation=max_events_per_operation,
                    sampling_key=(
                        f"{aoi_id}|{old_snapshot.timestamp}|{new_snapshot.timestamp}"
                    ),
                    sampling_seed=sampling_seed,
                ),
            )
