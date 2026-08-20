"""Derive typed object-level edits between two vector-map snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from activemap.geometry import (
    centroid_distance,
    geometry_iou,
    match_geometry_sets,
    repair_geometry,
)
from activemap.models import EditOperation, EditRecord, GeoJSONGeometry


@dataclass(frozen=True)
class EditEvent:
    op: EditOperation
    object_id: str
    old_geometry: BaseGeometry | None
    new_geometry: BaseGeometry | None
    iou: float | None
    match_source: str

    def to_edit_record(self) -> EditRecord:
        geometry = None
        if self.op in {EditOperation.ADD, EditOperation.RESHAPE}:
            if self.new_geometry is None:
                raise ValueError(f"{self.op.value} event is missing new geometry")
            geometry = GeoJSONGeometry.model_validate(mapping(self.new_geometry))
        return EditRecord(op=self.op, object_id=self.object_id, geometry=geometry)


def _object_ids(frame: gpd.GeoDataFrame, id_column: str) -> list[str]:
    if id_column not in frame.columns:
        return [f"row-{index}" for index in frame.index]
    return [str(value) for value in frame[id_column]]


def derive_edit_events(
    old_map: gpd.GeoDataFrame,
    new_map: gpd.GeoDataFrame,
    *,
    id_column: str = "object_id",
    keep_iou_min: float = 0.80,
    fallback_match_iou_min: float = 0.20,
    max_centroid_distance: float | None = None,
) -> list[EditEvent]:
    """Use persistent IDs first, then spatial matching for objects without correspondence."""
    if old_map.crs is not None and new_map.crs is not None and old_map.crs != new_map.crs:
        new_map = new_map.to_crs(old_map.crs)

    old_geometries = [repair_geometry(item) for item in old_map.geometry]
    new_geometries = [repair_geometry(item) for item in new_map.geometry]
    old_ids = _object_ids(old_map, id_column)
    new_ids = _object_ids(new_map, id_column)

    old_by_id = {value: index for index, value in enumerate(old_ids) if value and value != "None"}
    new_by_id = {value: index for index, value in enumerate(new_ids) if value and value != "None"}
    matched: list[tuple[int, int, str]] = []
    used_old: set[int] = set()
    used_new: set[int] = set()

    for object_id in sorted(set(old_by_id) & set(new_by_id)):
        old_index = old_by_id[object_id]
        new_index = new_by_id[object_id]
        if (
            max_centroid_distance is not None
            and centroid_distance(old_geometries[old_index], new_geometries[new_index])
            > max_centroid_distance
        ):
            continue
        matched.append((old_index, new_index, "persistent_id"))
        used_old.add(old_index)
        used_new.add(new_index)

    remaining_old = [index for index in range(len(old_geometries)) if index not in used_old]
    remaining_new = [index for index in range(len(new_geometries)) if index not in used_new]
    fallback_matches = match_geometry_sets(
        [old_geometries[index] for index in remaining_old],
        [new_geometries[index] for index in remaining_new],
        min_iou=fallback_match_iou_min,
        max_centroid_distance=max_centroid_distance,
    )
    for match in fallback_matches:
        old_index = remaining_old[match.old_index]
        new_index = remaining_new[match.new_index]
        matched.append((old_index, new_index, "spatial"))
        used_old.add(old_index)
        used_new.add(new_index)

    events: list[EditEvent] = []
    for old_index, new_index, source in matched:
        iou = geometry_iou(old_geometries[old_index], new_geometries[new_index])
        op = EditOperation.KEEP if iou >= keep_iou_min else EditOperation.RESHAPE
        object_id = new_ids[new_index] if new_ids[new_index] != "None" else old_ids[old_index]
        events.append(
            EditEvent(
                op=op,
                object_id=object_id,
                old_geometry=old_geometries[old_index],
                new_geometry=new_geometries[new_index],
                iou=iou,
                match_source=source,
            )
        )

    for old_index in sorted(set(range(len(old_geometries))) - used_old):
        events.append(
            EditEvent(
                op=EditOperation.DELETE,
                object_id=old_ids[old_index],
                old_geometry=old_geometries[old_index],
                new_geometry=None,
                iou=None,
                match_source="unmatched_old",
            )
        )
    for new_index in sorted(set(range(len(new_geometries))) - used_new):
        events.append(
            EditEvent(
                op=EditOperation.ADD,
                object_id=new_ids[new_index],
                old_geometry=None,
                new_geometry=new_geometries[new_index],
                iou=None,
                match_source="unmatched_new",
            )
        )

    operation_order = {operation: index for index, operation in enumerate(EditOperation)}
    return sorted(events, key=lambda event: (operation_order[event.op], event.object_id))


def edit_event_summary(events: list[EditEvent]) -> dict[str, Any]:
    counts = {operation.value: 0 for operation in EditOperation}
    for event in events:
        counts[event.op.value] += 1
    counts["total"] = len(events)
    return counts
