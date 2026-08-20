"""Geometry repair, matching, and numeric features for vector-map objects."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely import STRtree, make_valid
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class GeometryMatch:
    old_index: int
    new_index: int
    iou: float
    centroid_distance: float


def repair_geometry(geometry: BaseGeometry) -> BaseGeometry:
    if geometry.is_empty:
        return geometry
    repaired = make_valid(geometry)
    if repaired.geom_type == "GeometryCollection":
        polygonal = [item for item in repaired.geoms if "Polygon" in item.geom_type]
        if polygonal:
            repaired = max(polygonal, key=lambda item: item.area)
    return repaired


def geometry_iou(left: BaseGeometry, right: BaseGeometry) -> float:
    left = repair_geometry(left)
    right = repair_geometry(right)
    if left.is_empty or right.is_empty:
        return 0.0
    union_area = left.union(right).area
    if union_area <= 0:
        return 0.0
    return float(left.intersection(right).area / union_area)


def centroid_distance(left: BaseGeometry, right: BaseGeometry) -> float:
    return float(left.centroid.distance(right.centroid))


def geometry_features(
    geometry: BaseGeometry,
    *,
    reference_bounds: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Return stable, scale-aware features without serializing every vertex."""
    geometry = repair_geometry(geometry)
    min_x, min_y, max_x, max_y = geometry.bounds
    centroid = geometry.centroid
    width = max_x - min_x
    height = max_y - min_y
    scale = 1.0
    origin_x = 0.0
    origin_y = 0.0
    if reference_bounds is not None:
        ref_min_x, ref_min_y, ref_max_x, ref_max_y = reference_bounds
        scale = max(ref_max_x - ref_min_x, ref_max_y - ref_min_y, 1e-6)
        origin_x = ref_min_x
        origin_y = ref_min_y
    compactness = 0.0
    if geometry.length > 0:
        compactness = float(4.0 * np.pi * geometry.area / (geometry.length**2))
    return np.asarray(
        [
            (centroid.x - origin_x) / scale,
            (centroid.y - origin_y) / scale,
            width / scale,
            height / scale,
            geometry.area / (scale**2),
            geometry.length / scale,
            compactness,
            float(geometry.is_valid),
        ],
        dtype=np.float32,
    )


def match_geometry_sets(
    old_geometries: Sequence[BaseGeometry],
    new_geometries: Sequence[BaseGeometry],
    *,
    min_iou: float = 0.20,
    max_centroid_distance: float | None = None,
) -> list[GeometryMatch]:
    if not old_geometries or not new_geometries:
        return []

    repaired_old = [repair_geometry(geometry) for geometry in old_geometries]
    repaired_new = [repair_geometry(geometry) for geometry in new_geometries]
    tree = STRtree(repaired_new)
    edge_values: dict[tuple[int, int], tuple[float, float, float]] = {}
    old_neighbors: defaultdict[int, set[int]] = defaultdict(set)
    new_neighbors: defaultdict[int, set[int]] = defaultdict(set)

    for old_index, old_geometry in enumerate(repaired_old):
        candidate_indices: Iterable[int]
        if min_iou > 0:
            candidate_indices = tree.query(old_geometry, predicate="intersects")
        elif max_centroid_distance is not None:
            candidate_indices = tree.query(
                old_geometry.buffer(max_centroid_distance), predicate="intersects"
            )
        else:
            candidate_indices = range(len(repaired_new))
        for raw_new_index in candidate_indices:
            new_index = int(raw_new_index)
            new_geometry = repaired_new[new_index]
            iou = geometry_iou(old_geometry, new_geometry)
            distance = centroid_distance(old_geometry, new_geometry)
            if iou < min_iou:
                continue
            if max_centroid_distance is not None and distance > max_centroid_distance:
                continue
            distance_penalty = (
                min(distance / max_centroid_distance, 1.0)
                if max_centroid_distance
                else 0.0
            )
            cost = 1.0 - iou + 0.2 * distance_penalty
            edge_values[(old_index, new_index)] = (cost, iou, distance)
            old_neighbors[old_index].add(new_index)
            new_neighbors[new_index].add(old_index)

    matches: list[GeometryMatch] = []
    unseen_old = set(old_neighbors)
    while unseen_old:
        pending = [min(unseen_old)]
        component_old: set[int] = set()
        component_new: set[int] = set()
        while pending:
            old_index = pending.pop()
            if old_index in component_old:
                continue
            component_old.add(old_index)
            unseen_old.discard(old_index)
            for new_index in old_neighbors[old_index]:
                if new_index in component_new:
                    continue
                component_new.add(new_index)
                pending.extend(new_neighbors[new_index] - component_old)

        old_indices = sorted(component_old)
        new_indices = sorted(component_new)
        old_positions = {value: index for index, value in enumerate(old_indices)}
        new_positions = {value: index for index, value in enumerate(new_indices)}
        costs = np.full((len(old_indices), len(new_indices)), 10.0, dtype=np.float64)
        for old_index in old_indices:
            for new_index in old_neighbors[old_index] & component_new:
                costs[old_positions[old_index], new_positions[new_index]] = edge_values[
                    (old_index, new_index)
                ][0]
        assigned_old, assigned_new = linear_sum_assignment(costs)
        for old_position, new_position in zip(assigned_old, assigned_new, strict=True):
            old_index = old_indices[int(old_position)]
            new_index = new_indices[int(new_position)]
            values = edge_values.get((old_index, new_index))
            if values is None:
                continue
            _, iou, distance = values
            matches.append(GeometryMatch(old_index, new_index, iou, distance))
    return sorted(matches, key=lambda match: (match.old_index, match.new_index))
