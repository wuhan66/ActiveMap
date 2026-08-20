"""Clean-room conversion from one-pixel road skeletons to MUNO21 graph files."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from affine import Affine
from shapely.geometry import LineString

Pixel = tuple[int, int]


def _neighbors(point: Pixel, pixels: set[Pixel]) -> list[Pixel]:
    row, column = point
    return sorted(
        (row + dr, column + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr or dc) and (row + dr, column + dc) in pixels
    )


def _components(pixels: set[Pixel]) -> list[set[Pixel]]:
    remaining = set(pixels)
    output = []
    while remaining:
        seed = min(remaining)
        component = {seed}
        queue = deque([seed])
        remaining.remove(seed)
        while queue:
            point = queue.popleft()
            for neighbor in _neighbors(point, remaining):
                remaining.remove(neighbor)
                component.add(neighbor)
                queue.append(neighbor)
        output.append(component)
    return output


def trace_skeleton(skeleton: np.ndarray) -> list[list[Pixel]]:
    """Trace every undirected skeleton edge exactly once into pixel chains."""

    values = np.asarray(skeleton)
    if values.ndim != 2:
        raise ValueError("skeleton must be a two-dimensional array")
    pixels = {tuple(map(int, point)) for point in np.argwhere(values > 0)}
    if not pixels:
        return []
    nodes = {point for point in pixels if len(_neighbors(point, pixels)) != 2}
    for component in _components(pixels):
        if not (component & nodes):
            nodes.add(min(component))

    visited: set[tuple[Pixel, Pixel]] = set()

    def edge(left: Pixel, right: Pixel) -> tuple[Pixel, Pixel]:
        return (left, right) if left < right else (right, left)

    paths = []
    for start in sorted(nodes):
        for neighbor in _neighbors(start, pixels):
            first_edge = edge(start, neighbor)
            if first_edge in visited:
                continue
            visited.add(first_edge)
            path = [start, neighbor]
            previous, current = start, neighbor
            while current not in nodes:
                candidates = [item for item in _neighbors(current, pixels) if item != previous]
                if not candidates:
                    break
                following = candidates[0]
                current_edge = edge(current, following)
                if current_edge in visited:
                    break
                visited.add(current_edge)
                path.append(following)
                previous, current = current, following
            if len(path) >= 2:
                paths.append(path)
    return paths


class MunoRoadGraph:
    def __init__(self, *, coordinate_precision: int = 3) -> None:
        self.coordinate_precision = coordinate_precision
        self.vertices: list[tuple[float, float]] = []
        self._vertex_ids: dict[tuple[float, float], int] = {}
        self.edges: set[tuple[int, int]] = set()

    def _vertex(self, point: tuple[float, float]) -> int:
        key = tuple(round(float(value), self.coordinate_precision) for value in point)
        if key not in self._vertex_ids:
            self._vertex_ids[key] = len(self.vertices)
            self.vertices.append(key)
        return self._vertex_ids[key]

    def add_polyline(self, coordinates: list[tuple[float, float]]) -> None:
        ids = [self._vertex(point) for point in coordinates]
        for left, right in zip(ids, ids[1:], strict=False):
            if left != right:
                self.edges.add((left, right))
                self.edges.add((right, left))

    def add_skeleton(
        self,
        skeleton: np.ndarray,
        transform: Affine,
        *,
        simplify_tolerance: float = 1.0,
    ) -> None:
        for path in trace_skeleton(skeleton):
            pixel_coordinates = [(float(column) + 0.5, float(row) + 0.5) for row, column in path]
            line = LineString(pixel_coordinates)
            if simplify_tolerance > 0:
                line = line.simplify(simplify_tolerance, preserve_topology=False)
            coordinates = [transform * tuple(point) for point in line.coords]
            self.add_polyline(coordinates)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for x, y in self.vertices:
                handle.write(f"{x} {y}\n")
            handle.write("\n")
            for left, right in sorted(self.edges):
                handle.write(f"{left} {right}\n")
