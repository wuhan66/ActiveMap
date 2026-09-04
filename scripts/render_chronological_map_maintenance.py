#!/usr/bin/env python3
"""Render separate mask PNGs for chronological maintenance chains."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw


def _rings(geometry: dict[str, Any] | None) -> Iterable[list[list[list[float]]]]:
    if geometry is None:
        return
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        yield coordinates
    elif geometry_type == "MultiPolygon":
        yield from coordinates
    else:
        raise ValueError(f"unsupported mask geometry type: {geometry_type}")


def _points(geometry: dict[str, Any] | None) -> Iterable[tuple[float, float]]:
    for polygon in _rings(geometry):
        for ring in polygon:
            for point in ring:
                yield float(point[0]), float(point[1])


def _bounds(rows: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    points = [
        point
        for row in rows
        for geometry in row["geometries"].values()
        for point in _points(geometry)
    ]
    if not points:
        return 0.0, 0.0, 1.0, 1.0
    xs, ys = zip(*points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    extent = max(max_x - min_x, max_y - min_y, 1e-6)
    margin = extent * 0.12
    return min_x - margin, min_y - margin, max_x + margin, max_y + margin


def render_mask(
    geometry: dict[str, Any] | None,
    bounds: tuple[float, float, float, float],
    *,
    size: int,
) -> Image.Image:
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    min_x, min_y, max_x, max_y = bounds

    def project(point: list[float]) -> tuple[float, float]:
        x = (float(point[0]) - min_x) / (max_x - min_x) * (size - 1)
        y = (max_y - float(point[1])) / (max_y - min_y) * (size - 1)
        return x, y

    for polygon in _rings(geometry):
        if not polygon:
            continue
        draw.polygon([project(point) for point in polygon[0]], fill=255)
        for hole in polygon[1:]:
            draw.polygon([project(point) for point in hole], fill=0)
    return image


def render(
    trace_path: Path,
    output_dir: Path,
    *,
    size: int,
    maximum_chains: int,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "geometries" not in row:
            raise ValueError("chronological trace lacks retained geometries")
        groups[str(row["chain_id"])].append(row)
    ranked = sorted(
        groups.items(),
        key=lambda item: (
            -max(abs(row["risk_gated_iou"] - row["carry_iou"]) for row in item[1]),
            -len(item[1]),
            item[0],
        ),
    )[:maximum_chains]
    if not ranked:
        raise ValueError("no chronological chains to render")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    exported = []
    for rank, (chain_id, chain_rows) in enumerate(ranked):
        chain_rows.sort(key=lambda row: (str(row["timestamp"]), int(row["step"])))
        bounds = _bounds(chain_rows)
        chain_dir = output_dir / f"{rank:02d}_{chain_id}"
        for row in chain_rows:
            step_dir = chain_dir / f"{int(row['step']):02d}_{row['timestamp']}"
            step_dir.mkdir(parents=True)
            for label in ("target", "independent", "carry", "risk_gated"):
                render_mask(row["geometries"][label], bounds, size=size).save(
                    step_dir / f"{label}.png"
                )
            (step_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "task_id",
                            "aoi_id",
                            "object_id",
                            "timestamp",
                            "target_edit",
                            "independent_iou",
                            "carry_iou",
                            "risk_gated_iou",
                            "risk_gate_intervened",
                        )
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        exported.append(
            {
                "chain_id": chain_id,
                "directory": str(chain_dir.resolve()),
                "steps": len(chain_rows),
                "aoi_id": chain_rows[0]["aoi_id"],
                "object_id": chain_rows[0]["object_id"],
            }
        )
    summary = {
        "schema_version": "activemap-chronological-mask-visuals-v1",
        "source": str(trace_path.resolve()),
        "image_size": size,
        "background": "black",
        "foreground": "white",
        "text_embedded_in_png": False,
        "base_image_included": False,
        "chains": exported,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--maximum-chains", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            render(
                args.trace,
                args.output_dir,
                size=args.size,
                maximum_chains=args.maximum_chains,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
