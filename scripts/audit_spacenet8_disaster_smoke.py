#!/usr/bin/env python3
"""Audit raster readability and geospatial pairing for SpaceNet8 episodes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

from activemap.data.disaster_map import DisasterMapEpisode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-bounds-iou", type=float, default=0.8)
    return parser.parse_args()


def bounds_iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def raster_record(path: Path) -> dict[str, object]:
    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise ValueError(f"missing CRS: {path}")
        geographic_bounds = transform_bounds(dataset.crs, "EPSG:4326", *dataset.bounds)
        sample = dataset.read(
            indexes=list(range(1, min(dataset.count, 3) + 1)),
            out_shape=(min(dataset.count, 3), 64, 64),
            masked=True,
        )
        return {
            "path": str(path),
            "width": dataset.width,
            "height": dataset.height,
            "bands": dataset.count,
            "dtype": dataset.dtypes[0],
            "crs": str(dataset.crs),
            "bounds_wgs84": list(geographic_bounds),
            "valid_fraction": float(1.0 - np.ma.getmaskarray(sample).mean()),
            "mean": float(sample.mean()),
            "std": float(sample.std()),
        }


def main() -> None:
    args = parse_args()
    episodes = [
        DisasterMapEpisode.model_validate_json(line)
        for line in args.episodes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors: list[str] = []
    low_overlap: list[dict[str, object]] = []
    image_shapes: Counter[str] = Counter()
    overlap_values: list[float] = []
    image_count = 0
    for episode in episodes:
        imagery = [row for row in episode.evidence if row.modality.endswith("_rgb")]
        pre = next(row for row in imagery if row.modality == "pre_event_rgb")
        try:
            pre_record = raster_record(Path(pre.path))
            image_count += 1
            image_shapes[
                f"{pre_record['width']}x{pre_record['height']}x{pre_record['bands']}"
            ] += 1
        except Exception as exc:
            errors.append(f"{episode.episode_id}:pre:{exc}")
            continue
        for post in (row for row in imagery if row.modality == "post_event_rgb"):
            try:
                post_record = raster_record(Path(post.path))
                image_count += 1
                image_shapes[
                    f"{post_record['width']}x{post_record['height']}x{post_record['bands']}"
                ] += 1
                overlap = bounds_iou(
                    tuple(pre_record["bounds_wgs84"]), tuple(post_record["bounds_wgs84"])
                )
                overlap_values.append(overlap)
                if overlap < args.min_bounds_iou:
                    low_overlap.append(
                        {
                            "episode_id": episode.episode_id,
                            "post_path": post.path,
                            "bounds_iou": overlap,
                        }
                    )
            except Exception as exc:
                errors.append(f"{episode.episode_id}:post:{exc}")

    summary = {
        "schema_version": "activemap-spacenet8-raster-audit-v1",
        "episodes": len(episodes),
        "images_read": image_count,
        "errors": errors,
        "image_shapes": dict(sorted(image_shapes.items())),
        "pre_post_pairs": len(overlap_values),
        "bounds_iou_min": min(overlap_values) if overlap_values else None,
        "bounds_iou_mean": float(np.mean(overlap_values)) if overlap_values else None,
        "low_overlap_pairs": low_overlap,
        "test_assets_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
