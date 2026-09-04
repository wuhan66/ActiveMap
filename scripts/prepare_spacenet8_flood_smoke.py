#!/usr/bin/env python3
"""Align SpaceNet8 imagery and rasterize flood labels for a small GPU smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject, transform_bounds, transform_geom

from activemap.data.disaster_map import DisasterMapEpisode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--size", type=int, default=256)
    return parser.parse_args()


def bounds_iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    areas = (left[2] - left[0]) * (left[3] - left[1])
    areas += (right[2] - right[0]) * (right[3] - right[1])
    union = areas - intersection
    return intersection / union if union > 0.0 else 0.0


def normalize_rgb(array: np.ndarray) -> np.ndarray:
    array = array[:3].astype(np.float32)
    upper = float(np.percentile(array[array > 0], 99.5)) if np.any(array > 0) else 1.0
    scale = max(upper, 1.0)
    return np.clip(array / scale, 0.0, 1.0).astype(np.float16)


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    records: list[dict[str, object]] = []
    episodes = [
        DisasterMapEpisode.model_validate_json(line)
        for line in args.episodes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for episode in episodes:
        pre_path = Path(
            next(row.path for row in episode.evidence if row.modality == "pre_event_rgb")
        )
        post_paths = [
            Path(row.path) for row in episode.evidence if row.modality == "post_event_rgb"
        ]
        with rasterio.open(pre_path) as pre_source:
            if pre_source.crs is None:
                raise ValueError(f"missing PRE CRS: {pre_path}")
            pre_bounds = transform_bounds(pre_source.crs, "EPSG:4326", *pre_source.bounds)
            candidates: list[tuple[float, Path]] = []
            for post_path in post_paths:
                with rasterio.open(post_path) as post_source:
                    if post_source.crs is None:
                        raise ValueError(f"missing POST CRS: {post_path}")
                    post_bounds = transform_bounds(
                        post_source.crs, "EPSG:4326", *post_source.bounds
                    )
                candidates.append((bounds_iou(pre_bounds, post_bounds), post_path))
            selected_overlap, selected_post = max(candidates, key=lambda row: (row[0], row[1].name))
            destination_transform = pre_source.transform * Affine.scale(
                pre_source.width / args.size, pre_source.height / args.size
            )
            pre = pre_source.read(
                indexes=(1, 2, 3),
                out_shape=(3, args.size, args.size),
                resampling=Resampling.bilinear,
            )
            pre_valid = pre_source.dataset_mask(
                out_shape=(args.size, args.size), resampling=Resampling.nearest
            )
            destination_crs = pre_source.crs

        post = np.zeros((3, args.size, args.size), dtype=np.float32)
        post_valid = np.zeros((args.size, args.size), dtype=np.uint8)
        with rasterio.open(selected_post) as post_source:
            for band in range(3):
                reproject(
                    source=rasterio.band(post_source, band + 1),
                    destination=post[band],
                    src_transform=post_source.transform,
                    src_crs=post_source.crs,
                    dst_transform=destination_transform,
                    dst_crs=destination_crs,
                    resampling=Resampling.bilinear,
                )
            reproject(
                source=post_source.dataset_mask(),
                destination=post_valid,
                src_transform=post_source.transform,
                src_crs=post_source.crs,
                dst_transform=destination_transform,
                dst_crs=destination_crs,
                resampling=Resampling.nearest,
            )

        target_payload = json.loads(Path(episode.target_map_path).read_text(encoding="utf-8"))
        flooded_geometries = []
        for feature in target_payload.get("features", []):
            flooded = feature.get("properties", {}).get("flooded")
            if not isinstance(flooded, str) or flooded.lower() not in {"yes", "true", "1"}:
                continue
            flooded_geometries.append(
                transform_geom("EPSG:4326", destination_crs, feature["geometry"])
            )
        target = rasterize(
            ((geometry, 1) for geometry in flooded_geometries),
            out_shape=(args.size, args.size),
            transform=destination_transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )
        valid = ((pre_valid > 0) & (post_valid > 0)).astype(np.uint8)
        artifact = args.output_root / "arrays" / f"{episode.metadata['tile_id']}.npz"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact,
            pre=normalize_rgb(pre),
            post=normalize_rgb(post),
            target=target,
            valid=valid,
        )
        records.append(
            {
                "sample_id": episode.episode_id,
                "aoi_id": episode.aoi_id,
                "split": episode.split,
                "array_path": str(artifact),
                "positive_pixels": int((target & valid).sum()),
                "valid_pixels": int(valid.sum()),
                "post_candidates": len(post_paths),
                "selected_post": str(selected_post),
                "selected_bounds_iou": selected_overlap,
                "test_assets_read": False,
            }
        )

    manifest = args.output_root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "activemap-spacenet8-aligned-flood-smoke-v1",
        "records": len(records),
        "train": sum(record["split"] == "train" for record in records),
        "val": sum(record["split"] == "val" for record in records),
        "positive_pixels": sum(int(record["positive_pixels"]) for record in records),
        "minimum_selected_bounds_iou": min(
            float(record["selected_bounds_iou"]) for record in records
        ),
        "size": args.size,
        "test_assets_read": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
