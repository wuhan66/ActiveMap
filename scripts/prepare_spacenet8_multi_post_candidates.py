#!/usr/bin/env python3
"""Align every SpaceNet8 POST observation to the frozen spatial protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

from activemap.data.disaster_map import DisasterMapEpisode
from scripts.prepare_spacenet8_flood_smoke import bounds_iou, normalize_rgb


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("spatial_manifest", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)

    spatial = {str(row["sample_id"]): row for row in read_jsonl(args.spatial_manifest)}
    episodes = [
        DisasterMapEpisode.model_validate_json(line)
        for line in args.episodes.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = [episode for episode in episodes if episode.episode_id in spatial]
    args.output_root.mkdir(parents=True)
    records: list[dict] = []

    for episode in selected:
        spatial_row = spatial[episode.episode_id]
        base = np.load(str(spatial_row["array_path"]))
        target = base["target"].astype(np.uint8)
        pre_path = Path(next(row.path for row in episode.evidence if row.modality == "pre_event_rgb"))
        post_rows = [row for row in episode.evidence if row.modality == "post_event_rgb"]
        with rasterio.open(pre_path) as pre_source:
            pre_bounds = transform_bounds(pre_source.crs, "EPSG:4326", *pre_source.bounds)
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

        for candidate_index, evidence in enumerate(post_rows):
            post_path = Path(evidence.path)
            post = np.zeros((3, args.size, args.size), dtype=np.float32)
            post_valid = np.zeros((args.size, args.size), dtype=np.uint8)
            with rasterio.open(post_path) as post_source:
                post_bounds = transform_bounds(post_source.crs, "EPSG:4326", *post_source.bounds)
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
            valid = ((pre_valid > 0) & (post_valid > 0)).astype(np.uint8)
            tile_id = str(episode.metadata["tile_id"])
            artifact = args.output_root / "arrays" / f"{tile_id}__post{candidate_index}.npz"
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
                    "split": spatial_row["split"],
                    "candidate_id": evidence.evidence_id,
                    "candidate_index": candidate_index,
                    "candidate_count": len(post_rows),
                    "array_path": str(artifact),
                    "post_path": str(post_path),
                    "bounds_iou": bounds_iou(pre_bounds, post_bounds),
                    "valid_pixels": int(valid.sum()),
                    "positive_pixels": int((target & valid).sum()),
                    "cost": float(evidence.cost),
                    "test_assets_read": False,
                }
            )

    manifest = args.output_root / "manifest.jsonl"
    with manifest.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    counts = {split: sum(row["split"] == split for row in records) for split in ("train", "val")}
    summary = {
        "schema_version": "activemap-spacenet8-multi-post-v1",
        "episodes": len(selected),
        "candidates": len(records),
        "ambiguous_episodes": sum(
            sum(row.modality == "post_event_rgb" for row in episode.evidence) > 1
            for episode in selected
        ),
        "candidate_split_counts": counts,
        "size": args.size,
        "test_assets_read": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
