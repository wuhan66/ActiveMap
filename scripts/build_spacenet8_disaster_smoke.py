#!/usr/bin/env python3
"""Build an auditable SpaceNet8 disaster-map episode set from public training data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from activemap.data.disaster_map import (
    DisasterEvidence,
    DisasterMapEpisode,
    deterministic_disaster_split,
    match_spacenet8_assets,
    spacenet8_target_counts,
    validate_disaster_map_jsonl,
    write_disaster_map_jsonl,
    write_spacenet8_prior,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Episode limit; use zero to retain every matched public-training tile.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--selection", choices=("sorted", "flood-balanced"), default="flood-balanced"
    )
    parser.add_argument("--split-mode", choices=("hash", "flood-stratified"), default="hash")
    parser.add_argument("--region", default="germany")
    parser.add_argument("--aoi-id", default=None)
    parser.add_argument(
        "--fixed-split",
        choices=("train", "val"),
        default=None,
        help="Assign every exported tile to one split, for external validation-only use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)

    all_rows = match_spacenet8_assets(args.dataset_root)
    target_counts = {
        str(row["tile_id"]): spacenet8_target_counts(Path(row["annotation"])) for row in all_rows
    }
    if args.limit == 0:
        rows = all_rows
    elif args.selection == "flood-balanced":
        positive = [row for row in all_rows if target_counts[str(row["tile_id"])]["flooded"] > 0]
        negative = [row for row in all_rows if target_counts[str(row["tile_id"])]["flooded"] == 0]
        positive_limit = min((args.limit + 1) // 2, len(positive))
        rows = positive[:positive_limit] + negative[: args.limit - positive_limit]
    else:
        rows = all_rows[: args.limit]
    if args.fixed_split is not None:
        split_by_tile = {str(row["tile_id"]): args.fixed_split for row in rows}
    elif args.split_mode == "flood-stratified":
        split_by_tile: dict[str, str] = {}
        for positive_status in (False, True):
            group = [
                row
                for row in rows
                if (target_counts[str(row["tile_id"])]["flooded"] > 0) == positive_status
            ]
            group.sort(
                key=lambda row: hashlib.sha256(str(row["tile_id"]).encode("ascii")).hexdigest()
            )
            val_count = max(1, round(len(group) * args.val_fraction)) if group else 0
            for index, row in enumerate(group):
                split_by_tile[str(row["tile_id"])] = "val" if index < val_count else "train"
    else:
        split_by_tile = {
            str(row["tile_id"]): deterministic_disaster_split(
                str(row["tile_id"]), args.val_fraction
            )
            for row in rows
        }
    episodes: list[DisasterMapEpisode] = []
    totals = {"features": 0, "roads": 0, "buildings": 0, "flooded": 0}
    ambiguous_post_tiles = 0
    for row in rows:
        tile_id = str(row["tile_id"])
        target = Path(row["annotation"])
        pre = Path(row["pre"])
        post = [Path(path) for path in row["post"]]
        ambiguous_post_tiles += int(len(post) > 1)
        prior = args.output_root / "priors" / f"{tile_id}.geojson"
        counts = write_spacenet8_prior(target, prior)
        for key, value in counts.items():
            totals[key] += value

        evidence = [
            DisasterEvidence(
                evidence_id=f"{tile_id}:foundation",
                modality="foundation_map",
                path=str(prior),
                cost=0.05,
                timestamp=0,
            ),
            DisasterEvidence(
                evidence_id=f"{tile_id}:pre",
                modality="pre_event_rgb",
                path=str(pre),
                cost=0.25,
                timestamp=0,
            ),
        ]
        evidence.extend(
            DisasterEvidence(
                evidence_id=f"{tile_id}:post:{index}",
                modality="post_event_rgb",
                path=str(path),
                cost=1.0,
                timestamp=1,
            )
            for index, path in enumerate(post)
        )
        episodes.append(
            DisasterMapEpisode(
                schema_version="activemap-disaster-map-episode-v1",
                episode_id=f"spacenet8:{args.region}:{tile_id}",
                dataset="spacenet8",
                split=split_by_tile[tile_id],
                aoi_id=args.aoi_id or args.region,
                prior_map_path=str(prior),
                target_map_path=str(target),
                evidence=evidence,
                target_layers=[
                    "building",
                    "road",
                    "flooded_building",
                    "flooded_road",
                    "road_speed",
                ],
                budget=1.5,
                metadata={
                    "tile_id": tile_id,
                    "post_observations": len(post),
                    **counts,
                },
            )
        )

    manifest = args.output_root / "episodes.jsonl"
    write_disaster_map_jsonl(episodes, manifest)
    count, errors = validate_disaster_map_jsonl(manifest, check_paths=True)
    if errors:
        raise RuntimeError("\n".join(errors))
    split_counts = {
        split: sum(episode.split == split for episode in episodes) for split in ("train", "val")
    }
    summary = {
        "schema_version": "activemap-spacenet8-smoke-v1",
        "episodes": count,
        "split_counts": split_counts,
        "ambiguous_post_tiles": ambiguous_post_tiles,
        "positive_episodes": sum(int(episode.metadata["flooded"]) > 0 for episode in episodes),
        "region": args.region,
        "aoi_id": args.aoi_id or args.region,
        "fixed_split": args.fixed_split,
        "selection": args.selection,
        "split_mode": args.split_mode,
        "positive_split_counts": {
            split: sum(
                episode.split == split and int(episode.metadata["flooded"]) > 0
                for episode in episodes
            )
            for split in ("train", "val")
        },
        "totals": totals,
        "test_assets_read": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
