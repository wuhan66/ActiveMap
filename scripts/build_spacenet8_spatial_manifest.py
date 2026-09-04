#!/usr/bin/env python3
"""Create a contiguous, buffer-separated SpaceNet8 train/validation split."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Tile:
    record: dict
    zoom: int
    x: int
    y: int
    positive: bool
    ambiguous: bool


def _tile(record: dict) -> Tile:
    tile_id = str(record["sample_id"]).rsplit(":", 1)[-1]
    try:
        zoom, x, y = (int(value) for value in tile_id.split("_"))
    except ValueError as exc:
        raise ValueError(f"Unsupported SpaceNet8 tile id: {tile_id}") from exc
    return Tile(
        record,
        zoom,
        x,
        y,
        int(record.get("positive_pixels", 0)) > 0,
        int(record.get("post_candidates", 1)) > 1,
    )


def choose_rectangle(
    tiles: list[Tile], val_fraction: float, block_size: int, rectangle_rank: int = 1
) -> tuple[int, int, int, int, int]:
    zooms = {tile.zoom for tile in tiles}
    if len(zooms) != 1:
        raise ValueError(f"Expected one zoom level, found {sorted(zooms)}")
    zoom = next(iter(zooms))
    blocks = {(tile.x // block_size, tile.y // block_size) for tile in tiles}
    xs = range(min(x for x, _ in blocks), max(x for x, _ in blocks) + 1)
    ys = range(min(y for _, y in blocks), max(y for _, y in blocks) + 1)
    target_count = max(1, round(len(tiles) * val_fraction))
    target_positive = max(1, round(sum(tile.positive for tile in tiles) * val_fraction))
    target_ambiguous = max(1, round(sum(tile.ambiguous for tile in tiles) * val_fraction))
    candidates: list[tuple[float, tuple[int, int, int, int, int]]] = []
    for x0 in xs:
        for x1 in range(x0, xs.stop):
            for y0 in ys:
                for y1 in range(y0, ys.stop):
                    selected = [
                        tile
                        for tile in tiles
                        if x0 <= tile.x // block_size <= x1
                        and y0 <= tile.y // block_size <= y1
                    ]
                    positives = sum(tile.positive for tile in selected)
                    ambiguous = sum(tile.ambiguous for tile in selected)
                    if (
                        not selected
                        or positives == 0
                        or positives == len(selected)
                        or ambiguous == 0
                    ):
                        continue
                    occupied = {
                        (tile.x // block_size, tile.y // block_size) for tile in selected
                    }
                    rectangle_area = (x1 - x0 + 1) * (y1 - y0 + 1)
                    score = (
                        abs(len(selected) - target_count) / target_count
                        + abs(positives - target_positive) / target_positive
                        + abs(ambiguous - target_ambiguous) / target_ambiguous
                        + 0.02 * (rectangle_area - len(occupied))
                    )
                    candidate = (zoom, x0, x1, y0, y1)
                    candidates.append((score, candidate))
    if not candidates:
        raise RuntimeError("Could not find a mixed positive/negative spatial rectangle")
    candidates.sort()
    if rectangle_rank <= 0 or rectangle_rank > len(candidates):
        raise ValueError(
            f"rectangle_rank must be in [1, {len(candidates)}], got {rectangle_rank}"
        )
    return candidates[rectangle_rank - 1][1]


def spatial_split(
    records: list[dict],
    val_fraction: float,
    block_size: int,
    buffer_tiles: int,
    rectangle_rank: int = 1,
) -> tuple[list[dict], dict]:
    tiles = [_tile(record) for record in records]
    zoom, x0, x1, y0, y1 = choose_rectangle(
        tiles, val_fraction, block_size, rectangle_rank
    )
    validation = [
        tile
        for tile in tiles
        if tile.zoom == zoom
        and x0 <= tile.x // block_size <= x1
        and y0 <= tile.y // block_size <= y1
    ]
    val_coordinates = {(tile.x, tile.y) for tile in validation}
    training: list[Tile] = []
    excluded: list[Tile] = []
    for tile in tiles:
        if tile in validation:
            continue
        distance = min(
            max(abs(tile.x - val_x), abs(tile.y - val_y))
            for val_x, val_y in val_coordinates
        )
        (excluded if distance <= buffer_tiles else training).append(tile)
    if not training or not validation:
        raise RuntimeError("Spatial split produced an empty train or validation partition")
    min_distance = min(
        max(abs(train.x - val.x), abs(train.y - val.y))
        for train in training
        for val in validation
    )
    output = []
    for split, partition in (("train", training), ("val", validation)):
        for tile in partition:
            output.append({**tile.record, "split": split})
    output.sort(key=lambda row: (row["split"], str(row["sample_id"])))
    summary = {
        "schema_version": "activemap-spacenet8-spatial-split-v1",
        "source_records": len(records),
        "train_records": len(training),
        "val_records": len(validation),
        "excluded_buffer_records": len(excluded),
        "train_positive_records": sum(tile.positive for tile in training),
        "val_positive_records": sum(tile.positive for tile in validation),
        "train_ambiguous_records": sum(tile.ambiguous for tile in training),
        "val_ambiguous_records": sum(tile.ambiguous for tile in validation),
        "block_size": block_size,
        "buffer_tiles": buffer_tiles,
        "rectangle_rank": rectangle_rank,
        "minimum_train_val_chebyshev_distance": min_distance,
        "validation_block_rectangle": {
            "zoom": zoom,
            "x_min": x0,
            "x_max": x1,
            "y_min": y0,
            "y_max": y1,
        },
        "test_assets_read": False,
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--buffer-tiles", type=int, default=1)
    parser.add_argument("--rectangle-rank", type=int, default=1)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("val-fraction must be between zero and one")
    if args.block_size <= 0 or args.buffer_tiles < 0:
        raise ValueError("block-size must be positive and buffer-tiles non-negative")
    records = [
        json.loads(line)
        for line in args.source_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output, summary = spatial_split(
        records,
        args.val_fraction,
        args.block_size,
        args.buffer_tiles,
        args.rectangle_rank,
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    with (args.output_root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in output:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
