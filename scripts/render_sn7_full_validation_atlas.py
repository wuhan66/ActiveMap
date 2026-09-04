#!/usr/bin/env python3
"""Render every audited SN7 validation sample as compact paired residual tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts.render_sn7_updater_qualitative import (
    _load_audit,
    _load_mask,
    _load_rgb,
    _residual,
    _resize_binary,
    _resize_rgb,
)
from scripts.train_sn7_changemamba import _read_records


OPERATIONS = ("KEEP", "ADD", "DELETE", "RESHAPE")


def render_atlas(
    manifest: Path,
    primary_name: str,
    primary_dir: Path,
    baseline_name: str,
    baseline_dir: Path,
    output_dir: Path,
    *,
    columns: int,
    rows: int,
    tile_size: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    primary = _load_audit(primary_name, primary_dir)
    baseline = _load_audit(baseline_name, baseline_dir)
    records = {record.sample_id: record for record in _read_records(manifest, "val", None)}
    shared = sorted(set(records) & set(primary.rows) & set(baseline.rows))
    if not shared:
        raise ValueError("the manifest and audits share no validation samples")

    cells_per_page = columns * rows
    index_rows: list[dict[str, Any]] = []
    page_counts: dict[str, int] = {}
    for operation in OPERATIONS:
        sample_ids = [sample_id for sample_id in shared if records[sample_id].edit_type == operation]
        sample_ids.sort(
            key=lambda sample_id: (
                max(
                    float(primary.rows[sample_id].get("target_change_fraction", 0.0)),
                    float(primary.rows[sample_id].get("predicted_change_fraction", 0.0)),
                    float(baseline.rows[sample_id].get("predicted_change_fraction", 0.0)),
                ),
                float(primary.rows[sample_id]["committed_map_iou"])
                - float(baseline.rows[sample_id]["committed_map_iou"]),
                sample_id,
            ),
            reverse=True,
        )
        operation_dir = output_dir / operation.lower()
        operation_dir.mkdir()
        page_counts[operation] = int(np.ceil(len(sample_ids) / cells_per_page))
        for page_index, start in enumerate(range(0, len(sample_ids), cells_per_page), 1):
            page_ids = sample_ids[start : start + cells_per_page]
            canvas = Image.new(
                "RGB",
                (columns * tile_size * 2, rows * tile_size),
                "white",
            )
            for local_index, sample_id in enumerate(page_ids):
                record = records[sample_id]
                target_shape = primary.masks[sample_id].shape
                rgb = _resize_rgb(_load_rgb(record.image), target_shape)
                prior = _load_mask(record.prior, target_shape)
                target = _load_mask(record.target, target_shape)
                valid = _load_mask(record.valid, target_shape)
                truth_change = np.logical_xor(prior, target) & valid
                primary_panel = _residual(
                    rgb,
                    truth_change,
                    _resize_binary(primary.masks[sample_id], target_shape) & valid,
                    valid,
                ).resize((tile_size, tile_size), Image.Resampling.NEAREST)
                baseline_panel = _residual(
                    rgb,
                    truth_change,
                    _resize_binary(baseline.masks[sample_id], target_shape) & valid,
                    valid,
                ).resize((tile_size, tile_size), Image.Resampling.NEAREST)
                row, column = divmod(local_index, columns)
                left = column * tile_size * 2
                top = row * tile_size
                canvas.paste(primary_panel, (left, top))
                canvas.paste(baseline_panel, (left + tile_size, top))
                index_rows.append(
                    {
                        "operation": operation,
                        "page": page_index,
                        "cell": local_index,
                        "row": row,
                        "column": column,
                        "sample_id": sample_id,
                        "left_method": primary_name,
                        "right_method": baseline_name,
                        "primary_map_iou": primary.rows[sample_id]["committed_map_iou"],
                        "baseline_map_iou": baseline.rows[sample_id]["committed_map_iou"],
                        "paired_map_iou_delta": float(
                            primary.rows[sample_id]["committed_map_iou"]
                            - baseline.rows[sample_id]["committed_map_iou"]
                        ),
                        "target_change_fraction": primary.rows[sample_id].get(
                            "target_change_fraction"
                        ),
                    }
                )
            canvas.save(operation_dir / f"page_{page_index:03d}.png", optimize=True)

    with (output_dir / "atlas_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "schema_version": "sn7-full-validation-residual-atlas-v1",
        "split": "val",
        "sample_count": len(shared),
        "operation_counts": {
            operation: sum(records[sample_id].edit_type == operation for sample_id in shared)
            for operation in OPERATIONS
        },
        "page_counts": page_counts,
        "layout": {
            "columns": columns,
            "rows": rows,
            "tile_size": tile_size,
            "left": primary_name,
            "right": baseline_name,
            "tile_encoding": "TP green, FP orange-red, FN light blue on dimmed current RGB",
        },
        "selection": "none; every shared validation sample is rendered",
        "ordering": "visual-change fraction, paired primary-minus-baseline map IoU, sample id",
        "test_assets_read": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("primary_name")
    parser.add_argument("primary_dir", type=Path)
    parser.add_argument("baseline_name")
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=96)
    args = parser.parse_args()
    if min(args.columns, args.rows, args.tile_size) < 1:
        raise ValueError("atlas layout values must be positive")
    print(
        json.dumps(
            render_atlas(
                args.manifest,
                args.primary_name,
                args.primary_dir,
                args.baseline_name,
                args.baseline_dir,
                args.output_dir,
                columns=args.columns,
                rows=args.rows,
                tile_size=args.tile_size,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
