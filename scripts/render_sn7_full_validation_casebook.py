#!/usr/bin/env python3
"""Render every shared SN7 validation state as compact map-native caseboards.

This exporter never ranks cases.  Its output is a validation-only visual
casebook for later, documented qualitative selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from scripts.render_sn7_updater_qualitative import (
    Audit,
    _load_audit,
    _load_mask,
    _load_rgb,
    _prior_image_paths,
    _resize_binary,
    _resize_rgb,
    _zoom_box,
)
from scripts.train_sn7_changemamba import Record, _read_records
try:
    from scripts.visual_asset_utils import save_panel_layers
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from visual_asset_utils import save_panel_layers


AMBER = np.asarray((244, 162, 0), dtype=np.uint8)
GREEN = np.asarray((0, 158, 115), dtype=np.uint8)
BLUE = np.asarray((0, 114, 178), dtype=np.uint8)
RED = np.asarray((213, 94, 0), dtype=np.uint8)
CYAN = np.asarray((86, 180, 233), dtype=np.uint8)


def _parse_audit(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(raw_path)


def _outline(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    interior = mask.astype(bool).copy()
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        interior &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return mask.astype(bool) & ~interior


def _boundary_panel(rgb: np.ndarray, masks: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    panel = np.clip(rgb.astype(np.float32) * 0.68, 0, 255).astype(np.uint8)
    for mask, color in masks:
        panel[_outline(mask)] = color
    return panel


def _residual_panel(rgb: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    panel = np.clip(rgb.astype(np.float32) * 0.38 + 20.0, 0, 255).astype(np.uint8)
    for mask, color in (
        (prediction & target, GREEN),
        (prediction & ~target, RED),
        (target & ~prediction, CYAN),
    ):
        panel[_outline(mask)] = color
    return panel


def _tile(array: np.ndarray, label: str, size: int) -> Image.Image:
    image = Image.fromarray(array, mode="RGB").resize((size, size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size, 17), fill=(255, 255, 255))
    draw.text((4, 3), label, fill=(0, 0, 0))
    return image


def _caseboard(panels: list[tuple[str, np.ndarray]], tile_size: int) -> Image.Image:
    columns = 4
    rows = int(np.ceil(len(panels) / columns))
    board = Image.new("RGB", (columns * tile_size, rows * tile_size), "white")
    for index, (label, panel) in enumerate(panels):
        row, column = divmod(index, columns)
        board.paste(_tile(panel, label, tile_size), (column * tile_size, row * tile_size))
    return board


def _crop_panel(panel: np.ndarray, zoom: tuple[slice, slice]) -> np.ndarray:
    return panel[zoom]


def _shared_ids(records: dict[str, Record], audits: list[Audit]) -> list[str]:
    shared = set(records)
    for audit in audits:
        shared &= set(audit.rows) & set(audit.masks)
    return sorted(shared)


def render(
    manifest: Path,
    audit_paths: list[tuple[str, Path]],
    output_dir: Path,
    *,
    tile_size: int,
    asset_mode: str = "board",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if len(audit_paths) != 2:
        raise ValueError("exactly two audits are required for a paired casebook")
    if asset_mode not in {"board", "individual", "both"}:
        raise ValueError("asset_mode must be board, individual, or both")
    records = {record.sample_id: record for record in _read_records(manifest, "val", None)}
    audits = [_load_audit(name, path) for name, path in audit_paths]
    shared = _shared_ids(records, audits)
    if not shared:
        raise ValueError("no shared validation state")
    previous_images = _prior_image_paths(manifest, "val")
    cases_root = output_dir / "cases"
    cases_root.mkdir(parents=True)
    index_rows: list[dict[str, Any]] = []
    operation_counts: dict[str, int] = {}
    for index, sample_id in enumerate(shared, start=1):
        record = records[sample_id]
        operation_counts[record.edit_type] = operation_counts.get(record.edit_type, 0) + 1
        shape = audits[0].masks[sample_id].shape
        current = _resize_rgb(_load_rgb(record.image), shape)
        prior = _load_mask(record.prior, shape)
        target = _load_mask(record.target, shape)
        valid = _load_mask(record.valid, shape)
        previous = current
        if sample_id in previous_images:
            previous = _resize_rgb(_load_rgb(previous_images[sample_id]), shape)
        predictions = [
            _resize_binary(audit.masks[sample_id], shape) & valid for audit in audits
        ]
        committed = [np.logical_xor(prior, prediction) & valid for prediction in predictions]
        overview_panels: list[tuple[str, np.ndarray]] = [
            ("Previous image", previous),
            ("Current image", current),
            ("Editable prior", _boundary_panel(current, [(prior, AMBER)])),
            ("Reference map", _boundary_panel(current, [(target, GREEN)])),
        ]
        for audit, mask in zip(audits, committed, strict=True):
            overview_panels.extend(
                [
                    (
                        f"{audit.name} writeback",
                        _boundary_panel(current, [(target, GREEN), (mask, BLUE)]),
                    ),
                    (f"{audit.name} residual", _residual_panel(current, mask, target)),
                ]
            )
        comparison = np.logical_xor(committed[0], committed[1])
        truth_change = np.logical_xor(prior, target) & valid
        zoom = _zoom_box(
            [truth_change, comparison],
            fallback=prior | target | comparison,
            minimum_side=max(32, int(round(min(shape) * 0.25))),
        )
        case_dir = cases_root / record.edit_type.lower() / f"{index:05d}_{sample_id}"
        case_dir.mkdir(parents=True)
        assets = {}
        if asset_mode in {"board", "both"}:
            _caseboard(overview_panels, tile_size).save(case_dir / "overview.png", optimize=True)
            _caseboard(
                [(label, _crop_panel(panel, zoom)) for label, panel in overview_panels], tile_size
            ).save(case_dir / "zoom.png", optimize=True)
        if asset_mode in {"individual", "both"}:
            assets = save_panel_layers(case_dir, overview_panels, crop=zoom)
        method_rows = {audit.name: audit.rows[sample_id] for audit in audits}
        manifest_row = {
            "schema_version": "sn7-full-validation-casebook-v1",
            "dataset": "SpaceNet 7",
            "split": "val",
            "test_assets_read": False,
            "case_id": sample_id,
            "folder": str(case_dir.relative_to(output_dir)),
            "aoi_id": record.aoi_id,
            "target_edit": record.edit_type,
            "zoom_box_yxyx": [zoom[0].start, zoom[1].start, zoom[0].stop, zoom[1].stop],
            "methods": method_rows,
            "assets": assets,
            "paired": {
                "left": audits[0].name,
                "right": audits[1].name,
                "committed_map_iou_delta": float(
                    audits[0].rows[sample_id]["committed_map_iou"]
                    - audits[1].rows[sample_id]["committed_map_iou"]
                ),
                "change_iou_delta": float(
                    audits[0].rows[sample_id].get("change_iou", 0.0)
                    - audits[1].rows[sample_id].get("change_iou", 0.0)
                ),
            },
            "selection": "none; exhaustive shared validation export",
        }
        (case_dir / "manifest.json").write_text(
            json.dumps(manifest_row, indent=2) + "\n", encoding="utf-8"
        )
        index_rows.append(manifest_row)
    with (output_dir / "casebook_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "schema_version": "sn7-full-validation-casebook-v1",
        "dataset": "SpaceNet 7",
        "split": "val",
        "test_assets_read": False,
        "manifest": str(manifest),
        "audits": {name: str(path) for name, path in audit_paths},
        "case_count": len(index_rows),
        "operation_counts": operation_counts,
        "artifacts_per_case": (
            ["overview.png", "zoom.png", "manifest.json"]
            if asset_mode == "board"
            else ["layers/*.png", "crops/*.png", "manifest.json"]
            if asset_mode == "individual"
            else ["overview.png", "zoom.png", "layers/*.png", "crops/*.png", "manifest.json"]
        ),
        "asset_mode": asset_mode,
        "selection": "none; every shared validation state is rendered",
        "ordering": "target operation then sample id",
        "tile_size": tile_size,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--audit", action="append", type=_parse_audit, required=True)
    parser.add_argument("--tile-size", type=int, default=192)
    parser.add_argument("--asset-mode", choices=("board", "individual", "both"), default="board")
    args = parser.parse_args()
    if args.tile_size < 64:
        raise ValueError("tile-size must be at least 64")
    print(
        json.dumps(
            render(
                args.manifest,
                args.audit,
                args.output_dir,
                tile_size=args.tile_size,
                asset_mode=args.asset_mode,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
