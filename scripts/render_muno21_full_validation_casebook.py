#!/usr/bin/env python3
"""Render every paired MUNO21 validation writeback as a compact caseboard.

Selection is deliberately absent from this script.  Direct and ActiveMap
receipts are only paired by task and budget, then emitted in stable order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from scripts.render_muno21_map_native_qualitative import (
    BLUE,
    GREEN,
    ORANGE,
    _draw_boundary,
    _expand_bounds,
    _index_writebacks,
    _load_mask,
    _load_rgb,
    _raw_id,
    _residual,
)
try:
    from scripts.visual_asset_utils import save_panel_layers
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from visual_asset_utils import save_panel_layers


def _operation(row: dict[str, Any]) -> str:
    for key in ("target_edit", "edit_type", "operation", "requested_edit"):
        value = row.get(key)
        if value:
            return str(value).upper()
    return "UNSPECIFIED"


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


def render(
    direct_path: Path,
    active_path: Path,
    arrays_dir: Path,
    output_dir: Path,
    *,
    tile_size: int,
    zoom_size: int,
    asset_mode: str = "board",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if asset_mode not in {"board", "individual", "both"}:
        raise ValueError("asset_mode must be board, individual, or both")
    direct = _index_writebacks(direct_path)
    active = _index_writebacks(active_path)
    direct_keys, active_keys = set(direct), set(active)
    shared = sorted(direct_keys & active_keys, key=lambda key: (key[1], key[0]))
    if not shared:
        raise ValueError("no paired validation writebacks")
    cases_root = output_dir / "cases"
    cases_root.mkdir(parents=True)
    index_rows: list[dict[str, Any]] = []
    groups: dict[str, int] = {}
    for index, key in enumerate(shared, start=1):
        direct_row, active_row = direct[key], active[key]
        source_id = _raw_id(active_row)
        image_path = arrays_dir / f"{source_id}-image.npy"
        prior_path = arrays_dir / f"{source_id}-prior.npy"
        target_path = arrays_dir / f"{source_id}-target.npy"
        if not all(path.is_file() for path in (image_path, prior_path, target_path)):
            raise FileNotFoundError(f"missing processed arrays for {source_id}")
        rgb = _load_rgb(image_path)
        prior = np.asarray(np.load(prior_path), dtype=np.float32) >= 0.5
        target = np.asarray(np.load(target_path), dtype=np.float32) >= 0.5
        direct_mask = _load_mask(Path(str(direct_row["mask_artifact"])), "committed_mask")
        active_mask = _load_mask(Path(str(active_row["mask_artifact"])), "committed_mask")
        if not all(mask.shape == prior.shape for mask in (target, direct_mask, active_mask)):
            raise ValueError(f"mask shape mismatch for {source_id}")
        operation = _operation(active_row)
        groups[operation] = groups.get(operation, 0) + 1
        panels = [
            ("Current image", rgb),
            ("Editable prior", _draw_boundary(prior, ORANGE)),
            ("Direct writeback", _draw_boundary(direct_mask, BLUE)),
            ("ActiveMap writeback", _draw_boundary(active_mask, BLUE)),
            ("Reference map", _draw_boundary(target, GREEN)),
            ("Direct residual", _residual(direct_mask, target)),
            ("ActiveMap residual", _residual(active_mask, target)),
        ]
        focus = np.logical_xor(prior, target) | np.logical_xor(direct_mask, active_mask)
        bounds = _expand_bounds(focus, zoom_size)
        left, top, right, bottom = bounds
        case_dir = cases_root / operation.lower() / f"{index:04d}_{key[0]}_b{key[1]:g}"
        case_dir.mkdir(parents=True)
        crop = (slice(top, bottom), slice(left, right))
        assets = {}
        if asset_mode in {"board", "both"}:
            _caseboard(panels, tile_size).save(case_dir / "overview.png", optimize=True)
            _caseboard(
                [(label, panel[top:bottom, left:right]) for label, panel in panels], tile_size
            ).save(case_dir / "zoom.png", optimize=True)
        if asset_mode in {"individual", "both"}:
            assets = save_panel_layers(case_dir, panels, crop=crop)
        row = {
            "schema_version": "muno21-full-validation-casebook-v1",
            "dataset": "MUNO21",
            "split": "val",
            "test_assets_read": False,
            "case_id": key[0],
            "source_example_id": source_id,
            "aoi_id": active_row.get("aoi_id"),
            "budget": key[1],
            "target_edit": operation,
            "folder": str(case_dir.relative_to(output_dir)),
            "zoom_box_xyxy": [left, top, right, bottom],
            "direct_metrics": {
                "false_edit": bool(direct_row.get("false_edit")),
                "missed_edit": bool(direct_row.get("missed_edit")),
                "raster_iou_gain": float(direct_row.get("raster_iou_gain", 0.0)),
            },
            "activemap_metrics": {
                "false_edit": bool(active_row.get("false_edit")),
                "missed_edit": bool(active_row.get("missed_edit")),
                "raster_iou_gain": float(active_row.get("raster_iou_gain", 0.0)),
            },
            "paired_raster_iou_gain_delta": float(active_row.get("raster_iou_gain", 0.0))
            - float(direct_row.get("raster_iou_gain", 0.0)),
            "tool_call_count": int(active_row.get("semantic_tool_called") or 0),
            "selected_evidence_ids": list(active_row.get("selected_evidence_ids") or []),
            "assets": assets,
            "sources": {
                "image": str(image_path),
                "prior": str(prior_path),
                "reference": str(target_path),
                "direct_mask": str(direct_row["mask_artifact"]),
                "activemap_mask": str(active_row["mask_artifact"]),
            },
            "selection": "none; exhaustive paired validation export",
        }
        (case_dir / "manifest.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        index_rows.append(row)
    with (output_dir / "casebook_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "schema_version": "muno21-full-validation-casebook-v1",
        "dataset": "MUNO21",
        "split": "val",
        "test_assets_read": False,
        "direct_writeback": str(direct_path),
        "activemap_writeback": str(active_path),
        "case_count": len(index_rows),
        "direct_input_count": len(direct_keys),
        "activemap_input_count": len(active_keys),
        "unpaired_direct_count": len(direct_keys - active_keys),
        "unpaired_activemap_count": len(active_keys - direct_keys),
        "operation_counts": groups,
        "artifacts_per_case": (
            ["overview.png", "zoom.png", "manifest.json"]
            if asset_mode == "board"
            else ["layers/*.png", "crops/*.png", "manifest.json"]
            if asset_mode == "individual"
            else ["overview.png", "zoom.png", "layers/*.png", "crops/*.png", "manifest.json"]
        ),
        "asset_mode": asset_mode,
        "selection": "none; every paired validation task-budget state is rendered",
        "ordering": "budget then task id",
        "tile_size": tile_size,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("direct_writeback", type=Path)
    parser.add_argument("activemap_writeback", type=Path)
    parser.add_argument("arrays_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tile-size", type=int, default=192)
    parser.add_argument("--zoom-size", type=int, default=256)
    parser.add_argument("--asset-mode", choices=("board", "individual", "both"), default="board")
    args = parser.parse_args()
    if args.tile_size < 64 or args.zoom_size < 32:
        raise ValueError("tile-size must be at least 64 and zoom-size at least 32")
    print(
        json.dumps(
            render(
                args.direct_writeback,
                args.activemap_writeback,
                args.arrays_dir,
                args.output_dir,
                tile_size=args.tile_size,
                zoom_size=args.zoom_size,
                asset_mode=args.asset_mode,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
