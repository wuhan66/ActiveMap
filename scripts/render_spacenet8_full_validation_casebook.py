#!/usr/bin/env python3
"""Render all validation multi-POST SpaceNet8 cases before qualitative selection."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from scripts.render_spacenet8_active_qualitative import (
    BLUE,
    GREEN,
    _assert_non_test,
    _boundary,
    _bounds,
    _load_seed_rows,
    _read_jsonl,
    _residual,
    _rgb,
    _safe_gate,
)
try:
    from scripts.visual_asset_utils import save_panel_layers
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from visual_asset_utils import save_panel_layers


def _tile(array: np.ndarray, label: str, size: int) -> Image.Image:
    image = Image.fromarray(array, mode="RGB").resize((size, size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size, 17), fill=(255, 255, 255))
    draw.text((4, 3), label, fill=(0, 0, 0))
    return image


def _caseboard(panels: list[tuple[str, np.ndarray]], tile_size: int) -> Image.Image:
    columns = 3
    rows = int(np.ceil(len(panels) / columns))
    board = Image.new("RGB", (columns * tile_size, rows * tile_size), "white")
    for index, (label, panel) in enumerate(panels):
        row, column = divmod(index, columns)
        board.paste(_tile(panel, label, tile_size), (column * tile_size, row * tile_size))
    return board


def _ensemble_mask(rows: list[dict[str, Any]], valid: np.ndarray) -> np.ndarray:
    probability = np.mean(
        [np.asarray(np.load(str(row["mask_path"]))["probability"], dtype=np.float32) for row in rows],
        axis=0,
    )
    return (probability >= 0.5) & valid


def render(
    candidate_root: Path,
    selector_root: Path,
    output_dir: Path,
    *,
    tile_size: int,
    zoom_size: int,
    asset_mode: str = "board",
    include_single_candidate: bool = False,
    seed_pattern: str = "rank100_changer_seed*/per_candidate.jsonl",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if asset_mode not in {"board", "individual", "both"}:
        raise ValueError("asset_mode must be board, individual, or both")
    aggregate_path = selector_root / "aggregated_candidates.jsonl"
    summary_path = selector_root / "summary.json"
    aggregate = _read_jsonl(aggregate_path)
    selector_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if selector_summary["protocol"].get("test_assets_read") is not False:
        raise ValueError("selector summary is not validation-only")
    threshold = float(selector_summary["protocol"]["safe_commit_threshold"])
    all_rows = [dict(row) for row in aggregate]
    for row in all_rows:
        _assert_non_test(row, aggregate_path)
    probabilities = _safe_gate(all_rows, threshold)
    seed_rows = _load_seed_rows(candidate_root, seed_pattern)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        if row["split"] == "val":
            groups[str(row["sample_id"])].append(row)
    candidates: list[tuple[str, list[dict[str, Any]], dict[str, Any], bool]] = []
    skipped_single_candidate = 0
    for sample_id, group in groups.items():
        group.sort(key=lambda row: int(row["candidate_index"]))
        if len(group) < 2 and not include_single_candidate:
            skipped_single_candidate += 1
            continue
        selected = (
            max(group, key=lambda row: float(row["ranker_score"]))
            if len(group) > 1
            else group[0]
        )
        candidates.append((sample_id, group, selected, probabilities[str(id(selected))] >= threshold))
    if not candidates:
        raise ValueError("no validation multi-POST candidate groups")
    candidates.sort(key=lambda item: item[0])
    cases_root = output_dir / "cases"
    cases_root.mkdir(parents=True)
    index_rows: list[dict[str, Any]] = []
    for index, (sample_id, group, selected, safe_commit) in enumerate(candidates, start=1):
        first = group[0]
        selected_seed_rows = seed_rows[(sample_id, str(selected["candidate_id"]))]
        first_seed_rows = seed_rows[(sample_id, str(first["candidate_id"]))]
        with np.load(str(selected["array_path"])) as payload:
            pre = _rgb(payload["pre"])
            selected_post = _rgb(payload["post"])
            target = np.asarray(payload["target"], dtype=bool)
            valid = np.asarray(payload["valid"], dtype=bool)
        with np.load(str(first["array_path"])) as payload:
            first_post = _rgb(payload["post"])
        direct_mask = _ensemble_mask(first_seed_rows, valid)
        selected_mask = _ensemble_mask(selected_seed_rows, valid)
        safe_mask = selected_mask if safe_commit else np.zeros_like(selected_mask)
        panels = [
            ("Pre-event image", pre),
            ("First POST image", first_post),
            ("Selected POST image", selected_post),
            ("First-candidate draft", _boundary(direct_mask, BLUE)),
            ("Selected draft", _boundary(selected_mask, BLUE)),
            ("Safe Commit", _boundary(safe_mask, BLUE)),
            ("Reference change", _boundary(target, GREEN)),
            ("First-candidate residual", _residual(direct_mask, target)),
            ("Safe Commit residual", _residual(safe_mask, target)),
        ]
        focus = np.logical_xor(direct_mask, safe_mask) | np.logical_xor(safe_mask, target)
        left, top, right, bottom = _bounds(focus, zoom_size)
        case_dir = cases_root / f"{index:03d}_{sample_id}"
        case_dir.mkdir()
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
            "schema_version": "spacenet8-full-validation-casebook-v1",
            "dataset": "SpaceNet 8",
            "split": "val",
            "test_assets_read": False,
            "case_id": sample_id,
            "folder": str(case_dir.relative_to(output_dir)),
            "backend": "Changer rank-100, three-seed mean probability",
            "candidate_count": len(group),
            "selection": {
                "first_candidate_id": first["candidate_id"],
                "learned_selected_candidate_id": selected["candidate_id"],
                "first_map_iou": float(first["map_iou"]),
                "selected_map_iou": float(selected["map_iou"]),
                "learned_minus_first": float(selected["map_iou"]) - float(first["map_iou"]),
            },
            "safe_commit": {"threshold": threshold, "commit": bool(safe_commit)},
            "assets": assets,
            "zoom_box_xyxy": [left, top, right, bottom],
            "sources": {
                "aggregate": str(aggregate_path),
                "selector_summary": str(summary_path),
                "selected_array": str(selected["array_path"]),
                "first_array": str(first["array_path"]),
                "selected_seed_masks": [str(item["mask_path"]) for item in selected_seed_rows],
                "first_seed_masks": [str(item["mask_path"]) for item in first_seed_rows],
            },
            "selection_rule": (
                "none; exhaustive validation multi-POST export"
                if len(group) > 1
                else "single POST; exported for visual coverage only"
            ),
        }
        (case_dir / "manifest.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        index_rows.append(row)
    with (output_dir / "casebook_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row) + "\n")
    result = {
        "schema_version": "spacenet8-full-validation-casebook-v1",
        "dataset": "SpaceNet 8",
        "split": "val",
        "test_assets_read": False,
        "case_count": len(index_rows),
        "validation_sample_groups": len(groups),
        "skipped_single_candidate_groups": skipped_single_candidate,
        "artifacts_per_case": (
            ["overview.png", "zoom.png", "manifest.json"]
            if asset_mode == "board"
            else ["layers/*.png", "crops/*.png", "manifest.json"]
            if asset_mode == "individual"
            else ["overview.png", "zoom.png", "layers/*.png", "crops/*.png", "manifest.json"]
        ),
        "asset_mode": asset_mode,
        "selection": (
            "none; every validation episode is rendered"
            if include_single_candidate
            else "none; every validation multi-POST episode is rendered"
        ),
        "ordering": "sample id",
        "tile_size": tile_size,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("selector_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tile-size", type=int, default=192)
    parser.add_argument("--zoom-size", type=int, default=256)
    parser.add_argument("--asset-mode", choices=("board", "individual", "both"), default="board")
    parser.add_argument(
        "--include-single-candidate",
        action="store_true",
        help="Export single-POST validation transitions for visual coverage only.",
    )
    parser.add_argument(
        "--seed-pattern",
        default="rank100_changer_seed*/per_candidate.jsonl",
        help="Glob below candidate_root selecting one three-seed backend family.",
    )
    args = parser.parse_args()
    if args.tile_size < 64 or args.zoom_size < 32:
        raise ValueError("tile-size must be at least 64 and zoom-size at least 32")
    print(
        json.dumps(
            render(
                args.candidate_root,
                args.selector_root,
                args.output_dir,
                tile_size=args.tile_size,
                zoom_size=args.zoom_size,
                asset_mode=args.asset_mode,
                include_single_candidate=args.include_single_candidate,
                seed_pattern=args.seed_pattern,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
