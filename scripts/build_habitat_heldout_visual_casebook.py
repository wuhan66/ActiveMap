#!/usr/bin/env python3
"""Index all held-out Habitat trajectory boards before qualitative curation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


SEED_PATTERN = re.compile(r"seed(\d+)$")


def _seed(path: Path) -> int:
    match = SEED_PATTERN.fullmatch(path.name)
    if not match:
        raise ValueError(f"unexpected held-out case directory: {path}")
    return int(match.group(1))


def _read_source_rows(path: Path) -> dict[int, list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_seed: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        seed = int(row["seed"])
        by_seed.setdefault(seed, []).append(row)
    return by_seed


def _contact_sheet(files: list[Path], output_path: Path) -> None:
    tile_size = (400, 260)
    board = Image.new("RGB", (4 * tile_size[0], 2 * tile_size[1]), "white")
    for index, file in enumerate(files):
        with Image.open(file) as source:
            tile = ImageOps.fit(source.convert("RGB"), tile_size, method=Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, tile.width, 24), fill="white")
        draw.text((7, 6), file.parent.name.replace("seed", "Held-out trajectory "), fill=(24, 31, 40), font=ImageFont.load_default())
        row, column = divmod(index, 4)
        board.paste(tile, (column * tile.width, row * tile.height))
    board.save(output_path, optimize=True)


def build(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    visuals_dir = source_dir / "paper_visuals"
    source_csv = visuals_dir / "source_data.csv"
    if not source_csv.is_file():
        raise FileNotFoundError(source_csv)
    case_files = sorted(
        visuals_dir.glob("seed*/rgb_trajectory_map_story.png"), key=lambda path: _seed(path.parent)
    )
    if len(case_files) != 8:
        raise ValueError(f"expected eight held-out trajectory boards, found {len(case_files)}")
    source_rows = _read_source_rows(source_csv)
    output_dir.mkdir(parents=True)
    _contact_sheet(case_files, output_dir / "contact_sheet.png")
    selected_payload = json.loads((visuals_dir / "selected_cases.json").read_text(encoding="utf-8"))
    recommended = {int(seed) for seed in selected_payload.get("selected_seeds", [])}
    rows: list[dict[str, Any]] = []
    for file in case_files:
        seed = _seed(file.parent)
        row = {
            "schema_version": "habitat-heldout-visual-casebook-v1",
            "dataset": "Habitat RGB-D occupancy mapping",
            "split": "heldout",
            "test_assets_read": False,
            "trajectory_seed": seed,
            "storyboard": str(file),
            "source_rollout_rows": source_rows.get(seed, []),
            "recommended_for_layout": seed in recommended,
            "selection": "complete held-out export; recommendations fixed in the existing protocol",
            "scope": "development portability visualization; shared motion and frozen acquisition gates",
        }
        rows.append(row)
    with (output_dir / "casebook_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "schema_version": "habitat-heldout-visual-casebook-v1",
        "dataset": "Habitat RGB-D occupancy mapping",
        "split": "heldout",
        "test_assets_read": False,
        "trajectory_count": len(rows),
        "contact_sheet": str(output_dir / "contact_sheet.png"),
        "source_data": str(source_csv),
        "selected_layout_seeds": sorted(recommended),
        "selection": "all eight held-out trajectory boards indexed before curation",
        "scope": "development portability only; no cross-domain superiority claim",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
