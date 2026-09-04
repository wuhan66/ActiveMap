#!/usr/bin/env python3
"""Create review-only contact sheets from a stratified casebook registry."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOTS = {
    "SpaceNet 7": ("sn7", "full_casebook"),
    "MUNO21": ("muno21_aoi_v2",),
    "SpaceNet 8": ("spacenet8",),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _source(casebook_root: Path, row: dict[str, Any]) -> Path:
    if row["dataset"] not in ROOTS:
        raise ValueError(f"unsupported dataset: {row['dataset']}")
    return casebook_root.joinpath(*ROOTS[row["dataset"]], row["folder"])


def _card(source: Path, row: dict[str, Any], width: int) -> Image.Image:
    overview = Image.open(source / "overview.png").convert("RGB")
    zoom = Image.open(source / "zoom.png").convert("RGB")
    height = int(round(width * overview.height / overview.width))
    overview = overview.resize((width, height), Image.Resampling.LANCZOS)
    zoom = zoom.resize((width, height), Image.Resampling.LANCZOS)
    card = Image.new("RGB", (width, height * 2 + 26), "white")
    card.paste(overview, (0, 26))
    card.paste(zoom, (0, height + 26))
    draw = ImageDraw.Draw(card)
    draw.text((4, 5), f"{row['dataset']} | {' / '.join(row['stratum'])} | {row['case_id']}", fill=(0, 0, 0))
    return card


def render(registry_path: Path, casebook_root: Path, output_dir: Path, *, columns: int, card_width: int) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    rows = _read_jsonl(registry_path)
    if not rows:
        raise ValueError("empty selection registry")
    output_dir.mkdir(parents=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    output_rows = []
    pages = {}
    for dataset, entries in sorted(grouped.items()):
        cards = [(_card(_source(casebook_root, row), row, card_width), row) for row in entries]
        rows_per_page = 3
        capacity = columns * rows_per_page
        dataset_dir = output_dir / dataset.lower().replace(" ", "_")
        dataset_dir.mkdir()
        dataset_pages = 0
        for page_index, start in enumerate(range(0, len(cards), capacity), start=1):
            batch = cards[start : start + capacity]
            card_height = max(card.height for card, _ in batch)
            canvas = Image.new("RGB", (columns * card_width, rows_per_page * card_height), "white")
            for local_index, (card, row) in enumerate(batch):
                y, x = divmod(local_index, columns)
                canvas.paste(card, (x * card_width, y * card_height))
                output_rows.append({"dataset": dataset, "page": page_index, "cell": local_index, "case_id": row["case_id"], "stratum": row["stratum"]})
            canvas.save(dataset_dir / f"candidate_page_{page_index:02d}.png", optimize=True)
            dataset_pages += 1
        pages[dataset] = dataset_pages
    with (output_dir / "gallery_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "schema_version": "qualitative-selection-gallery-v1",
        "source_registry": str(registry_path),
        "case_count": len(rows),
        "page_counts": pages,
        "purpose": "review-only strata gallery; not a manuscript figure",
        "split": "val",
        "test_assets_read": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("casebook_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--card-width", type=int, default=384)
    args = parser.parse_args()
    if args.columns < 1 or args.card_width < 128:
        raise ValueError("columns must be positive and card-width at least 128")
    print(json.dumps(render(args.registry, args.casebook_root, args.output_dir, columns=args.columns, card_width=args.card_width), indent=2))


if __name__ == "__main__":
    main()
