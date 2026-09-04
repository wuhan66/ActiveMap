#!/usr/bin/env python3
"""Build a validation-only, AOI-balanced action sentinel for early diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assistant_action(row: dict[str, Any]) -> str:
    content = row["messages"][-1]["content"]
    text = content if isinstance(content, str) else next(
        part["text"] for part in content if part.get("type") == "text"
    )
    return str(json.loads(text)["selection"])


def build_sentinel(
    rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
    *,
    seed: int,
    max_acquire_per_aoi: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if max_acquire_per_aoi is not None and max_acquire_per_aoi <= 0:
        raise ValueError("max_acquire_per_aoi must be positive")
    index = {str(row["example_id"]): row for row in index_rows}
    if len(index) != len(index_rows):
        raise ValueError("evaluation index contains duplicate example_id values")
    if {str(row["example_id"]) for row in rows} != set(index):
        raise ValueError("validation rows and evaluation index do not match")
    if {str(row.get("split")) for row in rows} != {"val"}:
        raise ValueError("sentinel construction only permits validation rows")

    acquire_by_aoi: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stop_by_aoi: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        aoi = str(index[str(row["example_id"])]["aoi_id"])
        target = acquire_by_aoi if _assistant_action(row) == "ACQUIRE" else stop_by_aoi
        target[aoi].append(row)

    selected_ids: set[str] = set()
    per_aoi: dict[str, dict[str, int]] = {}
    for aoi, acquire_rows in sorted(acquire_by_aoi.items()):
        acquire_rows = sorted(
            acquire_rows,
            key=lambda row: hashlib.sha256(
                f"{seed}:ACQUIRE:{row['example_id']}".encode("utf-8")
            ).hexdigest(),
        )
        if max_acquire_per_aoi is not None:
            acquire_rows = acquire_rows[:max_acquire_per_aoi]
        stops = sorted(
            stop_by_aoi[aoi],
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['example_id']}".encode("utf-8")
            ).hexdigest(),
        )[: len(acquire_rows)]
        selected_ids.update(str(row["example_id"]) for row in acquire_rows + stops)
        per_aoi[aoi] = {"ACQUIRE": len(acquire_rows), "STOP": len(stops)}

    selected_rows = [row for row in rows if str(row["example_id"]) in selected_ids]
    selected_index = [
        row for row in index_rows if str(row["example_id"]) in selected_ids
    ]
    counts = defaultdict(int)
    for row in selected_rows:
        counts[_assistant_action(row)] += 1
    summary = {
        "schema_version": "active-catalog-balanced-sentinel-v1",
        "seed": seed,
        "max_acquire_per_aoi": max_acquire_per_aoi,
        "records": len(selected_rows),
        "action_counts": dict(sorted(counts.items())),
        "aoi_count": len(per_aoi),
        "per_aoi": per_aoi,
        "split": "val",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "test_assets_read": False,
    }
    return selected_rows, selected_index, summary


def materialize_images(
    rows: list[dict[str, Any]], source_root: Path, output_root: Path
) -> dict[str, Any]:
    source_root = source_root.resolve()
    assets: dict[str, dict[str, Any]] = {}
    for row in rows:
        for message in row["messages"]:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") != "image":
                    continue
                relative = Path(str(part["image"]))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"unsafe sentinel image path: {relative}")
                source = (source_root / relative).resolve()
                if source_root not in source.parents or not source.is_file():
                    raise FileNotFoundError(source)
                destination = output_root / relative
                if str(relative) in assets:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_file():
                    if _sha256(source) != _sha256(destination):
                        raise ValueError(f"sentinel image hash mismatch: {destination}")
                    method = "existing"
                else:
                    try:
                        os.link(source, destination)
                        method = "hardlink"
                    except OSError:
                        shutil.copy2(source, destination)
                        method = "copy"
                assets[str(relative)] = {
                    "sha256": _sha256(destination),
                    "bytes": destination.stat().st_size,
                    "materialization": method,
                }
    digest = hashlib.sha256(
        json.dumps(assets, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"count": len(assets), "manifest_sha256": digest, "assets": assets}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_jsonl", type=Path)
    parser.add_argument("evaluation_index", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--max-acquire-per-aoi", type=int)
    parser.add_argument("--repair-images", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.repair_images:
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    if args.repair_images:
        val_output = args.output_dir / "val.jsonl"
        index_output = args.output_dir / "val_evaluation_index.jsonl"
        summary_output = args.output_dir / "summary.json"
        for path in (val_output, index_output, summary_output):
            if not path.is_file():
                raise FileNotFoundError(path)
        selected_rows = [
            json.loads(line)
            for line in val_output.read_text(encoding="utf-8").splitlines()
            if line
        ]
        summary = json.loads(summary_output.read_text(encoding="utf-8"))
        summary["image_assets"] = materialize_images(
            selected_rows, args.validation_jsonl.parent, args.output_dir
        )
        summary_output.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return

    rows = [json.loads(line) for line in args.validation_jsonl.read_text(encoding="utf-8").splitlines() if line]
    index_rows = [json.loads(line) for line in args.evaluation_index.read_text(encoding="utf-8").splitlines() if line]
    selected_rows, selected_index, summary = build_sentinel(
        rows,
        index_rows,
        seed=args.seed,
        max_acquire_per_aoi=args.max_acquire_per_aoi,
    )
    args.output_dir.mkdir(parents=True)
    val_output = args.output_dir / "val.jsonl"
    index_output = args.output_dir / "val_evaluation_index.jsonl"
    val_output.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selected_rows), encoding="utf-8")
    index_output.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selected_index), encoding="utf-8")
    summary["image_assets"] = materialize_images(
        selected_rows, args.validation_jsonl.parent, args.output_dir
    )
    summary["sources"] = {
        "validation_sha256": _sha256(args.validation_jsonl),
        "evaluation_index_sha256": _sha256(args.evaluation_index),
    }
    summary["outputs"] = {
        "validation_sha256": _sha256(val_output),
        "evaluation_index_sha256": _sha256(index_output),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
