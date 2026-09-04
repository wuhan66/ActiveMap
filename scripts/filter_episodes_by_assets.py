#!/usr/bin/env python3
"""Create an episode subset whose remapped assets are all locally available."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.audit_episode_assets import (
        collect_assets,
        parse_asset_root_maps,
        remap_path,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    from audit_episode_assets import collect_assets, parse_asset_root_maps, remap_path


def filter_episodes(
    source: Path,
    output: Path,
    *,
    splits: set[str],
    mappings: tuple[tuple[Path, Path], ...],
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    existence_cache: dict[Path, bool] = {}
    kept: list[str] = []
    kept_splits: Counter[str] = Counter()
    kept_aois: Counter[str] = Counter()
    rejected_missing = 0
    considered = 0

    def is_available(path: Path) -> bool:
        if path not in existence_cache:
            existence_cache[path] = path.is_file()
        return existence_cache[path]

    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            split = str(row.get("split"))
            if split not in splits:
                continue
            considered += 1
            resolved = [remap_path(value, mappings) for value in collect_assets(row)]
            available = all(is_available(path) for path in resolved)
            if not available:
                rejected_missing += 1
                continue
            kept.append(line.rstrip("\n"))
            kept_splits[split] += 1
            kept_aois[str(row.get("aoi_id", "unknown"))] += 1
    if not kept:
        raise ValueError("no episodes have a complete set of remapped assets")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(kept) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "available-episode-subset-v1",
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output": str(output.resolve()),
        "splits": sorted(splits),
        "considered_episodes": considered,
        "kept_episodes": len(kept),
        "rejected_missing_assets": rejected_missing,
        "kept_split_counts": dict(sorted(kept_splits.items())),
        "kept_aoi_counts": dict(sorted(kept_aois.items())),
        "unique_assets_checked": len(existence_cache),
        "asset_root_maps": [
            {"source": str(source_root), "target": str(target_root)}
            for source_root, target_root in mappings
        ],
        "test_assets_read": "test" in splits,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--asset-root-map", action="append", default=[])
    args = parser.parse_args()
    splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    if not splits or not splits <= {"train", "val"}:
        raise ValueError("splits must contain train and/or val")
    summary = filter_episodes(
        args.source,
        args.output,
        splits=splits,
        mappings=parse_asset_root_maps(args.asset_root_map),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
