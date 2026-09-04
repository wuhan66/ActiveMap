#!/usr/bin/env python3
"""Verify that episode assets required by one split exist after root remapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ASSET_KEYS = {"map_before", "target_map", "image_path", "udm_path"}


def parse_asset_root_maps(values: list[str]) -> tuple[tuple[Path, Path], ...]:
    mappings = []
    for value in values:
        if "=" not in value:
            raise ValueError("asset root maps must use SOURCE=TARGET")
        source_text, target_text = value.split("=", 1)
        source, target = Path(source_text), Path(target_text)
        if not source.is_absolute() or not target.is_absolute():
            raise ValueError("asset root maps must contain absolute paths")
        mappings.append((source, target))
    return tuple(mappings)


def remap_path(value: str, mappings: tuple[tuple[Path, Path], ...]) -> Path:
    original = Path(value)
    for source, target in mappings:
        try:
            return target / original.relative_to(source)
        except ValueError:
            continue
    return original


def collect_assets(value: Any, *, key: str | None = None) -> set[str]:
    if isinstance(value, dict):
        result: set[str] = set()
        for child_key, child in value.items():
            result.update(collect_assets(child, key=child_key))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(collect_assets(child, key=key))
        return result
    if key in ASSET_KEYS and isinstance(value, str) and value:
        return {value}
    return set()


def audit(
    episodes_path: Path,
    *,
    split: str,
    mappings: tuple[tuple[Path, Path], ...],
) -> dict[str, Any]:
    assets: set[str] = set()
    episode_count = 0
    with episodes_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != split:
                continue
            episode_count += 1
            assets.update(collect_assets(row))

    resolved = sorted({remap_path(value, mappings) for value in assets}, key=str)
    missing = [str(path) for path in resolved if not path.is_file()]
    return {
        "episodes": episode_count,
        "unique_assets": len(resolved),
        "existing_assets": len(resolved) - len(missing),
        "missing_assets": len(missing),
        "missing_examples": missing[:20],
        "split": split,
        "asset_root_maps": [
            {"source": str(source), "target": str(target)} for source, target in mappings
        ],
        "test_assets_read": split == "test",
        "passed": episode_count > 0 and not missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--asset-root-map", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit(
        args.episodes,
        split=args.split,
        mappings=parse_asset_root_maps(args.asset_root_map),
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
