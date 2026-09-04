#!/usr/bin/env python3
"""Freeze derived validation regions while proving official test disjointness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare(
    summary_path: Path, output_path: Path, *, split: str = "val"
) -> dict[str, object]:
    if split not in {"val", "test"}:
        raise ValueError("region split must be val or test")
    test_assets_read = split == "test"
    if test_assets_read:
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite frozen test regions: {output_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = sorted(map(str, summary["derived_validation_regions"]))
    official_test = sorted(map(str, summary["official_test_regions"]))
    overlap = sorted(set(validation) & set(official_test))
    selected = official_test if test_assets_read else validation
    if not selected:
        raise ValueError(f"{split} region set is empty")
    if overlap:
        raise ValueError(f"validation regions overlap official test: {overlap}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    return {
        "split": split,
        "regions": selected,
        "count": len(selected),
        "official_test_overlap": overlap,
        "source_seed": summary.get("seed"),
        "test_assets_read": test_assets_read,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--frozen-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.frozen_test:
        raise PermissionError("test regions require --frozen-test")
    if args.split != "test" and args.frozen_test:
        raise ValueError("--frozen-test is valid only with --split test")
    print(json.dumps(prepare(args.summary, args.output, split=args.split), indent=2))


if __name__ == "__main__":
    main()
