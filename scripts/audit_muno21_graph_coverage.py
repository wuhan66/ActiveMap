#!/usr/bin/env python3
"""Require one exported graph per frozen MUNO21 validation annotation and budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(
    annotations_path: Path,
    regions_path: Path,
    graphs_root: Path,
    *,
    split: str = "val",
) -> dict[str, object]:
    if split not in {"val", "test"}:
        raise ValueError("coverage split must be val or test")
    test_assets_read = split == "test"
    if test_assets_read:
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    regions = set(map(str, json.loads(regions_path.read_text(encoding="utf-8"))))
    expected = {
        index
        for index, annotation in enumerate(annotations)
        if str(annotation["Cluster"]["Region"]) in regions
    }
    if not expected:
        raise ValueError("no annotations match the frozen validation regions")
    budgets = {}
    for budget_dir in sorted(graphs_root.glob("budget-*")):
        if not budget_dir.is_dir():
            continue
        actual = {int(path.stem) for path in budget_dir.glob("*.graph")}
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ValueError(
                f"graph coverage mismatch for {budget_dir.name}: "
                f"missing={missing[:20]}, extra={extra[:20]}"
            )
        budgets[budget_dir.name] = len(actual)
    if not budgets:
        raise ValueError(f"no budget graph directories in {graphs_root}")
    return {
        "regions": sorted(regions),
        "expected_annotation_count": len(expected),
        "budget_graph_counts": budgets,
        "coverage_complete": True,
        "split": split,
        "test_assets_read": test_assets_read,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    parser.add_argument("regions", type=Path)
    parser.add_argument("graphs_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--frozen-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.frozen_test:
        raise PermissionError("test graph coverage requires --frozen-test")
    if args.split != "test" and args.frozen_test:
        raise ValueError("--frozen-test is valid only with --split test")
    result = audit(
        args.annotations, args.regions, args.graphs_root, split=args.split
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
