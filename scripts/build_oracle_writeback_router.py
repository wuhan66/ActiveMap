#!/usr/bin/env python3
"""Build a diagnostic oracle frontier from two matched writeback outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"no writeback rows in {path}")
    return rows


def _key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["task_id"]), float(row["budget"])


def route(
    primary_rows: list[dict[str, Any]],
    alternate_rows: list[dict[str, Any]],
    *,
    metric: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary = {_key(row): row for row in primary_rows}
    alternate = {_key(row): row for row in alternate_rows}
    if len(primary) != len(primary_rows) or len(alternate) != len(alternate_rows):
        raise ValueError("writeback inputs contain duplicate task-budget rows")
    if primary.keys() != alternate.keys():
        raise ValueError("writeback inputs must contain identical task-budget rows")

    selected = []
    alternate_count = 0
    for key in sorted(primary):
        left, right = primary[key], alternate[key]
        if metric not in left or metric not in right:
            raise ValueError(f"routing metric is missing: {metric}")
        choose_alternate = float(right[metric]) > float(left[metric])
        source, row = ("alternate", right) if choose_alternate else ("primary", left)
        alternate_count += int(choose_alternate)
        selected.append(
            {
                **row,
                "oracle_routing_source": source,
                "oracle_routing_metric": metric,
                "oracle_primary_metric": float(left[metric]),
                "oracle_alternate_metric": float(right[metric]),
                "oracle_diagnostic_only": True,
            }
        )
    count = len(selected)
    return selected, {
        "schema_version": "oracle-writeback-router-v1",
        "diagnostic_only": True,
        "deployment_valid": False,
        "routing_metric": metric,
        "sample_count": count,
        "primary_selected": count - alternate_count,
        "alternate_selected": alternate_count,
        "alternate_selection_rate": alternate_count / count,
        "test_assets_read": any(bool(row.get("test_assets_read")) for row in selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("primary", type=Path)
    parser.add_argument("alternate", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--metric",
        choices=("raster_iou", "episode_utility_v2_balanced"),
        default="raster_iou",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    rows, summary = route(_load(args.primary), _load(args.alternate), metric=args.metric)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "writeback.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
