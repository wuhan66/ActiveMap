#!/usr/bin/env python3
"""Filter executable writebacks to an audited reference task-budget support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> tuple[dict[tuple[str, float], dict[str, Any]], list[tuple[str, float]]]:
    rows: dict[tuple[str, float], dict[str, Any]] = {}
    order = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            continue
        row = json.loads(line)
        key = (str(row["task_id"]), float(row["budget"]))
        if key in rows:
            raise ValueError(f"{path}:{line_number} duplicates {key}")
        rows[key] = row
        order.append(key)
    if not rows:
        raise ValueError(f"empty writeback: {path}")
    return rows, order


def filter_support(
    reference_path: Path,
    candidate_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference, order = _read(reference_path)
    candidate, _ = _read(candidate_path)
    missing = [key for key in order if key not in candidate]
    if missing:
        raise ValueError(
            f"candidate lacks {len(missing)} reference keys; first={missing[0]}"
        )
    output = []
    for key in order:
        reference_row = reference[key]
        candidate_row = candidate[key]
        for field in ("target", "aoi_id", "split"):
            if reference_row.get(field) != candidate_row.get(field):
                raise ValueError(f"metadata mismatch for {key}: {field}")
        if candidate_row.get("test_assets_read", False):
            raise ValueError(f"candidate contains test evidence for {key}")
        output.append(candidate_row)
    return output, {
        "schema_version": "activemap-writeback-support-filter-v1",
        "reference": str(reference_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "matched_count": len(output),
        "coverage": len(output) / len(reference),
        "split": "val",
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    rows, summary = filter_support(args.reference, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
