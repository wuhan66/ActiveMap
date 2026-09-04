#!/usr/bin/env python3
"""Audit temporal-change rasters used by the spatial Tool-Belief branch."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc
    return rows


def audit_spatial_artifacts(
    records: list[dict[str, Any]], artifact_dir: Path
) -> dict[str, Any]:
    temporal = [
        row for row in records if row.get("tool_result", {}).get("tool") == "TEMPORAL_CHANGE"
    ]
    split_counts: Counter[str] = Counter()
    edit_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    missing: list[str] = []
    invalid: list[str] = []
    blank_count = 0
    nonfinite_count = 0
    minima: list[float] = []
    maxima: list[float] = []
    means: list[float] = []
    expected_paths: set[Path] = set()

    for row in temporal:
        split = str(row.get("split", ""))
        call_id = str(row["tool_result"]["call_id"])
        split_counts[split] += 1
        edit_counts[str(row.get("gt_edit", ""))] += 1
        if split == "test":
            invalid.append(f"{call_id}: test record is present")
        path = artifact_dir / f"{call_id}_change.npy"
        expected_paths.add(path.resolve())
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            array = np.load(path, allow_pickle=False)
        except Exception as exc:
            invalid.append(f"{path}: {exc}")
            continue
        shape_counts[str(tuple(array.shape))] += 1
        dtype_counts[str(array.dtype)] += 1
        if array.ndim not in {2, 3} or array.size == 0:
            invalid.append(f"{path}: unsupported shape {array.shape}")
            continue
        finite = np.isfinite(array)
        if not finite.all():
            nonfinite_count += 1
            invalid.append(f"{path}: contains non-finite values")
            continue
        minimum = float(array.min())
        maximum = float(array.max())
        mean = float(array.mean())
        minima.append(minimum)
        maxima.append(maximum)
        means.append(mean)
        if maximum == minimum:
            blank_count += 1

    discovered = {path.resolve() for path in artifact_dir.glob("*_change.npy")}
    orphaned = sorted(str(path) for path in discovered - expected_paths)
    summary = {
        "record_count": len(records),
        "temporal_record_count": len(temporal),
        "artifact_count": len(discovered),
        "split_counts": dict(sorted(split_counts.items())),
        "edit_counts": dict(sorted(edit_counts.items())),
        "shape_counts": dict(sorted(shape_counts.items())),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "missing_count": len(missing),
        "orphaned_count": len(orphaned),
        "invalid_count": len(invalid),
        "nonfinite_count": nonfinite_count,
        "blank_count": blank_count,
        "value_summary": {
            "minimum": min(minima) if minima else None,
            "maximum": max(maxima) if maxima else None,
            "mean_of_means": float(np.mean(means)) if means else None,
        },
        "missing_examples": missing[:20],
        "orphaned_examples": orphaned[:20],
        "invalid_examples": invalid[:20],
    }
    summary["passed"] = (
        len(temporal) > 0
        and len(temporal) == len(discovered)
        and not missing
        and not orphaned
        and not invalid
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = audit_spatial_artifacts(_rows(args.records), args.artifact_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
