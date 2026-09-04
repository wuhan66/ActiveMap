#!/usr/bin/env python3
"""Audit whether executable writeback episodes contain raster map-change signal."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def audit(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("writeback audit requires at least one row")
    prior_iou = np.asarray([float(row["prior_raster_iou"]) for row in rows])
    target_counts = Counter(str(row["target"]) for row in rows)
    by_target: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_target[str(row["target"])].append(float(row["prior_raster_iou"]))
    changed = prior_iou < 1.0 - 1e-12
    return {
        "schema_version": "writeback-supervision-signal-audit-v1",
        "source": str(path.resolve()),
        "sample_count": len(rows),
        "task_count": len({str(row["task_id"]) for row in rows}),
        "prior_target_raster_iou": {
            "minimum": float(np.min(prior_iou)),
            "mean": float(np.mean(prior_iou)),
            "maximum": float(np.max(prior_iou)),
            "exact_match_rows": int(np.sum(~changed)),
            "changed_rows": int(np.sum(changed)),
            "changed_rate": float(np.mean(changed)),
        },
        "target_operation_counts": dict(sorted(target_counts.items())),
        "by_target": {
            target: {
                "count": len(values),
                "mean_prior_target_iou": float(np.mean(values)),
                "changed_rows": int(np.sum(np.asarray(values) < 1.0 - 1e-12)),
            }
            for target, values in sorted(by_target.items())
        },
        "has_executable_change_supervision": bool(np.any(changed)),
        "test_assets_read": any(bool(row.get("test_assets_read")) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("writeback", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.writeback)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
