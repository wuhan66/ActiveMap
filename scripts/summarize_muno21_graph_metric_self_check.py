#!/usr/bin/env python3
"""Summarize official metric self-check outputs without treating them as model results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _series(path: Path) -> list[float]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [float(row[1]) for row in rows]


def summarize(root: Path) -> dict[str, Any]:
    manifest = json.loads(
        (root / "self_check_manifest.json").read_text(encoding="utf-8")
    )
    apls = _series(root / "scores.json")
    pixel_f1 = _series(root / "geo.json")
    error_rate = float(json.loads((root / "error.json").read_text(encoding="utf-8")))
    expected_changed = int(manifest["changed_graph_count"])
    if len(apls) != expected_changed or len(pixel_f1) != expected_changed:
        raise ValueError(
            "changed-annotation metric coverage mismatch: "
            f"expected={expected_changed}, apls={len(apls)}, pixel_f1={len(pixel_f1)}"
        )
    return {
        "schema_version": "muno21-official-graph-self-check-report-v1",
        "purpose": "evaluator_self_consistency_only",
        "model_result": False,
        "regions": manifest["regions"],
        "changed_annotation_count": expected_changed,
        "nochange_annotation_count": int(manifest["nochange_graph_count"]),
        "apls": {
            "mean": statistics.fmean(apls),
            "minimum": min(apls),
            "maximum": max(apls),
        },
        "pixel_f1": {
            "mean": statistics.fmean(pixel_f1),
            "minimum": min(pixel_f1),
            "maximum": max(pixel_f1),
        },
        "error_rate": error_rate,
        "passed": statistics.fmean(apls) > 0.9 and statistics.fmean(pixel_f1) > 0.9,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
