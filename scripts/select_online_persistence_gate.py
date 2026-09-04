#!/usr/bin/env python3
"""Freeze an online Safe Commit threshold from chronological training chains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_candidate(value: str) -> tuple[float, Path]:
    threshold, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("candidate must be THRESHOLD=/path/to/summary.json")
    return float(threshold), Path(path)


def load(path: Path, threshold: float) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema_version") != "activemap-online-persistent-maintenance-v1":
        raise ValueError(f"not an online persistence summary: {path}")
    if result.get("split") != "train" or result.get("test_assets_read") is not False:
        raise ValueError(f"gate selection must use train-only output: {path}")
    protocol = result.get("protocol", {})
    selected = float(protocol.get("safe_commit_gate", {}).get("confidence_threshold", -1.0))
    if abs(selected - threshold) > 1e-9:
        raise ValueError(f"threshold key and summary disagree: {threshold} vs {selected}")
    branches = result.get("metrics", {}).get("branches", {})
    if set(branches) != {"independent_reset", "carry_always_commit", "carry_safe_commit"}:
        raise ValueError(f"incomplete online branches: {path}")
    return result


def select(
    candidates: list[tuple[float, dict[str, Any], Path]], *, false_edit_fraction: float
) -> dict[str, Any]:
    if not 0.0 <= false_edit_fraction <= 1.0:
        raise ValueError("false-edit fraction must be in [0, 1]")
    rows = []
    for threshold, result, path in candidates:
        branches = result["metrics"]["branches"]
        always = branches["carry_always_commit"]
        safe = branches["carry_safe_commit"]
        allowed_false_edit = false_edit_fraction * float(always["false_edit_rate"])
        eligible = float(safe["false_edit_rate"]) <= allowed_false_edit + 1e-12
        rows.append(
            {
                "threshold": threshold,
                "source": str(path.resolve()),
                "mean_step_raster_iou": float(safe["mean_step_raster_iou"]),
                "mean_final_chain_raster_iou": float(safe["mean_final_chain_raster_iou"]),
                "false_edit_rate": float(safe["false_edit_rate"]),
                "missed_edit_rate": float(safe["missed_edit_rate"]),
                "always_false_edit_rate": float(always["false_edit_rate"]),
                "allowed_false_edit_rate": allowed_false_edit,
                "eligible": eligible,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise ValueError("no train threshold meets the predeclared false-edit reduction constraint")
    selected = max(
        eligible,
        key=lambda row: (
            row["mean_step_raster_iou"],
            row["mean_final_chain_raster_iou"],
            -row["missed_edit_rate"],
            -row["threshold"],
        ),
    )
    return {
        "schema_version": "activemap-online-persistent-gate-selection-v1",
        "split": "train",
        "test_assets_read": False,
        "selection_rule": {
            "constraint": "safe false-edit rate <= configured fraction of unconditional carry false-edit rate",
            "false_edit_fraction": false_edit_fraction,
            "objective": "maximize safe-carry mean step raster IoU, then final-chain IoU, then minimize missed edits",
        },
        "candidates": sorted(rows, key=lambda row: row["threshold"]),
        "selected": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--false-edit-fraction", type=float, default=0.8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = select(
        [(threshold, load(path, threshold), path) for threshold, path in args.candidate],
        false_edit_fraction=args.false_edit_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
