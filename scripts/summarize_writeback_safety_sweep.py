#!/usr/bin/env python3
"""Select one non-degenerate validation writeback safety configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.compare_agent_writebacks import _load, _metrics


def _labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label:
        raise argparse.ArgumentTypeError("candidate must use LABEL=WRITEBACK_JSONL")
    return label, Path(raw_path)


def summarize(
    candidates: dict[str, Path], *, min_retained_change_rate: float = 0.5
) -> dict[str, Any]:
    if "raw" not in candidates or len(candidates) < 2:
        raise ValueError("raw and at least one safety candidate are required")
    if not 0.0 < min_retained_change_rate <= 1.0:
        raise ValueError("min_retained_change_rate must be in (0, 1]")
    loaded = {label: _load(path) for label, path in candidates.items()}
    reference_keys = loaded["raw"].keys()
    for label, rows in loaded.items():
        if rows.keys() != reference_keys:
            raise ValueError(f"writeback keys differ for {label}")
        if any(bool(row.get("test_assets_read")) for row in rows.values()):
            raise ValueError("safety sweep must use validation-only rows")
    raw_changed = sum(bool(row.get("writeback_changed")) for row in loaded["raw"].values())
    if not raw_changed:
        raise ValueError("raw writeback contains no executable map changes")
    rows = []
    for label, values in loaded.items():
        metrics = _metrics(list(values.values()))
        changed = sum(bool(row.get("writeback_changed")) for row in values.values())
        rows.append(
            {
                "label": label,
                "source": str(candidates[label].resolve()),
                "changed_rows": changed,
                "changed_rate": changed / len(values),
                "retained_change_rate": changed / raw_changed,
                "mean_fused_confidence": float(
                    np.mean([float(row["fused_confidence"]) for row in values.values()])
                ),
                **metrics,
            }
        )
    for row in rows:
        row["feasible"] = row["retained_change_rate"] >= min_retained_change_rate
    feasible = [row for row in rows if row["feasible"]]
    selected = max(
        feasible,
        key=lambda row: (
            row["episode_utility_v2_balanced_auc"],
            row["raster_iou_auc"],
            -row["false_edit_auc"],
            row["changed_rows"],
        ),
    )
    return {
        "schema_version": "writeback-safety-sweep-summary-v1",
        "selection_split": "validation",
        "selection_rule": (
            "retain at least the configured fraction of raw executable deltas; then maximize "
            "balanced Utility v2 AUC, raster IoU, and false-edit safety"
        ),
        "sample_count": len(reference_keys),
        "raw_changed_rows": raw_changed,
        "minimum_retained_change_rate": min_retained_change_rate,
        "candidates": sorted(rows, key=lambda row: row["label"]),
        "selected_label": selected["label"],
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", action="append", type=_labeled_path, required=True)
    parser.add_argument("--min-retained-change-rate", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = summarize(
        dict(args.candidate),
        min_retained_change_rate=args.min_retained_change_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
