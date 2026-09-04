#!/usr/bin/env python3
"""Break executable writeback gains and failures down by semantic edit type."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label:
        raise argparse.ArgumentTypeError("candidate must use LABEL=WRITEBACK_JSONL")
    return label, Path(raw_path)


def load_baseline(path: Path) -> dict[tuple[str, float], dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["task_id"]), float(row["budget"]))
            rows[key] = {
                "raster_iou": float(row["raster_iou"]),
                "utility": float(row["episode_utility_v2_balanced"]),
                "component_error": float(row["component_count_absolute_error"]),
                "false_edit": bool(row["false_edit"]),
                "missed_edit": bool(row["missed_edit"]),
            }
    if not rows:
        raise ValueError(f"empty baseline: {path}")
    return rows


def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean_iou_delta": fmean(row["iou_delta"] for row in values),
        "iou_degradation_rate": fmean(row["iou_delta"] < 0.0 for row in values),
        "mean_utility_delta": fmean(row["utility_delta"] for row in values),
        "utility_improvement_rate": fmean(row["utility_delta"] > 0.0 for row in values),
        "mean_component_error_delta": fmean(
            row["component_error_delta"] for row in values
        ),
        "false_edit_delta_rate": fmean(row["false_edit_delta"] for row in values),
        "missed_edit_delta_rate": fmean(row["missed_edit_delta"] for row in values),
        "mean_spent_cost": fmean(row["spent_cost"] for row in values),
    }


def analyze(
    baseline_path: Path,
    candidates: dict[str, Path],
    *,
    worst_examples: int,
) -> dict[str, Any]:
    baseline = load_baseline(baseline_path)
    results = []
    for label, path in candidates.items():
        rows = []
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                candidate = json.loads(line)
                key = (str(candidate["task_id"]), float(candidate["budget"]))
                reference = baseline.get(key)
                if reference is None:
                    raise ValueError(f"candidate has unmatched row: {label} {key}")
                row = {
                    "task_id": key[0],
                    "aoi_id": candidate.get("aoi_id"),
                    "budget": key[1],
                    "target": str(candidate["target"]),
                    "effective_operation": str(candidate["effective_operation"]),
                    "writeback_changed": bool(candidate["writeback_changed"]),
                    "iou_delta": float(candidate["raster_iou"])
                    - reference["raster_iou"],
                    "utility_delta": float(candidate["episode_utility_v2_balanced"])
                    - reference["utility"],
                    "component_error_delta": float(
                        candidate["component_count_absolute_error"]
                    )
                    - reference["component_error"],
                    "false_edit_delta": int(bool(candidate["false_edit"]))
                    - int(reference["false_edit"]),
                    "missed_edit_delta": int(bool(candidate["missed_edit"]))
                    - int(reference["missed_edit"]),
                    "spent_cost": float(candidate["spent_cost"]),
                    "fused_confidence": float(candidate["fused_confidence"]),
                    "selected_evidence_ids": candidate["selected_evidence_ids"],
                }
                rows.append(row)
                groups[("target", row["target"])].append(row)
                groups[("budget", str(row["budget"]))].append(row)
                groups[("changed", str(row["writeback_changed"]))].append(row)
        if len(rows) != len(baseline):
            raise ValueError(f"candidate row count differs: {label}")
        harmful = sorted(rows, key=lambda row: (row["iou_delta"], row["utility_delta"]))
        results.append(
            {
                "label": label,
                "source": str(path.resolve()),
                "overall": summarize(rows),
                "groups": [
                    {"group": kind, "value": value, **summarize(group_rows)}
                    for (kind, value), group_rows in sorted(groups.items())
                ],
                "worst_iou_examples": harmful[:worst_examples],
            }
        )
    return {
        "schema_version": "writeback-tradeoff-audit-v1",
        "baseline": str(baseline_path.resolve()),
        "candidates": results,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", action="append", type=labeled_path, required=True)
    parser.add_argument("--worst-examples", type=int, default=25)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = analyze(
        args.baseline,
        dict(args.candidate),
        worst_examples=args.worst_examples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
