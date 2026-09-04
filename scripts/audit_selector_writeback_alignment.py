#!/usr/bin/env python3
"""Audit whether selector rollout utility agrees with executable writeback utility."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def _mean(rows: list[dict[str, Any]], name: str) -> float:
    return fmean(float(row[name]) for row in rows)


def audit(
    trace_path: Path,
    baseline_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    traces = {str(row["sample_id"]): row for row in _jsonl(trace_path)}
    baseline = {
        (str(row["task_id"]), float(row["budget"])): row for row in _jsonl(baseline_path)
    }
    joined = []
    for candidate in _jsonl(candidate_path):
        key = (str(candidate["task_id"]), float(candidate["budget"]))
        reference = baseline.get(key)
        trace = traces.get(str(candidate["source_example_id"]))
        if reference is None or trace is None:
            raise ValueError(f"unmatched candidate row: {key}")
        joined.append(
            {
                "budget": float(candidate["budget"]),
                "acquisitions": int(trace["acquisitions"]),
                "proxy_utility": float(trace["quality_cost_utility"]),
                "actual_utility": float(candidate["episode_utility_v2_balanced"]),
                "baseline_utility": float(reference["episode_utility_v2_balanced"]),
                "raster_iou": float(candidate["raster_iou"]),
                "baseline_raster_iou": float(reference["raster_iou"]),
                "spent_cost": float(candidate["spent_cost"]),
            }
        )
    if len(joined) != len(baseline):
        raise ValueError("candidate and baseline row counts differ")

    groups: dict[tuple[float | str, int | str], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        groups[(row["budget"], row["acquisitions"])].append(row)
        groups[(row["budget"], "all")].append(row)
        groups[("all", row["acquisitions"])].append(row)
        groups[("all", "all")].append(row)

    summaries = []
    for (budget, acquisitions), rows in sorted(
        groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        actual_delta = [row["actual_utility"] - row["baseline_utility"] for row in rows]
        iou_delta = [row["raster_iou"] - row["baseline_raster_iou"] for row in rows]
        alignment_gap = [row["actual_utility"] - row["proxy_utility"] for row in rows]
        summaries.append(
            {
                "budget": budget,
                "acquisitions": acquisitions,
                "count": len(rows),
                "mean_proxy_utility": _mean(rows, "proxy_utility"),
                "mean_actual_utility": _mean(rows, "actual_utility"),
                "mean_baseline_utility": _mean(rows, "baseline_utility"),
                "mean_actual_utility_delta": fmean(actual_delta),
                "mean_actual_minus_proxy": fmean(alignment_gap),
                "mean_raster_iou_delta": fmean(iou_delta),
                "actual_improvement_rate": fmean(value > 0 for value in actual_delta),
                "raster_iou_degradation_rate": fmean(value < 0 for value in iou_delta),
                "mean_spent_cost": _mean(rows, "spent_cost"),
            }
        )
    return {
        "schema_version": "selector-writeback-alignment-audit-v1",
        "sample_count": len(joined),
        "trace": str(trace_path.resolve()),
        "baseline": str(baseline_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "groups": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.trace, args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
