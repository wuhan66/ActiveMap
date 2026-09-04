#!/usr/bin/env python3
"""Aggregate calibration-selected MUNO21 operating points across fixed seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

METRICS = ("accuracy", "macro_f1", "false_edit_rate", "missed_edit_rate")


def _select_at_cap(summary: dict[str, Any], false_edit_cap: float) -> dict[str, Any]:
    calibration_rows = [
        row
        for row in summary["calibration"]["operation"]["grid"]
        if row["metrics"]["false_edit_rate"] <= false_edit_cap + 1e-12
    ]
    if not calibration_rows:
        raise ValueError(f"no calibration point satisfies false-edit cap {false_edit_cap}")
    selected = max(
        calibration_rows,
        key=lambda row: (
            row["metrics"]["macro_f1"],
            -row["metrics"]["false_edit_rate"],
            -row["commit_threshold"],
        ),
    )
    target_rows = [
        row
        for row in summary["target_validation"]["operation"]["grid"]
        if abs(row["commit_threshold"] - selected["commit_threshold"]) < 1e-12
    ]
    if len(target_rows) != 1:
        raise ValueError("calibration threshold is unavailable on target validation")
    return {"calibration": selected, "target_validation": target_rows[0]}


def _aggregate(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    result = {"seeds": len(rows)}
    for metric in METRICS:
        values = [row[split]["metrics"][metric] for row in rows]
        result[metric] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--false-edit-caps", default="0.05,0.10")
    parser.add_argument("--seeds", required=True)
    args = parser.parse_args()

    caps = tuple(float(value) for value in args.false_edit_caps.split(","))
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(seeds) != len(summaries):
        raise ValueError("--seeds must contain one value per summary")
    if len(set(seeds)) != len(seeds):
        raise ValueError("summary seeds must be unique")
    if any(summary.get("target_validation") is None for summary in summaries):
        raise ValueError("all summaries must have passed the target-validation gate")

    policies = {}
    csv_rows = []
    for cap in caps:
        selected_rows = []
        for seed, summary, source in zip(seeds, summaries, args.summaries, strict=True):
            selected = _select_at_cap(summary, cap)
            row = {"seed": seed, "source": str(source), **selected}
            selected_rows.append(row)
            flat = {
                "false_edit_cap": cap,
                "seed": seed,
                "source": str(source),
                "commit_threshold": selected["calibration"]["commit_threshold"],
            }
            for split in ("calibration", "target_validation"):
                for metric in METRICS:
                    flat[f"{split}_{metric}"] = selected[split]["metrics"][metric]
            csv_rows.append(flat)
        policies[str(cap)] = {
            "selection_split": "atlanta_calibration",
            "target_split": "chicago_houston_validation",
            "false_edit_cap": cap,
            "per_seed": selected_rows,
            "calibration": _aggregate(selected_rows, "calibration"),
            "target_validation": _aggregate(selected_rows, "target_validation"),
        }

    payload = {
        "schema_version": "muno21-sparse-change-three-seed-aggregate-v1",
        "seeds": seeds,
        "frozen_test_access": False,
        "policies": policies,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
