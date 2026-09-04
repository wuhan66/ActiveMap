#!/usr/bin/env python3
"""Paired comparison of structured MUNO21 official validation metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PAIRED_METRICS = ("apls_improvement", "pixel_f1_improvement")


def _load(path: Path) -> tuple[dict[tuple[str, float], dict[str, Any]], dict[float, float]]:
    changes = {}
    errors = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        budget = float(row["budget"])
        if row["official_unit"] == "change_scenario":
            changes[(str(row["task_id"]), budget)] = row
        elif row["official_unit"] == "official_nochange_aggregate":
            errors[budget] = float(row["no_change_error_rate"])
    if not changes or not errors:
        raise ValueError(f"incomplete official metrics: {path}")
    return changes, errors


def compare(
    baseline_path: Path,
    candidate_path: Path,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    baseline, baseline_errors = _load(baseline_path)
    candidate, candidate_errors = _load(candidate_path)
    if baseline.keys() != candidate.keys():
        raise ValueError("official comparisons require identical task-budget support")
    if baseline_errors.keys() != candidate_errors.keys():
        raise ValueError("official comparisons require identical budget support")

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for key in sorted(baseline):
        grouped[key[0]].append((baseline[key], candidate[key]))
    task_ids = sorted(grouped)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(task_ids), size=(repetitions, len(task_ids)))
    intervals = {}
    for metric in PAIRED_METRICS:
        deltas = np.asarray(
            [
                np.mean(
                    [float(candidate_row[metric]) - float(baseline_row[metric])
                     for baseline_row, candidate_row in grouped[task_id]]
                )
                for task_id in task_ids
            ],
            dtype=np.float64,
        )
        draws = deltas[indices].mean(axis=1)
        intervals[metric] = {
            "delta": float(deltas.mean()),
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
        }

    error_deltas = {
        str(budget): candidate_errors[budget] - baseline_errors[budget]
        for budget in sorted(baseline_errors)
    }
    return {
        "schema_version": "muno21-official-paired-comparison-v1",
        "task_count": len(task_ids),
        "budgets": sorted(baseline_errors),
        "bootstrap_repetitions": repetitions,
        "paired_task_mean_delta": intervals,
        "no_change_error_rate_delta": error_deltas,
        "mean_no_change_error_rate_delta": float(np.mean(list(error_deltas.values()))),
        "interpretation": {
            "positive_apls_and_pixel_f1_delta_is_better": True,
            "negative_no_change_error_rate_delta_is_better": True,
            "test_assets_read": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = compare(
        args.baseline,
        args.candidate,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
