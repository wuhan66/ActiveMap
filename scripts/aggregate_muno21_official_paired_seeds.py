#!/usr/bin/env python3
"""Hierarchical bootstrap for seed-paired MUNO21 official metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import stdev
from typing import Any

import numpy as np

from scripts.compare_muno21_official_metrics import PAIRED_METRICS, _load, compare


def paired_paths(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "pair must use LABEL=BASELINE_JSONL=CANDIDATE_JSONL"
        )
    return parts[0], Path(parts[1]), Path(parts[2])


def aggregate(
    pairs: list[tuple[str, Path, Path]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if len(pairs) < 2:
        raise ValueError("hierarchical aggregation requires at least two seed pairs")
    labels = [label for label, _, _ in pairs]
    if len(labels) != len(set(labels)):
        raise ValueError("duplicate seed-pair labels")

    task_ids: list[str] | None = None
    budgets: list[float] | None = None
    seed_results: dict[str, Any] = {}
    metric_rows: dict[str, list[np.ndarray]] = {metric: [] for metric in PAIRED_METRICS}
    error_rows: list[float] = []

    for pair_index, (label, baseline_path, candidate_path) in enumerate(pairs):
        baseline, baseline_errors = _load(baseline_path)
        candidate, candidate_errors = _load(candidate_path)
        if candidate.keys() != baseline.keys():
            raise ValueError(f"{label} has different candidate and baseline support")
        current_tasks = sorted({task_id for task_id, _ in baseline})
        current_budgets = sorted(baseline_errors)
        if task_ids is None:
            task_ids = current_tasks
            budgets = current_budgets
        elif current_tasks != task_ids or current_budgets != budgets:
            raise ValueError(f"{label} does not match cross-seed task-budget support")

        seed_results[label] = compare(
            baseline_path,
            candidate_path,
            repetitions=repetitions,
            seed=seed + pair_index,
        )
        assert task_ids is not None and budgets is not None
        for metric in PAIRED_METRICS:
            metric_rows[metric].append(
                np.asarray(
                    [
                        np.mean(
                            [
                                float(candidate[(task_id, budget)][metric])
                                - float(baseline[(task_id, budget)][metric])
                                for budget in budgets
                            ]
                        )
                        for task_id in task_ids
                    ],
                    dtype=np.float64,
                )
            )
        error_rows.append(
            float(
                np.mean(
                    [
                        candidate_errors[budget] - baseline_errors[budget]
                        for budget in budgets
                    ]
                )
            )
        )

    assert task_ids is not None and budgets is not None
    rng = np.random.default_rng(seed)
    seed_draws = rng.integers(0, len(pairs), size=(repetitions, len(pairs)))
    task_draws = rng.integers(0, len(task_ids), size=(repetitions, len(task_ids)))
    aggregate_metrics = {}
    for metric, rows in metric_rows.items():
        matrix = np.stack(rows)
        draws = np.empty(repetitions, dtype=np.float64)
        for draw_index in range(repetitions):
            draws[draw_index] = matrix[
                seed_draws[draw_index][:, None], task_draws[draw_index][None, :]
            ].mean()
        per_seed = matrix.mean(axis=1)
        aggregate_metrics[metric] = {
            "mean_delta": float(matrix.mean()),
            "seed_standard_deviation": float(stdev(per_seed)),
            "hierarchical_ci95_low": float(np.quantile(draws, 0.025)),
            "hierarchical_ci95_high": float(np.quantile(draws, 0.975)),
            "per_seed_delta": dict(zip(labels, map(float, per_seed), strict=True)),
        }

    return {
        "schema_version": "muno21-official-seed-paired-hierarchical-bootstrap-v1",
        "seed_pair_count": len(pairs),
        "task_count": len(task_ids),
        "budgets": budgets,
        "bootstrap_repetitions": repetitions,
        "seed_results": seed_results,
        "aggregate": {
            "paired_metrics": aggregate_metrics,
            "mean_no_change_error_rate_delta": float(np.mean(error_rows)),
            "no_change_error_seed_standard_deviation": float(stdev(error_rows)),
            "per_seed_no_change_error_rate_delta": dict(
                zip(labels, error_rows, strict=True)
            ),
        },
        "protocol": {
            "pairing": "candidate and baseline are paired by model seed",
            "hierarchical_bootstrap": "resample seed pairs and tasks with replacement",
            "task_delta": "mean over matched budgets before resampling",
            "test_assets_read": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--pair", action="append", type=paired_paths, required=True)
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = aggregate(args.pair, repetitions=args.repetitions, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
