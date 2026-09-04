#!/usr/bin/env python3
"""Aggregate official MUNO21 paired metrics across model seeds and tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import stdev
from typing import Any

import numpy as np

from scripts.compare_muno21_official_metrics import PAIRED_METRICS, _load, compare


def labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("candidate must use LABEL=JSONL")
    return label, Path(raw_path)


def aggregate(
    baseline_path: Path,
    candidates: list[tuple[str, Path]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if len(candidates) < 2:
        raise ValueError("cross-seed aggregation requires at least two candidates")
    baseline, baseline_errors = _load(baseline_path)
    labels = [label for label, _ in candidates]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate seed labels")

    task_ids = sorted({task_id for task_id, _ in baseline})
    budgets = sorted(baseline_errors)
    seed_results = {}
    deltas_by_metric: dict[str, list[np.ndarray]] = {
        metric: [] for metric in PAIRED_METRICS
    }
    error_deltas = []
    for label, candidate_path in candidates:
        candidate, candidate_errors = _load(candidate_path)
        if candidate.keys() != baseline.keys():
            raise ValueError(f"{label} has different task-budget support")
        if candidate_errors.keys() != baseline_errors.keys():
            raise ValueError(f"{label} has different budget support")
        seed_results[label] = compare(
            baseline_path,
            candidate_path,
            repetitions=repetitions,
            seed=seed,
        )
        for metric in PAIRED_METRICS:
            task_deltas = np.asarray(
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
            deltas_by_metric[metric].append(task_deltas)
        error_deltas.append(
            np.mean(
                [
                    candidate_errors[budget] - baseline_errors[budget]
                    for budget in budgets
                ]
            )
        )

    rng = np.random.default_rng(seed)
    seed_indices = rng.integers(0, len(candidates), size=(repetitions, len(candidates)))
    task_indices = rng.integers(0, len(task_ids), size=(repetitions, len(task_ids)))
    aggregate_metrics = {}
    for metric, seed_arrays in deltas_by_metric.items():
        matrix = np.stack(seed_arrays)
        draws = np.empty(repetitions, dtype=np.float64)
        for index in range(repetitions):
            draws[index] = matrix[
                seed_indices[index][:, None],
                task_indices[index][None, :],
            ].mean()
        seed_means = matrix.mean(axis=1)
        aggregate_metrics[metric] = {
            "mean_delta": float(matrix.mean()),
            "seed_standard_deviation": float(stdev(seed_means)),
            "hierarchical_ci95_low": float(np.quantile(draws, 0.025)),
            "hierarchical_ci95_high": float(np.quantile(draws, 0.975)),
            "per_seed_delta": {
                label: float(value) for label, value in zip(labels, seed_means)
            },
        }

    return {
        "schema_version": "muno21-official-three-seed-aggregate-v1",
        "seed_count": len(candidates),
        "task_count": len(task_ids),
        "budgets": budgets,
        "bootstrap_repetitions": repetitions,
        "seed_results": seed_results,
        "aggregate": {
            "paired_metrics": aggregate_metrics,
            "mean_no_change_error_rate_delta": float(np.mean(error_deltas)),
            "no_change_error_seed_standard_deviation": float(stdev(error_deltas)),
            "per_seed_no_change_error_rate_delta": {
                label: float(value) for label, value in zip(labels, error_deltas)
            },
        },
        "protocol": {
            "hierarchical_bootstrap": "resample model seeds and tasks with replacement",
            "task_delta": "mean over identical budgets before resampling",
            "test_assets_read": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", action="append", type=labeled_path, required=True)
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = aggregate(
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
