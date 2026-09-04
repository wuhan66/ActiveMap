#!/usr/bin/env python3
"""Aggregate vector-writeback metrics with task-grouped paired bootstrap."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

METRICS = [
    "raster_iou",
    "prior_raster_iou",
    "raster_iou_gain",
    "added_change_iou",
    "removed_change_iou",
    "added_polygon_iou",
    "removed_polygon_iou",
    "vector_replay_iou",
    "component_count_absolute_error",
    "prior_component_count_absolute_error",
    "vector_delta_topology_valid",
]


def _read(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty writeback result: {path}")
    if any(not row.get("source_example_id") for row in rows):
        raise ValueError("writeback rows lack source_example_id provenance")
    return rows


def grouped_bootstrap(
    rows: list[dict[str, Any]], metric: str, *, repetitions: int, seed: int
) -> dict[str, float]:
    by_task: dict[str, list[float]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_id"]), []).append(float(row[metric]))
    tasks = sorted(by_task)
    task_means = np.asarray([np.mean(by_task[task]) for task in tasks], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected = rng.integers(0, len(tasks), size=len(tasks))
        samples[index] = float(np.mean(task_means[selected]))
    return {
        "task_group_count": len(tasks),
        "mean": float(np.mean(task_means)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "repetitions": repetitions,
    }


def aggregate(
    paths: dict[int, Path], *, repetitions: int, bootstrap_seed: int
) -> dict[str, Any]:
    rows_by_seed = {seed: _read(path) for seed, path in paths.items()}
    identities = {
        seed: {(str(row["task_id"]), str(row["source_example_id"])) for row in rows}
        for seed, rows in rows_by_seed.items()
    }
    reference = next(iter(identities.values()))
    if any(values != reference for values in identities.values()):
        raise ValueError("writeback seeds do not contain identical paired examples")
    per_seed = []
    for seed, rows in rows_by_seed.items():
        means = {
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in METRICS
        }
        gates = {
            "nonnegative_raster_gain": means["raster_iou_gain"] >= 0.0,
            "topology_valid": means["vector_delta_topology_valid"] >= 0.99,
            "vector_replay": means["vector_replay_iou"] >= 0.98,
            "fragmentation_not_worse": means["component_count_absolute_error"]
            <= means["prior_component_count_absolute_error"],
        }
        per_seed.append(
            {
                "model_training_seed": seed,
                "sample_count": len(rows),
                "metrics": means,
                "raster_gain_task_bootstrap": grouped_bootstrap(
                    rows,
                    "raster_iou_gain",
                    repetitions=repetitions,
                    seed=bootstrap_seed + seed,
                ),
                "passed": all(gates.values()),
                "gates": gates,
                "failed_gates": [name for name, passed in gates.items() if not passed],
                "source": str(paths[seed].resolve()),
            }
        )
    aggregate_metrics = {}
    for metric in METRICS:
        values = [row["metrics"][metric] for row in per_seed]
        aggregate_metrics[metric] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    passed = all(row["passed"] for row in per_seed)
    return {
        "schema_version": "semantic-vlm-writeback-seed-aggregate-v1",
        "model_training_seeds": list(paths),
        "paired_example_count": len(reference),
        "per_seed": per_seed,
        "aggregate_metrics": aggregate_metrics,
        "all_writeback_gates_passed": passed,
        "rl_ready": passed,
        "paper_claim_ready": False,
        "test_assets_read": False,
        "bootstrap_protocol": "task-grouped-within-seed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    args = parser.parse_args()
    if args.repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions are required")
    paths = {}
    for value in args.result:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in paths:
            raise ValueError(f"duplicate seed: {seed}")
        paths[seed] = Path(path_text)
    result = aggregate(
        paths, repetitions=args.repetitions, bootstrap_seed=args.bootstrap_seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
