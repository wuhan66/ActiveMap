#!/usr/bin/env python3
"""Aggregate immutable Habitat online-acquisition pilot summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

import numpy as np

METRICS = (
    "post_initial_acquisitions",
    "initial_known_cell_fraction",
    "final_known_cell_fraction",
    "post_initial_known_cell_gain",
    "new_known_cells_per_post_initial_acquisition",
    "reference_covered_cells_per_sensor_call",
    "final_reference_map_quality",
    "reference_known_coverage",
    "reference_unknown_rate",
    "occupied_iou",
    "free_iou",
    "balanced_iou",
    "obstacle_precision",
    "obstacle_recall",
    "false_free_rate",
    "false_obstacle_rate",
    "success",
    "spl",
    "collision_count",
)

PAIRED_REFERENCE_METRICS = (
    "final_reference_map_quality",
    "reference_known_coverage",
    "occupied_iou",
    "free_iou",
    "balanced_iou",
    "false_free_rate",
    "false_obstacle_rate",
    "reference_covered_cells_per_sensor_call",
)

POLICY_CONTRAST_METRICS = (
    "post_initial_acquisitions",
    "final_reference_map_quality",
    "reference_known_coverage",
    "occupied_iou",
    "free_iou",
    "balanced_iou",
    "false_free_rate",
    "false_obstacle_rate",
    "reference_covered_cells_per_sensor_call",
)


def _label(payload: dict[str, object]) -> str:
    policy = payload.get("policy")
    if policy in {"unknown_gate", "novelty_gate"}:
        score = payload.get("min_unknown_score")
        if policy == "novelty_gate":
            score = payload.get("min_novelty_score")
        if not isinstance(score, int):
            raise ValueError(f"{policy} summary is missing its threshold")
        return f"{policy}@{score}"
    if not isinstance(policy, str):
        raise ValueError("summary is missing policy")
    return policy


def load_grouped(run_root: Path) -> dict[str, list[dict[str, object]]]:
    summaries = sorted(run_root.glob("*/summary.json"))
    if not summaries:
        raise ValueError(f"no online acquisition summaries under {run_root}")
    grouped: dict[str, list[dict[str, object]]] = {}
    for path in summaries:
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema_version = payload.get("schema_version")
        if schema_version in {
            "activemap-habitat-online-acquisition-aggregate-v1",
            "activemap-habitat-online-acquisition-aggregate-v2",
        }:
            continue
        if schema_version != "activemap-habitat-online-acquisition-pilot-v1":
            raise ValueError(f"unsupported summary schema: {path}")
        if payload.get("development_only") is not True:
            raise ValueError(f"summary is not marked development_only: {path}")
        grouped.setdefault(_label(payload), []).append(payload)
    return grouped


def aggregate(run_root: Path) -> list[dict[str, object]]:
    grouped = load_grouped(run_root)
    rows: list[dict[str, object]] = []
    for label, payloads in sorted(grouped.items()):
        row: dict[str, object] = {"policy": label, "episode_count": len(payloads)}
        for metric in METRICS:
            values = [payload.get(metric) for payload in payloads]
            numeric = [float(value) for value in values if isinstance(value, int | float)]
            row[metric] = fmean(numeric) if numeric else None
        rows.append(row)
    return rows


def paired_bootstrap(
    run_root: Path,
    *,
    repetitions: int = 20_000,
    seed: int = 20260830,
) -> list[dict[str, object]]:
    """Pair each gated policy with acquire-all by trajectory seed."""
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    grouped = load_grouped(run_root)
    reference = grouped.get("acquire_all")
    if reference is None:
        return []

    def by_seed(payloads: list[dict[str, object]]) -> dict[int, dict[str, object]]:
        result: dict[int, dict[str, object]] = {}
        for payload in payloads:
            row_seed = payload.get("seed")
            if not isinstance(row_seed, int) or row_seed in result:
                raise ValueError("paired summaries require unique integer seeds")
            result[row_seed] = payload
        return result

    reference_by_seed = by_seed(reference)
    rng = np.random.default_rng(seed)
    reports: list[dict[str, object]] = []
    for label, payloads in sorted(grouped.items()):
        if label == "acquire_all":
            continue
        policy_by_seed = by_seed(payloads)
        if set(policy_by_seed) != set(reference_by_seed):
            raise ValueError(f"seed mismatch between acquire_all and {label}")
        seeds = sorted(reference_by_seed)
        calls_saved = np.asarray(
            [
                float(reference_by_seed[row_seed]["post_initial_acquisitions"])
                - float(policy_by_seed[row_seed]["post_initial_acquisitions"])
                for row_seed in seeds
            ],
            dtype=np.float64,
        )
        indices = rng.integers(0, len(seeds), size=(repetitions, len(seeds)))

        def statistic(
            values: np.ndarray, bootstrap_indices: np.ndarray = indices
        ) -> dict[str, object]:
            bootstrapped = values[bootstrap_indices].mean(axis=1)
            return {
                "mean": float(values.mean()),
                "ci95": [
                    float(np.quantile(bootstrapped, 0.025)),
                    float(np.quantile(bootstrapped, 0.975)),
                ],
                "per_seed": values.tolist(),
            }

        report: dict[str, object] = {
            "policy": label,
            "episode_count": len(seeds),
            "calls_saved_vs_acquire_all": statistic(calls_saved),
        }
        for metric in PAIRED_REFERENCE_METRICS:
            values = [policy_by_seed[row_seed].get(metric) for row_seed in seeds]
            if all(isinstance(value, int | float) for value in values):
                report[metric] = statistic(np.asarray(values, dtype=np.float64))
        # Retain the original public key for old consumers.
        if "final_reference_map_quality" in report:
            report["reference_map_quality"] = report["final_reference_map_quality"]
        reports.append(report)
    return reports


def paired_policy_contrast(
    run_root: Path,
    *,
    candidate: str = "novelty_gate@8",
    baseline: str = "unknown_gate@15",
    repetitions: int = 20_000,
    seed: int = 20260830,
) -> dict[str, object] | None:
    """Bootstrap candidate-minus-baseline differences on matched trajectory seeds."""
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    grouped = load_grouped(run_root)
    if candidate not in grouped or baseline not in grouped:
        return None

    def by_seed(payloads: list[dict[str, object]]) -> dict[int, dict[str, object]]:
        result: dict[int, dict[str, object]] = {}
        for payload in payloads:
            row_seed = payload.get("seed")
            if not isinstance(row_seed, int) or row_seed in result:
                raise ValueError("policy contrast requires unique integer seeds")
            result[row_seed] = payload
        return result

    candidate_by_seed = by_seed(grouped[candidate])
    baseline_by_seed = by_seed(grouped[baseline])
    if set(candidate_by_seed) != set(baseline_by_seed):
        raise ValueError(f"seed mismatch between {candidate} and {baseline}")
    seeds = sorted(candidate_by_seed)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(seeds), size=(repetitions, len(seeds)))
    metrics: dict[str, object] = {}
    for metric in POLICY_CONTRAST_METRICS:
        candidate_values = [candidate_by_seed[row_seed].get(metric) for row_seed in seeds]
        baseline_values = [baseline_by_seed[row_seed].get(metric) for row_seed in seeds]
        if not all(
            isinstance(value, int | float) for value in [*candidate_values, *baseline_values]
        ):
            continue
        differences = np.asarray(candidate_values, dtype=np.float64) - np.asarray(
            baseline_values, dtype=np.float64
        )
        bootstrapped = differences[indices].mean(axis=1)
        metrics[metric] = {
            "mean_difference": float(differences.mean()),
            "ci95": [
                float(np.quantile(bootstrapped, 0.025)),
                float(np.quantile(bootstrapped, 0.975)),
            ],
            "per_seed_difference": differences.tolist(),
        }
    return {
        "candidate": candidate,
        "baseline": baseline,
        "difference_definition": "candidate_minus_baseline",
        "episode_count": len(seeds),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = aggregate(args.run_root)
    paired = paired_bootstrap(
        args.run_root,
        repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed,
    )
    contrast = paired_policy_contrast(
        args.run_root,
        repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed,
    )
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "activemap-habitat-online-acquisition-aggregate-v2",
                "run_root": str(args.run_root.resolve()),
                "development_only": True,
                "rows": rows,
                "paired_bootstrap": paired,
                "policy_contrast": contrast,
                "bootstrap_repetitions": args.bootstrap_repetitions,
                "bootstrap_seed": args.bootstrap_seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (args.output / "summary.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
