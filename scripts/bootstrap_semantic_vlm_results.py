#!/usr/bin/env python3
"""Task-grouped paired bootstrap for fixed-seed visual-policy evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from activemap.agent.vlm_evaluation import multiclass_metrics
from activemap.models import EditOperation


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty evaluation records: {path}")
    required = {
        "example_id",
        "task_id",
        "split",
        "target_operation",
        "policy_operation",
        "baseline_operation",
        "baseline_utility",
        "policy_utility",
    }
    for row in rows:
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"evaluation row missing fields {missing}: {path}")
        if row["split"] != "val":
            raise ValueError(f"paired bootstrap only permits validation rows: {path}")
    return rows


def _false_edit_rate(rows: list[dict[str, Any]], prediction_key: str) -> float:
    keep = EditOperation.KEEP.value
    eligible = [row for row in rows if row["target_operation"] == keep]
    if not eligible:
        return math.nan
    return sum(row[prediction_key] != keep for row in eligible) / len(eligible)


def _missed_edit_rate(rows: list[dict[str, Any]], prediction_key: str) -> float:
    keep = EditOperation.KEEP.value
    eligible = [row for row in rows if row["target_operation"] != keep]
    if not eligible:
        return math.nan
    return sum(row[prediction_key] == keep for row in eligible) / len(eligible)


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    labels = [operation.value for operation in EditOperation]
    targets = [str(row["target_operation"]) for row in rows]
    policy = [str(row["policy_operation"]) for row in rows]
    baseline = [str(row["baseline_operation"]) for row in rows]
    policy_f1 = float(multiclass_metrics(targets, policy, labels)["macro_f1"])
    baseline_f1 = float(multiclass_metrics(targets, baseline, labels)["macro_f1"])
    policy_utility = statistics.fmean(float(row["policy_utility"]) for row in rows)
    baseline_utility = statistics.fmean(float(row["baseline_utility"]) for row in rows)
    policy_false = _false_edit_rate(rows, "policy_operation")
    baseline_false = _false_edit_rate(rows, "baseline_operation")
    policy_missed = _missed_edit_rate(rows, "policy_operation")
    baseline_missed = _missed_edit_rate(rows, "baseline_operation")
    return {
        "policy_operation_macro_f1": policy_f1,
        "baseline_operation_macro_f1": baseline_f1,
        "operation_macro_f1_delta": policy_f1 - baseline_f1,
        "policy_mean_utility": policy_utility,
        "baseline_mean_utility": baseline_utility,
        "mean_utility_delta": policy_utility - baseline_utility,
        "policy_false_edit_rate": policy_false,
        "baseline_false_edit_rate": baseline_false,
        "false_edit_rate_delta": policy_false - baseline_false,
        "policy_missed_edit_rate": policy_missed,
        "baseline_missed_edit_rate": baseline_missed,
        "missed_edit_rate_delta": policy_missed - baseline_missed,
        "tool_call_rate": statistics.fmean(
            float(bool(row.get("predicted_use_tool"))) for row in rows
        ),
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap(
    paths: dict[int, Path], *, repetitions: int, bootstrap_seed: int
) -> dict[str, Any]:
    if len(paths) < 2 or repetitions <= 0:
        raise ValueError("at least two fixed seeds and positive repetitions are required")
    rows_by_seed = {seed: _load(path) for seed, path in sorted(paths.items())}
    reference_seed = next(iter(rows_by_seed))
    reference = {
        (str(row["task_id"]), str(row["example_id"])): str(row["target_operation"])
        for row in rows_by_seed[reference_seed]
    }
    for seed, rows in rows_by_seed.items():
        protocol = {
            (str(row["task_id"]), str(row["example_id"])): str(row["target_operation"])
            for row in rows
        }
        if protocol != reference or len(protocol) != len(rows):
            raise ValueError(f"seed {seed} does not share the paired validation protocol")
    task_ids = sorted({task_id for task_id, _ in reference})
    grouped = {
        seed: {
            task_id: [row for row in rows if str(row["task_id"]) == task_id]
            for task_id in task_ids
        }
        for seed, rows in rows_by_seed.items()
    }
    observed_by_seed = {seed: _metrics(rows) for seed, rows in rows_by_seed.items()}
    metric_names = list(next(iter(observed_by_seed.values())))
    observed_mean = {
        name: statistics.fmean(values[name] for values in observed_by_seed.values())
        for name in metric_names
    }
    rng = random.Random(bootstrap_seed)
    distributions = {name: [] for name in metric_names}
    for _ in range(repetitions):
        sampled_tasks = rng.choices(task_ids, k=len(task_ids))
        replicate_by_seed = {}
        for seed in rows_by_seed:
            sampled_rows = [
                row
                for task_id in sampled_tasks
                for row in grouped[seed][task_id]
            ]
            replicate_by_seed[seed] = _metrics(sampled_rows)
        for name in metric_names:
            finite_values = [
                metrics[name]
                for metrics in replicate_by_seed.values()
                if math.isfinite(metrics[name])
            ]
            value = statistics.fmean(finite_values) if finite_values else math.nan
            distributions[name].append(value)
    intervals = {
        name: {
            "observed_fixed_seed_mean": observed_mean[name],
            "ci95_low": _quantile(values, 0.025),
            "ci95_high": _quantile(values, 0.975),
        }
        for name, values in distributions.items()
    }
    intervals["operation_macro_f1_delta"]["bootstrap_probability_gt_zero"] = sum(
        math.isfinite(value) and value > 0.0
        for value in distributions["operation_macro_f1_delta"]
    ) / max(
        sum(math.isfinite(value) for value in distributions["operation_macro_f1_delta"]),
        1,
    )
    intervals["mean_utility_delta"]["bootstrap_probability_gt_zero"] = sum(
        math.isfinite(value) and value > 0.0
        for value in distributions["mean_utility_delta"]
    ) / max(
        sum(math.isfinite(value) for value in distributions["mean_utility_delta"]),
        1,
    )
    intervals["false_edit_rate_delta"]["bootstrap_probability_le_0p02"] = sum(
        math.isfinite(value) and value <= 0.02
        for value in distributions["false_edit_rate_delta"]
    ) / max(
        sum(math.isfinite(value) for value in distributions["false_edit_rate_delta"]),
        1,
    )
    return {
        "schema_version": "semantic-vlm-task-paired-bootstrap-v1",
        "model_training_seeds": sorted(paths),
        "seed_count": len(paths),
        "sample_count_per_seed": len(reference),
        "task_count": len(task_ids),
        "repetitions": repetitions,
        "bootstrap_seed": bootstrap_seed,
        "grouping_unit": "task_id",
        "shared_resample_indices_across_model_seeds": True,
        "per_seed_observed": observed_by_seed,
        "fixed_seed_mean_intervals": intervals,
        "sources": [
            {"seed": seed, "path": str(path.resolve()), "sha256": _sha256(path)}
            for seed, path in sorted(paths.items())
        ],
        "test_assets_read": False,
    }


def _seed_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise ValueError("records must use SEED=PATH")
    seed, path = value.split("=", 1)
    return int(seed), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--records", action="append", required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    args = parser.parse_args()
    parsed = dict(_seed_path(value) for value in args.records)
    if len(parsed) != len(args.records):
        raise ValueError("duplicate model-training seed")
    result = paired_bootstrap(
        parsed,
        repetitions=args.repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
