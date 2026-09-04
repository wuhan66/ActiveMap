#!/usr/bin/env python3
"""Build selector/Agent point and quality-cost AUC bundles from rollout traces."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.build_paper_result_bundle import _select_cell, _sha256

SUPPORTED_POINT_METRICS = {
    "mean_quality_cost_utility",
    "regret",
    "mean_cost",
    "terminal_accuracy",
    "false_edit_rate",
    "schema_valid_rate",
    "executable_valid_rate",
    "grounded_tool_recall",
    "belief_action_flip_rate",
}
SUPPORTED_CURVE_METRICS = {"quality_cost_auc"}


def _parse_seed_paths(specifications: list[str] | None) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for specification in specifications or []:
        seed, separator, raw_path = specification.partition("=")
        if not separator or not seed or not raw_path:
            raise ValueError("seed files must use SEED=PATH")
        if seed in result:
            raise ValueError(f"duplicate seed file {seed}")
        result[seed] = Path(raw_path)
    return result


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _index_rollouts(
    path: Path,
    *,
    split: str,
) -> dict[tuple[str, float], dict[str, Any]]:
    result: dict[tuple[str, float], dict[str, Any]] = {}
    for row in _read_rows(path):
        if row.get("split") != split:
            raise ValueError(f"{path}: rollout split mismatch")
        key = (str(row["task_id"]), float(row["budget"]))
        if key in result:
            raise ValueError(f"{path}: duplicate task-budget row {key}")
        result[key] = row
    return result


def _index_validity(
    path: Path,
    *,
    split: str,
) -> dict[tuple[str, float], list[dict[str, Any]]]:
    result: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in _read_rows(path):
        if row.get("split") != split:
            raise ValueError(f"{path}: validity split mismatch")
        key = (str(row["task_id"]), float(row["budget"]))
        result[key].append(row)
    return result


def _normalized_auc(rows: list[dict[str, Any]]) -> float:
    by_budget: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        by_budget[float(row["budget"])].append(float(row["quality_cost_utility"]))
    points = [(budget, float(np.mean(values))) for budget, values in sorted(by_budget.items())]
    if len(points) < 2:
        raise ValueError("quality-cost AUC requires at least two budgets")
    area = sum(
        (right[0] - left[0]) * (left[1] + right[1]) / 2.0
        for left, right in zip(points, points[1:], strict=False)
    )
    return float(area / (points[-1][0] - points[0][0]))


def _point_metrics(
    rows: list[dict[str, Any]],
    *,
    required_metrics: set[str],
    validity_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
    assume_deterministic_validity: bool,
) -> dict[str, float]:
    metrics = {
        "mean_quality_cost_utility": float(
            np.mean([float(row["quality_cost_utility"]) for row in rows])
        ),
        "mean_cost": float(np.mean([float(row["spent_cost"]) for row in rows])),
        "terminal_accuracy": float(np.mean([bool(row["terminal_correct"]) for row in rows])),
        "false_edit_rate": float(np.mean([bool(row["false_edit"]) for row in rows])),
        "grounded_tool_recall": _grounded_tool_recall(rows),
        "belief_action_flip_rate": _belief_action_flip_rate(rows),
    }
    if "regret" in required_metrics:
        if not oracle_rows:
            raise ValueError("regret requires seed-matched oracle rollouts")
        oracle_by_task = {str(row["task_id"]): row for row in oracle_rows}
        if set(oracle_by_task) != {str(row["task_id"]) for row in rows}:
            raise ValueError("oracle and policy tasks do not match")
        metrics["regret"] = float(
            np.mean(
                [
                    float(oracle_by_task[str(row["task_id"])]["quality_cost_utility"])
                    - float(row["quality_cost_utility"])
                    for row in rows
                ]
            )
        )
    if {"schema_valid_rate", "executable_valid_rate"} & required_metrics:
        if validity_rows:
            metrics["schema_valid_rate"] = float(
                np.mean([bool(row["schema_valid"]) for row in validity_rows])
            )
            metrics["executable_valid_rate"] = float(
                np.mean([bool(row["executable"]) for row in validity_rows])
            )
        elif assume_deterministic_validity:
            metrics["schema_valid_rate"] = 1.0
            metrics["executable_valid_rate"] = 1.0
        else:
            raise ValueError(
                "Agent validity metrics require call logs or explicit deterministic mode"
            )
    missing = sorted(required_metrics - set(metrics))
    if missing:
        raise ValueError(f"cannot compute rollout metrics {missing}")
    return {metric: metrics[metric] for metric in required_metrics}


def _grounded_tool_recall(rows: list[dict[str, Any]]) -> float:
    positives = [row for row in rows if bool(row.get("tool_positive_episode", False))]
    if not positives:
        return 0.0
    return float(np.mean([int(row.get("tool_calls", 0)) > 0 for row in positives]))


def _belief_action_flip_rate(rows: list[dict[str, Any]]) -> float:
    calls = sum(int(row.get("tool_calls", 0)) for row in rows)
    flips = sum(int(row.get("tool_action_flips", 0)) for row in rows)
    return float(flips / calls) if calls else 0.0


def build_rollout_bundle(
    registry_path: Path,
    seed_rollout_paths: dict[str, Path],
    *,
    experiment_id: str,
    variant: str | None,
    budget: float | None,
    seed_validity_paths: dict[str, Path] | None = None,
    seed_oracle_paths: dict[str, Path] | None = None,
    assume_deterministic_validity: bool = False,
    frozen_test_ledger: Path | None = None,
) -> dict[str, Any]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    cell = _select_cell(registry, experiment_id, variant, budget)
    if cell["family"] not in {"selector", "agent", "rl"}:
        raise ValueError("rollout bundle requires a selector, agent, or rl cell")
    expected_seeds = set(map(str, cell["seeds"]))
    if set(seed_rollout_paths) != expected_seeds:
        raise ValueError("rollout seeds do not match the registry cell")
    validity_paths = seed_validity_paths or {}
    oracle_paths = seed_oracle_paths or {}
    if validity_paths and set(validity_paths) != expected_seeds:
        raise ValueError("validity seeds do not match the registry cell")
    if oracle_paths and set(oracle_paths) != expected_seeds:
        raise ValueError("oracle seeds do not match the registry cell")

    required_metrics = set(cell["required_metrics"]) | set(cell["primary_metrics"])
    supported = (
        SUPPORTED_CURVE_METRICS
        if cell["cell_kind"] == "curve_summary"
        else SUPPORTED_POINT_METRICS
    )
    unsupported = sorted(required_metrics - supported)
    if unsupported:
        raise ValueError(f"unsupported rollout metrics: {unsupported}")

    seed_order = sorted(expected_seeds)
    rollouts = {
        seed: _index_rollouts(path, split=cell["split"])
        for seed, path in sorted(seed_rollout_paths.items())
    }
    validity = {
        seed: _index_validity(path, split=cell["split"])
        for seed, path in sorted(validity_paths.items())
    }
    oracle = {
        seed: _index_rollouts(path, split=cell["split"])
        for seed, path in sorted(oracle_paths.items())
    }
    expected_budgets = list(map(float, registry["protocol"]["muno21_budgets"]))
    if cell["cell_kind"] == "curve_summary":
        selected_keys = {
            seed: sorted(rollouts[seed]) for seed in seed_order
        }
        for seed, keys in selected_keys.items():
            budgets = {key[1] for key in keys}
            if budgets != set(expected_budgets):
                raise ValueError(f"{seed}: curve rollouts do not cover frozen budgets")
    else:
        if budget is None:
            raise ValueError("budget-point rollout bundle requires --budget")
        selected_keys = {
            seed: sorted(key for key in rollouts[seed] if math.isclose(key[1], budget))
            for seed in seed_order
        }
    task_sets = {
        seed: {key[0] for key in keys} for seed, keys in selected_keys.items()
    }
    task_order = sorted(task_sets[seed_order[0]])
    if not task_order or any(task_sets[seed] != set(task_order) for seed in seed_order):
        raise ValueError("task ids must be non-empty and identical across seeds")
    if cell["cell_kind"] == "curve_summary":
        for seed, keys in selected_keys.items():
            by_task: dict[str, set[float]] = defaultdict(set)
            for task_id, row_budget in keys:
                by_task[task_id].add(row_budget)
            if any(budgets != set(expected_budgets) for budgets in by_task.values()):
                raise ValueError(f"{seed}: every task must contain all frozen budgets")

    def metrics_for(seed: str, sampled_tasks: list[str]) -> dict[str, float]:
        if cell["cell_kind"] == "curve_summary":
            rows = [
                rollouts[seed][(task, row_budget)]
                for task in sampled_tasks
                for row_budget in expected_budgets
            ]
            return {"quality_cost_auc": _normalized_auc(rows)}
        rows = [rollouts[seed][(task, float(budget))] for task in sampled_tasks]
        calls = [
            call
            for task in sampled_tasks
            for call in validity.get(seed, {}).get((task, float(budget)), [])
        ]
        oracle_rows = [
            oracle[seed][(task, float(budget))]
            for task in sampled_tasks
            if seed in oracle
        ]
        return _point_metrics(
            rows,
            required_metrics=required_metrics,
            validity_rows=calls,
            oracle_rows=oracle_rows,
            assume_deterministic_validity=assume_deterministic_validity,
        )

    seed_metrics = {seed: metrics_for(seed, task_order) for seed in seed_order}
    replicates = int(registry["protocol"]["bootstrap_replicates"])
    confidence_level = float(registry["protocol"]["confidence_level"])
    bootstrap_seed = int(registry["protocol"].get("bootstrap_seed", 20260715))
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(task_order), size=(replicates, len(task_order)))
    draws = {metric: [] for metric in required_metrics}
    for replicate in indices:
        sampled_tasks = [task_order[index] for index in replicate]
        per_seed = {seed: metrics_for(seed, sampled_tasks) for seed in seed_order}
        for metric in required_metrics:
            draws[metric].append(float(np.mean([per_seed[seed][metric] for seed in seed_order])))
    alpha = (1.0 - confidence_level) / 2.0
    summaries: dict[str, dict[str, Any]] = {}
    for metric in sorted(required_metrics):
        seed_values = np.asarray(
            [seed_metrics[seed][metric] for seed in seed_order], dtype=np.float64
        )
        bootstrap = np.asarray(draws[metric], dtype=np.float64)
        summaries[metric] = {
            "mean": float(seed_values.mean()),
            "std": float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else 0.0,
            "ci95": [
                float(np.quantile(bootstrap, alpha)),
                float(np.quantile(bootstrap, 1.0 - alpha)),
            ],
            "seed_means": {
                seed: float(seed_metrics[seed][metric]) for seed in seed_order
            },
        }

    source_paths = list(seed_rollout_paths.items())
    source_paths.extend((f"validity:{seed}", path) for seed, path in validity_paths.items())
    source_paths.extend((f"oracle:{seed}", path) for seed, path in oracle_paths.items())
    rows_by_seed = {seed: len(selected_keys[seed]) for seed in seed_order}
    bundle: dict[str, Any] = {
        "schema_version": "activemap-paper-result-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "variant": variant,
        "budget": budget,
        "family": cell["family"],
        "cell_kind": cell["cell_kind"],
        "split": cell["split"],
        "seeds": seed_order,
        "sample_count": sum(rows_by_seed.values()),
        "unit_count": len(task_order),
        "rows_by_seed": rows_by_seed,
        "bootstrap_unit": registry["protocol"]["bootstrap_unit"],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "registry_sha256": _sha256(registry_path),
        "source_observations": [
            {"seed": seed, "path": str(path.resolve()), "sha256": _sha256(path)}
            for seed, path in source_paths
        ],
        "aggregation": {"type": "task_paired_rollout_v1"},
        "metrics": summaries,
    }
    if cell["split"] == "test":
        if frozen_test_ledger is None:
            raise ValueError("test bundles require a frozen test ledger")
        ledger = json.loads(frozen_test_ledger.read_text(encoding="utf-8"))
        if ledger.get("status") != "complete" or ledger.get("returncode") != 0:
            raise ValueError("frozen test ledger is not complete")
        bundle["frozen_test_ledger"] = str(frozen_test_ledger.resolve())
        bundle["frozen_test_ledger_sha256"] = _sha256(frozen_test_ledger)
    elif frozen_test_ledger is not None:
        raise ValueError("validation bundles must not reference a frozen test ledger")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--variant")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--seed-rollouts", action="append", required=True)
    parser.add_argument("--seed-validity", action="append")
    parser.add_argument("--seed-oracle", action="append")
    parser.add_argument("--assume-deterministic-validity", action="store_true")
    parser.add_argument("--frozen-test-ledger", type=Path)
    args = parser.parse_args()
    bundle = build_rollout_bundle(
        args.registry,
        _parse_seed_paths(args.seed_rollouts),
        experiment_id=args.experiment_id,
        variant=args.variant,
        budget=args.budget,
        seed_validity_paths=_parse_seed_paths(args.seed_validity),
        seed_oracle_paths=_parse_seed_paths(args.seed_oracle),
        assume_deterministic_validity=args.assume_deterministic_validity,
        frozen_test_ledger=args.frozen_test_ledger,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "sample_count": bundle["sample_count"]}))


if __name__ == "__main__":
    main()
