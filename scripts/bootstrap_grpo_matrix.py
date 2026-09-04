#!/usr/bin/env python3
"""Hierarchical task-and-seed bootstrap for a GRPO policy matrix."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.compare_agent_rollouts import _load
from scripts.summarize_muno21_grpo_v2 import METRICS, metrics


def parse_run(value: str) -> tuple[str, str, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--run requires VARIANT=SEED=JSONL")
    return parts[0], parts[1], Path(parts[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")

    baseline = _load(args.baseline)
    grouped: dict[str, list[tuple[str, dict[tuple[str, float], dict[str, Any]]]]] = defaultdict(list)
    for variant, seed, path in args.run:
        candidate = _load(path)
        if set(candidate) != set(baseline):
            raise ValueError(f"{path} does not match baseline task-budget support")
        if any(candidate[key]["target"] != baseline[key]["target"] for key in baseline):
            raise ValueError(f"{path} has target labels inconsistent with baseline")
        grouped[variant].append((seed, candidate))

    task_keys: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for key in sorted(baseline):
        task_keys[key[0]].append(key)
    task_ids = np.asarray(sorted(task_keys), dtype=object)
    baseline_full = metrics(list(baseline.values()))

    observed: dict[str, dict[str, float]] = {}
    for variant, runs in grouped.items():
        run_metrics = [metrics(list(rows.values())) for _, rows in runs]
        observed[variant] = {
            metric: float(np.mean([row[metric] for row in run_metrics]) - baseline_full[metric])
            for metric in METRICS
        }

    rng = np.random.default_rng(args.seed)
    draws: dict[str, dict[str, list[float]]] = {
        variant: {metric: [] for metric in METRICS} for variant in grouped
    }
    for _ in range(args.bootstrap):
        sampled_tasks = rng.choice(task_ids, size=len(task_ids), replace=True)
        sampled_keys = [key for task in sampled_tasks for key in task_keys[str(task)]]
        baseline_draw = metrics([baseline[key] for key in sampled_keys])
        for variant, runs in grouped.items():
            sampled_run_indices = rng.integers(0, len(runs), size=len(runs))
            candidate_draws = [
                metrics([runs[index][1][key] for key in sampled_keys])
                for index in sampled_run_indices
            ]
            for metric in METRICS:
                candidate_mean = float(np.mean([row[metric] for row in candidate_draws]))
                draws[variant][metric].append(candidate_mean - baseline_draw[metric])

    variants: dict[str, Any] = {}
    for variant, runs in grouped.items():
        variants[variant] = {
            "seeds": [seed for seed, _ in runs],
            "seed_count": len(runs),
            "paired_delta": {
                metric: {
                    "delta": observed[variant][metric],
                    "ci95_low": float(np.quantile(draws[variant][metric], 0.025)),
                    "ci95_high": float(np.quantile(draws[variant][metric], 0.975)),
                }
                for metric in METRICS
            },
        }
    result = {
        "schema_version": "activemap-grpo-hierarchical-bootstrap-v1",
        "protocol": {
            "task_paired": True,
            "task_resampling": "opaque task_id with all budgets retained",
            "policy_seed_resampling": "with replacement within variant",
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "test_assets_read": False,
        },
        "task_count": len(task_ids),
        "sample_count": len(baseline),
        "baseline": baseline_full,
        "variants": variants,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "hierarchical_bootstrap.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    selected = (
        "terminal_accuracy",
        "macro_f1",
        "false_edit_rate",
        "missed_edit_rate",
        "episode_utility_v2_proxy_balanced_auc",
    )
    lines = [
        "| Variant | Seeds | Metric | Delta vs SFT | 95% CI |",
        "|---|---:|---|---:|---:|",
    ]
    for variant, payload in variants.items():
        for metric in selected:
            item = payload["paired_delta"][metric]
            lines.append(
                f"| {variant} | {payload['seed_count']} | {metric} | "
                f"{item['delta']:+.6f} | [{item['ci95_low']:+.6f}, {item['ci95_high']:+.6f}] |"
            )
    (args.output / "hierarchical_bootstrap.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
