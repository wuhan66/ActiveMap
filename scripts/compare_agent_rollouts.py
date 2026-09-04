#!/usr/bin/env python3
"""Task-grouped paired bootstrap comparison for Agent closed-loop rollouts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[tuple[str, float], dict[str, Any]]:
    rows: dict[tuple[str, float], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        key = (str(row["task_id"]), float(row["budget"]))
        if key in rows:
            raise ValueError(f"duplicate rollout key in {path}: {key}")
        rows[key] = row
    if not rows:
        raise ValueError(f"no rollout rows in {path}")
    return rows


def _normalized_auc(rows: list[dict[str, Any]], key: str) -> float:
    by_budget: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        by_budget[float(row["budget"])].append(float(row[key]))
    points = [(budget, float(np.mean(values))) for budget, values in sorted(by_budget.items())]
    if len(points) == 1:
        return points[0][1]
    area = sum(
        (right[0] - left[0]) * (left[1] + right[1]) / 2.0
        for left, right in pairwise(points)
    )
    return area / (points[-1][0] - points[0][0])


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics = {
        "terminal_accuracy": float(np.mean([row["terminal_correct"] for row in rows])),
        "false_edit_rate": float(np.mean([row["false_edit"] for row in rows])),
        "missed_edit_rate": float(np.mean([row["missed_edit"] for row in rows])),
        "mean_cost": float(np.mean([row["spent_cost"] for row in rows])),
        "mean_acquisitions": float(np.mean([row["acquisitions"] for row in rows])),
        "joint_utility_auc": _normalized_auc(rows, "joint_utility"),
    }
    if all("quality_cost_utility" in row for row in rows):
        metrics.update(
            {
                "final_evidence_quality": float(
                    np.mean([row["final_evidence_quality"] for row in rows])
                ),
                "evidence_quality_gain": float(
                    np.mean([row["evidence_quality_gain"] for row in rows])
                ),
                "quality_cost_utility_auc": _normalized_auc(
                    rows, "quality_cost_utility"
                ),
            }
        )
    if all("tool_calls" in row for row in rows):
        metrics.update(
            {
                "mean_tool_calls": float(np.mean([row["tool_calls"] for row in rows])),
                "mean_tool_cost": float(np.mean([row["tool_cost"] for row in rows])),
                "mean_tool_belief_l1_delta": float(
                    np.mean([row["mean_tool_belief_l1_delta"] for row in rows])
                ),
                "mean_tool_action_flips": float(
                    np.mean([row["tool_action_flips"] for row in rows])
                ),
                "terminal_edit_flip_rate": float(
                    np.mean([row["terminal_edit_changed_after_tools"] for row in rows])
                ),
            }
        )
    for suffix in ("balanced", "safety", "cost_aware"):
        writeback_key = f"episode_utility_v2_{suffix}"
        proxy_key = f"episode_utility_v2_proxy_{suffix}"
        if all(writeback_key in row for row in rows):
            metrics[f"episode_utility_v2_{suffix}_auc"] = _normalized_auc(
                rows, writeback_key
            )
        elif all(proxy_key in row for row in rows):
            metrics[f"episode_utility_v2_proxy_{suffix}_auc"] = _normalized_auc(
                rows, proxy_key
            )
    return metrics


def compare(
    baseline_path: Path,
    candidate_path: Path,
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    if set(baseline) != set(candidate):
        raise ValueError("rollout files do not contain identical task-budget keys")
    for key in baseline:
        if baseline[key]["target"] != candidate[key]["target"]:
            raise ValueError(f"paired rollout targets differ for {key}")

    model_seed_values = {
        int(row["evaluation_seed"])
        for row in list(baseline.values()) + list(candidate.values())
        if row.get("evaluation_seed") is not None
    }
    if len(model_seed_values) != 1:
        raise ValueError(
            "paired rollout files must contain one shared evaluation_seed"
        )
    model_seed = next(iter(model_seed_values))

    keys = sorted(baseline)
    task_keys: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for key in keys:
        task_keys[key[0]].append(key)
    expected_budgets = {key[1] for key in keys}
    budget_coverage = {
        str(budget): sum(1 for key in keys if key[1] == budget)
        for budget in sorted(expected_budgets)
    }
    incomplete_task_count = sum(
        1
        for values in task_keys.values()
        if {key[1] for key in values} != expected_budgets
    )

    baseline_metrics = _metrics([baseline[key] for key in keys])
    candidate_metrics = _metrics([candidate[key] for key in keys])
    observed = {
        name: candidate_metrics[name] - baseline_metrics[name]
        for name in baseline_metrics
    }
    task_ids = sorted(task_keys)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in observed}
    for _ in range(bootstrap):
        sampled_tasks = rng.choice(task_ids, size=len(task_ids), replace=True)
        sampled_keys = [key for task in sampled_tasks for key in task_keys[str(task)]]
        left = _metrics([baseline[key] for key in sampled_keys])
        right = _metrics([candidate[key] for key in sampled_keys])
        for name in draws:
            draws[name].append(right[name] - left[name])

    intervals = {
        name: {
            "delta": observed[name],
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for name, values in draws.items()
    }
    return {
        "protocol": {
            "paired": True,
            "group": "opaque task_id across all budgets",
            "partial_budget_groups_allowed": True,
            "incomplete_task_count": incomplete_task_count,
            "budget_coverage": budget_coverage,
            "bootstrap": bootstrap,
            "seed": seed,
            "bootstrap_seed": seed,
            "model_seed": model_seed,
            "test_assets_read": False,
        },
        "sample_count": len(keys),
        "task_count": len(task_ids),
        "budgets": sorted(expected_budgets),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "paired_delta": intervals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    result = compare(
        args.baseline,
        args.candidate,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
