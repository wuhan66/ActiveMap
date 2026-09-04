#!/usr/bin/env python3
"""Task-grouped paired bootstrap for executable Agent map writebacks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

HIGHER_IS_BETTER = (
    "raster_iou",
    "raster_iou_gain",
    "added_change_iou",
    "removed_change_iou",
    "added_polygon_iou",
    "removed_polygon_iou",
    "vector_replay_iou",
    "vector_delta_topology_valid",
)
LOWER_IS_BETTER = ("component_count_absolute_error",)
UTILITY_V2_HIGHER_IS_BETTER = (
    "episode_utility_v2_balanced",
    "episode_utility_v2_safety",
    "episode_utility_v2_cost_aware",
)
SAFETY_COST_LOWER_IS_BETTER = (
    "false_edit",
    "missed_edit",
    "wrong_edit",
    "spent_cost",
)


def _load(path: Path) -> dict[tuple[str, float], dict[str, Any]]:
    rows = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        missing = [name for name in (*HIGHER_IS_BETTER, *LOWER_IS_BETTER) if name not in row]
        if missing:
            raise ValueError(f"{path}:{line_number} missing writeback metrics: {missing}")
        key = (str(row["task_id"]), float(row["budget"]))
        if key in rows:
            raise ValueError(f"duplicate writeback key in {path}: {key}")
        rows[key] = row
    if not rows:
        raise ValueError(f"no writeback rows in {path}")
    return rows


def _auc(rows: list[dict[str, Any]], name: str) -> float:
    by_budget: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        by_budget[float(row["budget"])].append(float(row[name]))
    points = [(budget, float(np.mean(values))) for budget, values in sorted(by_budget.items())]
    if len(points) == 1:
        return points[0][1]
    area = sum(
        (right[0] - left[0]) * (left[1] + right[1]) / 2.0 for left, right in pairwise(points)
    )
    return float(area / (points[-1][0] - points[0][0]))


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    result = {
        **{f"{name}_auc": _auc(rows, name) for name in HIGHER_IS_BETTER},
        **{f"{name}_auc": _auc(rows, name) for name in LOWER_IS_BETTER},
    }
    optional_groups = (
        UTILITY_V2_HIGHER_IS_BETTER,
        SAFETY_COST_LOWER_IS_BETTER,
    )
    for names in optional_groups:
        present = [all(name in row for name in names) for row in rows]
        if any(present) and not all(present):
            raise ValueError(f"writeback rows contain a partial metric group: {names}")
        if all(present):
            result.update({f"{name}_auc": _auc(rows, name) for name in names})
    return result


def _grouped_delta_draws(
    baseline: dict[tuple[str, float], dict[str, Any]],
    candidate: dict[tuple[str, float], dict[str, Any]],
    group_keys: dict[str, list[tuple[str, float]]],
    group_ids: list[str],
    sampled_group_counts: np.ndarray,
    metric_names: list[str],
) -> dict[str, np.ndarray]:
    """Compute paired bootstrap deltas after grouping once by AOI and budget."""
    budgets = sorted({key[1] for values in group_keys.values() for key in values})
    budget_index = {budget: index for index, budget in enumerate(budgets)}
    metric_sources = [name.removesuffix("_auc") for name in metric_names]
    sums = np.zeros((len(group_ids), len(budgets), len(metric_names)), dtype=np.float64)
    counts = np.zeros((len(group_ids), len(budgets)), dtype=np.float64)
    for group_index, group_id in enumerate(group_ids):
        for key in group_keys[group_id]:
            column = budget_index[key[1]]
            counts[group_index, column] += 1.0
            for metric_index, source in enumerate(metric_sources):
                sums[group_index, column, metric_index] += float(candidate[key][source]) - float(
                    baseline[key][source]
                )
    if np.any(counts == 0):
        raise ValueError("each bootstrap group must contain every evaluated budget")
    sampled_sums = np.einsum("rg,gbm->rbm", sampled_group_counts, sums)
    sampled_counts = sampled_group_counts @ counts
    budget_means = sampled_sums / sampled_counts[..., None]
    if len(budgets) == 1:
        auc_weights = np.ones(1, dtype=np.float64)
    else:
        positions = np.asarray(budgets, dtype=np.float64)
        intervals = np.diff(positions)
        auc_weights = np.zeros(len(budgets), dtype=np.float64)
        auc_weights[:-1] += intervals / 2.0
        auc_weights[1:] += intervals / 2.0
        auc_weights /= positions[-1] - positions[0]
    draw_matrix = np.einsum("rbm,b->rm", budget_means, auc_weights)
    return {name: draw_matrix[:, index] for index, name in enumerate(metric_names)}


def compare(
    baseline_path: Path,
    candidate_path: Path,
    *,
    bootstrap: int,
    seed: int,
    model_seed: int | None = None,
    evaluation_seed: int | None = None,
    group_key: str = "task_id",
    split: str = "val",
    frozen_test: bool = False,
) -> dict[str, Any]:
    test_assets_read = split == "test"
    if test_assets_read:
        if not frozen_test:
            raise PermissionError("test writeback comparison requires --frozen-test")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    elif frozen_test:
        raise ValueError("--frozen-test is valid only for the test split")
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    if baseline.keys() != candidate.keys():
        raise ValueError("writeback files do not contain identical task-budget keys")
    keys = sorted(baseline)
    for key in keys:
        if baseline[key].get("target") != candidate[key].get("target"):
            raise ValueError(f"paired writeback targets differ for {key}")
        for row in (baseline[key], candidate[key]):
            if row.get("split", "val") != split or row.get(
                "test_assets_read", False
            ) is not test_assets_read:
                raise ValueError(f"paired writeback row is not audited {split} evidence")
    if group_key not in {"task_id", "aoi_id"}:
        raise ValueError("group_key must be task_id or aoi_id")
    task_keys: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for key in keys:
        baseline_group = str(baseline[key].get(group_key, key[0]))
        candidate_group = str(candidate[key].get(group_key, key[0]))
        if baseline_group != candidate_group:
            raise ValueError(f"paired writeback groups differ for {key}")
        task_keys[baseline_group].append(key)
    expected_budgets = {key[1] for key in keys}
    if group_key == "task_id" and any(
        {key[1] for key in values} != expected_budgets for values in task_keys.values()
    ):
        raise ValueError("each task must contain the same budget set")

    baseline_metrics = _metrics([baseline[key] for key in keys])
    candidate_metrics = _metrics([candidate[key] for key in keys])
    if baseline_metrics.keys() != candidate_metrics.keys():
        raise ValueError("baseline and candidate expose different writeback metric sets")
    observed = {name: candidate_metrics[name] - baseline_metrics[name] for name in baseline_metrics}
    rng = np.random.default_rng(seed)
    task_ids = sorted(task_keys)
    sampled_indices = rng.integers(0, len(task_ids), size=(bootstrap, len(task_ids)))
    sampled_counts = np.eye(len(task_ids), dtype=np.float64)[sampled_indices].sum(axis=1)
    draws = _grouped_delta_draws(
        baseline,
        candidate,
        task_keys,
        task_ids,
        sampled_counts,
        list(observed),
    )
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
            "group": f"{group_key} across all contained tasks and budgets",
            "bootstrap": bootstrap,
            "seed": seed,
            "bootstrap_seed": seed,
            "model_seed": model_seed,
            "evaluation_seed": evaluation_seed,
            "budget_coverage": sorted(expected_budgets),
            "higher_is_better": [f"{name}_auc" for name in HIGHER_IS_BETTER],
            "lower_is_better": [f"{name}_auc" for name in LOWER_IS_BETTER],
            "utility_v2_higher_is_better": [
                f"{name}_auc" for name in UTILITY_V2_HIGHER_IS_BETTER
            ],
            "safety_cost_lower_is_better": [
                f"{name}_auc" for name in SAFETY_COST_LOWER_IS_BETTER
            ],
            "episode_utility_v2_available": (
                "episode_utility_v2_balanced_auc" in candidate_metrics
            ),
            "test_assets_read": test_assets_read,
        },
        "sample_count": len(keys),
        "task_count": len({key[0] for key in keys}),
        "group_count": len(task_ids),
        "group_key": group_key,
        "budgets": sorted(expected_budgets),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "paired_delta": intervals,
        "split": split,
        "test_assets_read": test_assets_read,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--group-key", choices=("task_id", "aoi_id"), default="task_id")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--frozen-test", action="store_true")
    args = parser.parse_args()
    result = compare(
        args.baseline,
        args.candidate,
        bootstrap=args.bootstrap,
        seed=args.seed,
        model_seed=args.model_seed,
        evaluation_seed=args.evaluation_seed,
        group_key=args.group_key,
        split=args.split,
        frozen_test=args.frozen_test,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
