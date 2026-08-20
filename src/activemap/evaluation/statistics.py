"""AOI-grouped confidence intervals and paired method comparisons."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from activemap.evaluation.update import UpdatePrediction, evaluate_updates

MetricFunction = Callable[[list[UpdatePrediction]], dict[str, Any]]


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value is not None
        and key not in {"sample_count", "aoi_count"}
    }


def grouped_bootstrap_intervals(
    records: Iterable[UpdatePrediction],
    *,
    iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 20260710,
    metric_fn: MetricFunction = evaluate_updates,
) -> dict[str, dict[str, float]]:
    """Resample complete AOIs so temporal frames never become independent units."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    grouped: dict[str, list[UpdatePrediction]] = defaultdict(list)
    for record in records:
        grouped[record.aoi_id].append(record)
    group_ids = sorted(grouped)
    if not group_ids:
        raise ValueError("bootstrap requires at least one AOI")
    point = _numeric_metrics(metric_fn([item for group in grouped.values() for item in group]))
    draws: dict[str, list[float]] = {name: [] for name in point}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        sampled_ids = rng.choice(group_ids, size=len(group_ids), replace=True)
        sampled = [record for group_id in sampled_ids for record in grouped[str(group_id)]]
        metrics = _numeric_metrics(metric_fn(sampled))
        for name in draws:
            if name in metrics:
                draws[name].append(metrics[name])
    alpha = (1.0 - confidence_level) / 2.0
    return {
        name: {
            "estimate": value,
            "bootstrap_mean": float(np.mean(draws[name])),
            "lower": float(np.quantile(draws[name], alpha)),
            "upper": float(np.quantile(draws[name], 1.0 - alpha)),
        }
        for name, value in point.items()
        if draws[name]
    }


def paired_group_bootstrap_difference(
    baseline: Iterable[UpdatePrediction],
    challenger: Iterable[UpdatePrediction],
    *,
    metric: str,
    iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 20260710,
    metric_fn: MetricFunction = evaluate_updates,
) -> dict[str, float]:
    """Estimate challenger-minus-baseline on identical AOI bootstrap draws."""

    baseline_values = list(baseline)
    challenger_values = list(challenger)
    baseline_ids = {record.sample_id for record in baseline_values}
    challenger_ids = {record.sample_id for record in challenger_values}
    if baseline_ids != challenger_ids:
        raise ValueError("paired comparison requires identical sample_id sets")
    baseline_groups: dict[str, list[UpdatePrediction]] = defaultdict(list)
    challenger_groups: dict[str, list[UpdatePrediction]] = defaultdict(list)
    for record in baseline_values:
        baseline_groups[record.aoi_id].append(record)
    for record in challenger_values:
        challenger_groups[record.aoi_id].append(record)
    group_ids = sorted(baseline_groups)
    if group_ids != sorted(challenger_groups):
        raise ValueError("paired comparison requires identical AOI sets")
    point_baseline = _numeric_metrics(metric_fn(baseline_values))
    point_challenger = _numeric_metrics(metric_fn(challenger_values))
    if metric not in point_baseline or metric not in point_challenger:
        raise ValueError(f"metric is not a scalar evaluation output: {metric}")

    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(iterations):
        sampled_ids = rng.choice(group_ids, size=len(group_ids), replace=True)
        baseline_sample = [
            record for group_id in sampled_ids for record in baseline_groups[str(group_id)]
        ]
        challenger_sample = [
            record for group_id in sampled_ids for record in challenger_groups[str(group_id)]
        ]
        baseline_metric = _numeric_metrics(metric_fn(baseline_sample))[metric]
        challenger_metric = _numeric_metrics(metric_fn(challenger_sample))[metric]
        differences.append(challenger_metric - baseline_metric)
    alpha = (1.0 - confidence_level) / 2.0
    differences_array = np.asarray(differences)
    probability_nonpositive = float(np.mean(differences_array <= 0.0))
    probability_nonnegative = float(np.mean(differences_array >= 0.0))
    return {
        "baseline": point_baseline[metric],
        "challenger": point_challenger[metric],
        "difference": point_challenger[metric] - point_baseline[metric],
        "lower": float(np.quantile(differences_array, alpha)),
        "upper": float(np.quantile(differences_array, 1.0 - alpha)),
        "p_two_sided": min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative)),
    }
