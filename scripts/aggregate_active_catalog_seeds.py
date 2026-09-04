#!/usr/bin/env python3
"""Aggregate fixed-seed active-catalog evaluations with shared AOI bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from scripts.evaluate_active_catalog_selector import (
    _project_policy,
    active_catalog_metrics,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or {row.get("split") for row in rows} != {"val"}:
        raise ValueError(f"active-catalog seed records must be nonempty validation data: {path}")
    return rows


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def aggregate(
    paths: dict[int, Path], *, repetitions: int, bootstrap_seed: int
) -> dict[str, Any]:
    if len(paths) < 2 or repetitions <= 0:
        raise ValueError("at least two model seeds and positive repetitions are required")
    rows_by_seed = {seed: _load(path) for seed, path in sorted(paths.items())}
    reference_seed = next(iter(rows_by_seed))
    protocol_keys = (
        "target_selection",
        "target_evidence_id",
        "oracle_utility",
        "stop_utility",
        "budget",
        "gt_edit",
        "aoi_id",
        "source_episode",
    )
    reference = {
        str(row["example_id"]): tuple(row.get(key) for key in protocol_keys)
        for row in rows_by_seed[reference_seed]
    }
    if len(reference) != len(rows_by_seed[reference_seed]):
        raise ValueError("duplicate example IDs in reference seed")
    for seed, rows in rows_by_seed.items():
        protocol = {
            str(row["example_id"]): tuple(row.get(key) for key in protocol_keys)
            for row in rows
        }
        if protocol != reference or len(protocol) != len(rows):
            raise ValueError(f"seed {seed} does not share the frozen validation protocol")

    aoi_ids = sorted({str(row["aoi_id"]) for row in rows_by_seed[reference_seed]})
    grouped = {
        seed: {
            aoi: [row for row in rows if str(row["aoi_id"]) == aoi]
            for aoi in aoi_ids
        }
        for seed, rows in rows_by_seed.items()
    }
    per_seed = {seed: active_catalog_metrics(rows) for seed, rows in rows_by_seed.items()}
    metric_names = (
        "selection_macro_f1",
        "selection_macro_f1_delta_vs_always_stop",
        "realized_utility_mean",
        "exact_evidence_recall",
        "mean_regret",
        "false_call_rate",
        "harmful_call_rate_all_states",
        "mean_cost_per_state",
    )
    observed_mean = {
        key: statistics.fmean(metrics[key] for metrics in per_seed.values())
        for key in metric_names
    }
    baseline_names = ("always_stop", "cheapest", "clear_per_cost", "random")
    reference_rows = rows_by_seed[reference_seed]
    baseline_metrics = {
        name: active_catalog_metrics(_project_policy(reference_rows, name))
        for name in baseline_names
    }
    distributions = {key: [] for key in metric_names}
    delta_keys = ("selection_macro_f1", "realized_utility_mean", "mean_cost_per_state")
    baseline_deltas = {
        name: {key: [] for key in delta_keys} for name in baseline_names
    }
    rng = random.Random(bootstrap_seed)
    for _ in range(repetitions):
        sampled_aois = rng.choices(aoi_ids, k=len(aoi_ids))
        replicate_seed_metrics = {}
        for seed in rows_by_seed:
            sampled_rows = [
                row for aoi in sampled_aois for row in grouped[seed][aoi]
            ]
            replicate_seed_metrics[seed] = active_catalog_metrics(sampled_rows)
        replicate_mean = {
            key: statistics.fmean(metrics[key] for metrics in replicate_seed_metrics.values())
            for key in metric_names
        }
        for key in metric_names:
            distributions[key].append(replicate_mean[key])
        baseline_rows = [
            row
            for aoi in sampled_aois
            for row in grouped[reference_seed][aoi]
        ]
        for name in baseline_names:
            baseline = active_catalog_metrics(_project_policy(baseline_rows, name))
            for key in delta_keys:
                baseline_deltas[name][key].append(replicate_mean[key] - baseline[key])

    intervals = {
        key: {
            "observed_fixed_seed_mean": observed_mean[key],
            "ci95_low": _quantile(values, 0.025),
            "ci95_high": _quantile(values, 0.975),
        }
        for key, values in distributions.items()
    }
    for key in ("selection_macro_f1_delta_vs_always_stop", "realized_utility_mean"):
        intervals[key]["bootstrap_probability_gt_zero"] = sum(
            value > 0 for value in distributions[key]
        ) / repetitions
    delta_intervals = {
        name: {
            key: {
                "observed_delta": observed_mean[key] - baseline_metrics[name][key],
                "ci95_low": _quantile(values, 0.025),
                "ci95_high": _quantile(values, 0.975),
                "bootstrap_probability_gt_zero": sum(value > 0 for value in values)
                / repetitions,
            }
            for key, values in metrics.items()
        }
        for name, metrics in baseline_deltas.items()
    }
    return {
        "schema_version": "active-catalog-fixed-seed-aoi-bootstrap-v1",
        "model_seeds": sorted(paths),
        "seed_count": len(paths),
        "sample_count_per_seed": len(reference),
        "aoi_count": len(aoi_ids),
        "grouping_unit": "aoi_id",
        "shared_resample_indices_across_model_seeds": True,
        "repetitions": repetitions,
        "bootstrap_seed": bootstrap_seed,
        "per_seed_metrics": per_seed,
        "fixed_seed_mean_intervals": intervals,
        "baseline_metrics": baseline_metrics,
        "paired_baseline_delta_intervals": delta_intervals,
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
    parser.add_argument("--bootstrap-seed", type=int, default=20260717)
    args = parser.parse_args()
    paths = dict(_seed_path(value) for value in args.records)
    if len(paths) != len(args.records):
        raise ValueError("duplicate model seed")
    result = aggregate(paths, repetitions=args.repetitions, bootstrap_seed=args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
