#!/usr/bin/env python3
"""Aggregate one recurrent policy across model seeds against a shared baseline."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.compare_active_catalog_closed_loop import (
    COMPARISON_METRICS,
    load_rows,
    metric_delta,
)
from scripts.evaluate_active_catalog_closed_loop import metrics


def parse_candidate(value: str) -> tuple[int, Path]:
    seed, separator, path = value.partition("=")
    if not separator or not seed or not path:
        raise argparse.ArgumentTypeError("candidate must be SEED=PATH")
    return int(seed), Path(path)


def aggregate(
    candidates: list[tuple[int, Path]],
    reference_path: Path,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    paths = dict(candidates)
    if len(paths) != len(candidates) or len(paths) < 2:
        raise ValueError("at least two unique model seeds are required")
    reference = load_rows(reference_path)
    rows = {
        model_seed: load_rows(path)
        for model_seed, path in sorted(paths.items())
    }
    protocol = {
        key: (str(row["aoi_id"]), str(row["target_edit"]))
        for key, row in reference.items()
    }
    for model_seed, values in rows.items():
        current = {
            key: (str(row["aoi_id"]), str(row["target_edit"]))
            for key, row in values.items()
        }
        if current != protocol:
            raise ValueError(f"protocol mismatch for seed {model_seed}")

    ordered_keys = sorted(reference)
    per_seed = {
        model_seed: metrics([values[key] for key in ordered_keys])
        for model_seed, values in rows.items()
    }
    reference_metrics = metrics([reference[key] for key in ordered_keys])
    observed = {
        name: statistics.fmean(
            metric_delta(
                [values[key] for key in ordered_keys],
                [reference[key] for key in ordered_keys],
            )[name]
            for values in rows.values()
        )
        for name in COMPARISON_METRICS
    }

    groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for key, row in reference.items():
        groups[str(row["aoi_id"])].append(key)
    if len(groups) < 2 or repetitions < 1:
        raise ValueError("AOI bootstrap requires multiple groups and draws")
    group_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in COMPARISON_METRICS}
    for _ in range(repetitions):
        selected = rng.choice(group_ids, size=len(group_ids), replace=True)
        keys = [key for group in selected for key in groups[str(group)]]
        seed_deltas = [
            metric_delta(
                [values[key] for key in keys],
                [reference[key] for key in keys],
            )
            for values in rows.values()
        ]
        for name in COMPARISON_METRICS:
            draws[name].append(
                statistics.fmean(delta[name] for delta in seed_deltas)
            )
    intervals = {
        name: {
            "observed_delta": observed[name],
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for name, values in draws.items()
    }
    return {
        "schema_version": "active-catalog-policy-seed-aggregate-v1",
        "model_seeds": sorted(rows),
        "seed_count": len(rows),
        "record_count_per_seed": len(reference),
        "aoi_count": len(groups),
        "shared_aoi_resampling_across_model_seeds": True,
        "repetitions": repetitions,
        "bootstrap_seed": seed,
        "reference": str(reference_path),
        "reference_metrics": reference_metrics,
        "per_seed_metrics": per_seed,
        "mean_candidate_metrics": {
            name: statistics.fmean(
                values[name] for values in per_seed.values()
            )
            for name in next(iter(per_seed.values()))
        },
        "candidate_minus_reference": intervals,
        "non_dominated_observed": (
            observed["mean_quality_cost_utility"] > 0.0
            and observed["terminal_accuracy"] >= 0.0
            and observed["false_edit_rate"] <= 0.0
        ),
        "strict_utility_gain_ci95": (
            intervals["mean_quality_cost_utility"]["ci95_low"] > 0.0
        ),
        "split": "val",
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = aggregate(
        args.candidate,
        args.reference,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
