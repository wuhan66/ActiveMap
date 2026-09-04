#!/usr/bin/env python3
"""Aggregate recurrent policy gains against seed-matched SFT references."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.compare_active_catalog_closed_loop import (
    COMPARISON_FIELDS,
    COMPARISON_METRICS,
    load_rows,
    metric_delta,
)
from scripts.evaluate_active_catalog_closed_loop import metrics


def parse_pair(value: str) -> tuple[int, Path, Path]:
    seed, separator, paths = value.partition("=")
    candidate, comma, reference = paths.partition(",")
    if not separator or not comma or not seed or not candidate or not reference:
        raise argparse.ArgumentTypeError(
            "pair must be SEED=/path/to/candidate.jsonl,/path/to/reference.jsonl"
        )
    return int(seed), Path(candidate), Path(reference)


def aggregate(
    pairs: list[tuple[int, Path, Path]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    seeds = [model_seed for model_seed, _, _ in pairs]
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("at least two unique model-seed pairs are required")
    if repetitions < 1:
        raise ValueError("at least one bootstrap draw is required")

    rows = {
        model_seed: (load_rows(candidate), load_rows(reference))
        for model_seed, candidate, reference in pairs
    }
    canonical_protocol = None
    groups: dict[str, list[Any]] = defaultdict(list)
    for model_seed, (candidate, reference) in sorted(rows.items()):
        if set(candidate) != set(reference):
            raise ValueError(f"candidate/reference support mismatch for seed {model_seed}")
        protocol = {
            key: (
                str(row["aoi_id"]),
                str(row["target_edit"]),
                float(row["budget"]),
            )
            for key, row in reference.items()
        }
        if canonical_protocol is None:
            canonical_protocol = protocol
            for key, (aoi_id, _, _) in protocol.items():
                groups[aoi_id].append(key)
        elif protocol != canonical_protocol:
            raise ValueError(f"cross-seed protocol mismatch for seed {model_seed}")
    if len(groups) < 2:
        raise ValueError("AOI bootstrap requires multiple groups")

    ordered_keys = sorted(canonical_protocol)
    per_seed = {}
    observed_by_seed = {}
    for model_seed, (candidate, reference) in sorted(rows.items()):
        candidate_values = [candidate[key] for key in ordered_keys]
        reference_values = [reference[key] for key in ordered_keys]
        per_seed[model_seed] = {
            "candidate": metrics(candidate_values),
            "reference": metrics(reference_values),
        }
        observed_by_seed[model_seed] = metric_delta(
            candidate_values, reference_values
        )

    group_ids = sorted(groups)
    group_counts = np.asarray(
        [len(groups[group_id]) for group_id in group_ids], dtype=np.float64
    )
    group_delta_sums = np.zeros(
        (len(rows), len(group_ids), len(COMPARISON_METRICS)), dtype=np.float64
    )
    for seed_index, model_seed in enumerate(sorted(rows)):
        candidate, reference = rows[model_seed]
        for group_index, group_id in enumerate(group_ids):
            for key in groups[group_id]:
                for metric_index, (_, field) in enumerate(COMPARISON_FIELDS):
                    group_delta_sums[seed_index, group_index, metric_index] += (
                        float(candidate[key][field]) - float(reference[key][field])
                    )

    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0, len(group_ids), size=(repetitions, len(group_ids))
    )
    sampled_group_counts = np.zeros(
        (repetitions, len(group_ids)), dtype=np.float64
    )
    for draw_index, indices in enumerate(sampled_indices):
        sampled_group_counts[draw_index] = np.bincount(
            indices, minlength=len(group_ids)
        )
    sampled_rows = sampled_group_counts @ group_counts
    seed_draws = np.stack(
        [
            (sampled_group_counts @ group_delta_sums[seed_index])
            / sampled_rows[:, None]
            for seed_index in range(len(rows))
        ],
        axis=0,
    )
    draw_matrix = seed_draws.mean(axis=0)
    observed = {
        name: statistics.fmean(
            observed_by_seed[model_seed][name] for model_seed in sorted(rows)
        )
        for name in COMPARISON_METRICS
    }
    seed_variation = {
        name: {
            "mean_delta": observed[name],
            "sample_std_delta": (
                statistics.stdev(
                    observed_by_seed[model_seed][name]
                    for model_seed in sorted(rows)
                )
                if len(rows) > 1
                else 0.0
            ),
        }
        for name in COMPARISON_METRICS
    }
    intervals = {
        name: {
            "observed_delta": observed[name],
            "ci95_low": float(np.quantile(draw_matrix[:, index], 0.025)),
            "ci95_high": float(np.quantile(draw_matrix[:, index], 0.975)),
        }
        for index, name in enumerate(COMPARISON_METRICS)
    }
    return {
        "schema_version": "active-catalog-paired-policy-seed-aggregate-v1",
        "model_seeds": sorted(rows),
        "seed_count": len(rows),
        "record_count_per_seed": len(ordered_keys),
        "aoi_count": len(groups),
        "shared_aoi_resampling_across_model_seeds": True,
        "seed_matched_references": True,
        "repetitions": repetitions,
        "bootstrap_seed": seed,
        "per_seed_metrics": per_seed,
        "per_seed_candidate_minus_sft": observed_by_seed,
        "seed_variation": seed_variation,
        "candidate_minus_seed_matched_sft": intervals,
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
    parser.add_argument("--pair", action="append", type=parse_pair, required=True)
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = aggregate(args.pair, repetitions=args.repetitions, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
