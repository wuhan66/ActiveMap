#!/usr/bin/env python3
"""Aggregate aligned robustness slices across matched controller seeds."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.build_active_catalog_robustness_slices import (
    CORE_METRICS,
    _continuous_metadata,
    _episodes,
    _quantile_labels,
    _sha256,
    render_markdown,
)
from scripts.compare_active_catalog_closed_loop import (
    COMPARISON_FIELDS,
    COMPARISON_METRICS,
    load_rows,
    metric_delta,
)
from scripts.evaluate_active_catalog_closed_loop import metrics


def _pair(value: str) -> tuple[int, Path, Path]:
    raw_seed, separator, paths = value.partition("=")
    reference, comma, candidate = paths.partition(",")
    if not separator or not comma:
        raise argparse.ArgumentTypeError("pair must be SEED=REFERENCE,CANDIDATE")
    return int(raw_seed), Path(reference), Path(candidate)


def _shared_bootstrap(
    pairs: dict[int, dict[str, dict[tuple[str, float], dict[str, Any]]]],
    keys: list[tuple[str, float]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any] | None:
    groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
    first_seed = next(iter(pairs))
    for key in keys:
        groups[str(pairs[first_seed]["reference"][key]["aoi_id"])].append(key)
    if len(groups) < 2 or repetitions <= 0:
        return None
    group_ids = sorted(groups)
    rng = random.Random(seed)
    counts = np.zeros((repetitions, len(group_ids)), dtype=np.float64)
    group_index = {group: index for index, group in enumerate(group_ids)}
    for repetition in range(repetitions):
        for group in rng.choices(group_ids, k=len(group_ids)):
            counts[repetition, group_index[group]] += 1.0
    group_sizes = np.asarray([len(groups[group]) for group in group_ids], dtype=np.float64)
    sampled_rows = counts @ group_sizes
    seed_draws = []
    for pair in pairs.values():
        sums = np.zeros((len(group_ids), len(COMPARISON_METRICS)), dtype=np.float64)
        for group_idx, group in enumerate(group_ids):
            for key in groups[group]:
                for metric_idx, (_, field) in enumerate(COMPARISON_FIELDS):
                    sums[group_idx, metric_idx] += (
                        float(pair["candidate"][key][field])
                        - float(pair["reference"][key][field])
                    )
        seed_draws.append((counts @ sums) / sampled_rows[:, None])
    draws = np.mean(seed_draws, axis=0)
    return {
        "group_key": "aoi_id",
        "group_count": len(group_ids),
        "repetitions": repetitions,
        "shared_resampling_across_seeds": True,
        "intervals": {
            name: {
                "ci95_low": float(np.quantile(draws[:, index], 0.025)),
                "ci95_high": float(np.quantile(draws[:, index], 0.975)),
            }
            for index, name in enumerate(COMPARISON_METRICS)
            if name in CORE_METRICS
        },
    }


def aggregate(
    records: list[tuple[int, Path, Path]],
    episodes_path: Path,
    *,
    expected_records: int,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if len(records) < 2 or len({item[0] for item in records}) != len(records):
        raise ValueError("at least two unique matched seed pairs are required")
    pairs = {
        model_seed: {
            "reference": load_rows(reference),
            "candidate": load_rows(candidate),
        }
        for model_seed, reference, candidate in records
    }
    first_seed = next(iter(pairs))
    support = set(pairs[first_seed]["reference"])
    if len(support) != expected_records:
        raise ValueError(f"expected {expected_records} records")
    for model_seed, pair in pairs.items():
        if set(pair["reference"]) != support or set(pair["candidate"]) != support:
            raise ValueError(f"support mismatch for seed {model_seed}")
    episodes = _episodes(episodes_path)
    first_reference = pairs[first_seed]["reference"]
    continuous = {
        name: {} for name in (
            "object_area",
            "image_quality",
            "prior_confidence",
            "prior_uncertainty",
        )
    }
    categorical = {"edit_type": {}, "budget": {}, "aoi": {}}
    for key, row in first_reference.items():
        episode = episodes[str(row["source_episode"])]
        for name, value in _continuous_metadata(row, episode).items():
            continuous[name][key] = value
        categorical["edit_type"][key] = str(row["target_edit"])
        categorical["budget"][key] = f"{float(row['budget']):g}"
        categorical["aoi"][key] = str(row["aoi_id"])
    quantile_protocol = {}
    for name, values in continuous.items():
        categorical[name], quantile_protocol[name] = _quantile_labels(values)

    dimensions = {}
    for dimension, labels in categorical.items():
        slices = {}
        for value in sorted(set(labels.values())):
            keys = sorted(key for key in support if labels[key] == value)
            per_seed = {}
            for model_seed, pair in pairs.items():
                reference_rows = [pair["reference"][key] for key in keys]
                candidate_rows = [pair["candidate"][key] for key in keys]
                per_seed[str(model_seed)] = {
                    "reference": metrics(reference_rows),
                    "candidate": metrics(candidate_rows),
                    "candidate_minus_reference": metric_delta(
                        candidate_rows, reference_rows
                    ),
                }
            observed = {
                name: statistics.fmean(
                    row["candidate_minus_reference"][name]
                    for row in per_seed.values()
                )
                for name in CORE_METRICS
            }
            slices[value] = {
                "record_count_per_seed": len(keys),
                "aoi_count": len(
                    {str(first_reference[key]["aoi_id"]) for key in keys}
                ),
                "per_seed": per_seed,
                "candidate_minus_reference_observed_mean": observed,
                "candidate_minus_reference_seed_std": {
                    name: (
                        statistics.stdev(
                            row["candidate_minus_reference"][name]
                            for row in per_seed.values()
                        )
                        if len(per_seed) > 1
                        else 0.0
                    )
                    for name in CORE_METRICS
                },
                "candidate_minus_reference_aoi_bootstrap": _shared_bootstrap(
                    pairs, keys, repetitions=repetitions, seed=seed
                ),
            }
        dimensions[dimension] = slices
    return {
        "schema_version": "active-catalog-robustness-slices-matched-seeds-v1",
        "split": "val",
        "seed_count": len(pairs),
        "model_seeds": sorted(pairs),
        "record_count_per_seed": len(support),
        "reference": "matched_no_tool",
        "candidate": "matched_selective_recurrent",
        "dimensions": dimensions,
        "quantile_protocol": quantile_protocol,
        "bootstrap": {
            "unit": "aoi_id",
            "repetitions": repetitions,
            "seed": seed,
            "shared_resampling_across_seeds": True,
        },
        "sources": {
            "episodes": {
                "path": str(episodes_path.resolve()),
                "sha256": _sha256(episodes_path),
            },
            "pairs": {
                str(model_seed): {
                    "reference": str(reference.resolve()),
                    "reference_sha256": _sha256(reference),
                    "candidate": str(candidate.resolve()),
                    "candidate_sha256": _sha256(candidate),
                }
                for model_seed, reference, candidate in records
            },
        },
        "claim_boundary": (
            "observed validation strata; synthetic corruption and jitter are separate"
        ),
        "test_assets_read": False,
    }


def render(payload: dict[str, Any]) -> str:
    normalized = {
        **payload,
        "dimensions": {
            dimension: {
                label: {
                    "record_count": row["record_count_per_seed"],
                    "aoi_count": row["aoi_count"],
                    "candidate_minus_reference_observed": (
                        row["candidate_minus_reference_observed_mean"]
                    ),
                }
                for label, row in slices.items()
            }
            for dimension, slices in payload["dimensions"].items()
        },
    }
    return render_markdown(normalized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pair", action="append", type=_pair, required=True)
    parser.add_argument("--expected-records", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    payload = aggregate(
        args.pair,
        args.episodes,
        expected_records=args.expected_records,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "robustness_slices.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "robustness_slices.md").write_text(
        render(payload), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
