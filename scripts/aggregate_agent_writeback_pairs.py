#!/usr/bin/env python3
"""Aggregate executable writeback gains against seed-matched SFT outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.compare_agent_writebacks import (
    _grouped_delta_draws,
    _load,
    _metrics,
)


def parse_pair(value: str) -> tuple[int, Path, Path]:
    seed, separator, paths = value.partition("=")
    baseline, comma, candidate = paths.partition(",")
    if not separator or not comma or not seed or not baseline or not candidate:
        raise argparse.ArgumentTypeError(
            "pair must be SEED=/path/to/sft.jsonl,/path/to/candidate.jsonl"
        )
    return int(seed), Path(baseline), Path(candidate)


def aggregate(
    pairs: list[tuple[int, Path, Path]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    model_seeds = [model_seed for model_seed, _, _ in pairs]
    if len(model_seeds) < 2 or len(model_seeds) != len(set(model_seeds)):
        raise ValueError("at least two unique model-seed pairs are required")
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")

    rows = {
        model_seed: (_load(baseline), _load(candidate))
        for model_seed, baseline, candidate in pairs
    }
    canonical_protocol = None
    group_keys: dict[str, list[tuple[str, float]]] = defaultdict(list)
    per_seed_metrics = {}
    per_seed_delta = {}
    metric_names = None
    for model_seed, (baseline, candidate) in sorted(rows.items()):
        if baseline.keys() != candidate.keys():
            raise ValueError(f"writeback support mismatch for seed {model_seed}")
        protocol = {
            key: (
                str(row.get("aoi_id")),
                str(row.get("target")),
                float(row["budget"]),
            )
            for key, row in baseline.items()
        }
        for key in baseline:
            if protocol[key] != (
                str(candidate[key].get("aoi_id")),
                str(candidate[key].get("target")),
                float(candidate[key]["budget"]),
            ):
                raise ValueError(f"paired writeback metadata mismatch for seed {model_seed}")
            for row in (baseline[key], candidate[key]):
                if row.get("split") != "val" or row.get("test_assets_read") is not False:
                    raise ValueError("writeback aggregation is validation-only")
        if canonical_protocol is None:
            canonical_protocol = protocol
            for key, (aoi_id, _, _) in protocol.items():
                group_keys[aoi_id].append(key)
        elif protocol != canonical_protocol:
            raise ValueError(f"cross-seed writeback protocol mismatch for seed {model_seed}")
        baseline_metrics = _metrics(list(baseline.values()))
        candidate_metrics = _metrics(list(candidate.values()))
        if baseline_metrics.keys() != candidate_metrics.keys():
            raise ValueError(f"metric mismatch for seed {model_seed}")
        if metric_names is None:
            metric_names = list(baseline_metrics)
        elif list(baseline_metrics) != metric_names:
            raise ValueError(f"cross-seed metric mismatch for seed {model_seed}")
        per_seed_metrics[model_seed] = {
            "sft": baseline_metrics,
            "candidate": candidate_metrics,
        }
        per_seed_delta[model_seed] = {
            name: candidate_metrics[name] - baseline_metrics[name]
            for name in metric_names
        }

    if len(group_keys) < 2:
        raise ValueError("AOI bootstrap requires multiple groups")
    group_ids = sorted(group_keys)
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0, len(group_ids), size=(repetitions, len(group_ids))
    )
    sampled_counts = np.eye(len(group_ids), dtype=np.float64)[
        sampled_indices
    ].sum(axis=1)
    seed_draws = []
    for model_seed in sorted(rows):
        baseline, candidate = rows[model_seed]
        draws = _grouped_delta_draws(
            baseline,
            candidate,
            group_keys,
            group_ids,
            sampled_counts,
            metric_names,
        )
        seed_draws.append(np.column_stack([draws[name] for name in metric_names]))
    draw_matrix = np.stack(seed_draws, axis=0).mean(axis=0)
    observed = {
        name: statistics.fmean(
            per_seed_delta[model_seed][name] for model_seed in sorted(rows)
        )
        for name in metric_names
    }
    intervals = {
        name: {
            "observed_delta": observed[name],
            "ci95_low": float(np.quantile(draw_matrix[:, index], 0.025)),
            "ci95_high": float(np.quantile(draw_matrix[:, index], 0.975)),
        }
        for index, name in enumerate(metric_names)
    }
    seed_variation = {
        name: {
            "mean_delta": observed[name],
            "sample_std_delta": statistics.stdev(
                per_seed_delta[model_seed][name] for model_seed in sorted(rows)
            ),
        }
        for name in metric_names
    }
    return {
        "schema_version": "agent-writeback-seed-matched-aggregate-v1",
        "model_seeds": sorted(rows),
        "seed_count": len(rows),
        "record_count_per_seed": len(canonical_protocol),
        "aoi_count": len(group_keys),
        "seed_matched_references": True,
        "shared_aoi_resampling_across_model_seeds": True,
        "repetitions": repetitions,
        "bootstrap_seed": seed,
        "per_seed_metrics": per_seed_metrics,
        "per_seed_candidate_minus_sft": per_seed_delta,
        "seed_variation": seed_variation,
        "candidate_minus_seed_matched_sft": intervals,
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
