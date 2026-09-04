#!/usr/bin/env python3
"""Aggregate true online persistence traces across updater seeds and AOIs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


BRANCHES = ("independent_reset", "carry_always_commit", "carry_safe_commit")
METRICS = (
    "final_raster_iou",
    "executed_raster_iou_gain",
    "false_edit",
    "missed_edit",
    "wrong_edit",
    "commit_accepted",
    "safe_commit_rejected",
    "recovered_from_prior_error",
    "input_prior_matches_canonical",
)


def parse_record(value: str) -> tuple[str, Path]:
    seed, separator, path = value.partition("=")
    if not separator or not seed:
        raise argparse.ArgumentTypeError("record must be SEED=/path/to/online_persistent_traces.jsonl")
    return seed, Path(path)


def load(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty online trace: {path}")
    result = {}
    for row in rows:
        if row.get("test_assets_read") is not False:
            raise ValueError("online aggregate is validation-only")
        key = (str(row["chain_id"]), int(row["step"]), str(row["branch"]))
        if key in result:
            raise ValueError(f"duplicate online trace identity in {path}: {key}")
        if key[2] not in BRANCHES:
            raise ValueError(f"unknown branch in {path}: {key[2]}")
        result[key] = row
    expected = {(key[0], key[1]) for key in result}
    for identity in expected:
        if {(identity[0], identity[1], branch) for branch in BRANCHES} - set(result):
            raise ValueError(f"incomplete branch trio in {path}: {identity}")
    return result


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    final_by_chain = {}
    for row in rows:
        final_by_chain[str(row["chain_id"])] = row
    final = list(final_by_chain.values())
    output = {
        "mean_step_raster_iou": float(np.mean([row["final_raster_iou"] for row in rows])),
        "mean_final_chain_raster_iou": float(np.mean([row["final_raster_iou"] for row in final])),
        "mean_step_raster_iou_gain": float(np.mean([row["executed_raster_iou_gain"] for row in rows])),
        "false_edit_rate": float(np.mean([row["false_edit"] for row in rows])),
        "missed_edit_rate": float(np.mean([row["missed_edit"] for row in rows])),
        "wrong_edit_rate": float(np.mean([row["wrong_edit"] for row in rows])),
        "commit_rate": float(np.mean([row["commit_accepted"] for row in rows])),
        "safe_rejection_rate": float(np.mean([row["safe_commit_rejected"] for row in rows])),
        "recovery_rate_after_prior_error": float(np.mean([row["recovered_from_prior_error"] for row in rows])),
        "noncanonical_online_input_rate": float(1.0 - np.mean([row["input_prior_matches_canonical"] for row in rows])),
    }
    return output


def aggregate(
    traces: dict[str, dict[tuple[str, int, str], dict[str, Any]]], *, repetitions: int, seed: int
) -> dict[str, Any]:
    seeds = sorted(traces)
    if len(seeds) < 2:
        raise ValueError("online persistence aggregation requires at least two updater seeds")
    support = None
    for trace in traces.values():
        keys = set(trace)
        if support is None:
            support = keys
        elif keys != support:
            raise ValueError("online traces do not share the same chain-step-branch support")
    assert support is not None
    by_seed_branch = {
        seed_label: {
            branch: [row for row in trace.values() if row["branch"] == branch]
            for branch in BRANCHES
        }
        for seed_label, trace in traces.items()
    }
    per_seed = {
        seed_label: {branch: _summarize(rows) for branch, rows in by_branch.items()}
        for seed_label, by_branch in by_seed_branch.items()
    }
    summary = {
        branch: {
            metric: {
                "mean": float(np.mean([per_seed[label][branch][metric] for label in seeds])),
                "seed_std": float(np.std([per_seed[label][branch][metric] for label in seeds], ddof=1)),
            }
            for metric in next(iter(per_seed.values()))[branch]
        }
        for branch in BRANCHES
    }

    comparisons = {
        "always_minus_reset": ("carry_always_commit", "independent_reset"),
        "safe_minus_always": ("carry_safe_commit", "carry_always_commit"),
        "safe_minus_reset": ("carry_safe_commit", "independent_reset"),
    }
    comparison_metrics = (
        "mean_step_raster_iou",
        "mean_final_chain_raster_iou",
        "false_edit_rate",
        "missed_edit_rate",
        "wrong_edit_rate",
        "commit_rate",
        "recovery_rate_after_prior_error",
    )
    observed = {
        name: {
            metric: float(np.mean([per_seed[label][left][metric] - per_seed[label][right][metric] for label in seeds]))
            for metric in comparison_metrics
        }
        for name, (left, right) in comparisons.items()
    }

    aois = sorted({str(row["aoi_id"]) for row in next(iter(traces.values())).values()})
    aoi_index = {aoi: index for index, aoi in enumerate(aois)}
    group_sums = np.zeros((len(seeds), len(comparisons), len(aois), len(comparison_metrics)))
    group_counts = np.zeros((len(seeds), len(comparisons), len(aois), len(comparison_metrics)))
    step_support = sorted({(chain_id, step) for chain_id, step, _ in support})
    for seed_index, seed_label in enumerate(seeds):
        trace = traces[seed_label]
        final_step_by_chain = {
            chain_id: max(step for candidate_chain, step in step_support if candidate_chain == chain_id)
            for chain_id, _ in step_support
        }
        for comparison_index, (_, (left, right)) in enumerate(comparisons.items()):
            for chain_id, step in step_support:
                left_row = trace[(chain_id, step, left)]
                right_row = trace[(chain_id, step, right)]
                index = aoi_index[str(left_row["aoi_id"])]
                values = {
                    "mean_step_raster_iou": left_row["final_raster_iou"] - right_row["final_raster_iou"],
                    "false_edit_rate": float(left_row["false_edit"]) - float(right_row["false_edit"]),
                    "missed_edit_rate": float(left_row["missed_edit"]) - float(right_row["missed_edit"]),
                    "wrong_edit_rate": float(left_row["wrong_edit"]) - float(right_row["wrong_edit"]),
                    "commit_rate": float(left_row["commit_accepted"]) - float(right_row["commit_accepted"]),
                    "recovery_rate_after_prior_error": float(left_row["recovered_from_prior_error"]) - float(right_row["recovered_from_prior_error"]),
                }
                for metric_index, metric in enumerate(comparison_metrics):
                    if metric == "mean_final_chain_raster_iou":
                        continue
                    group_sums[seed_index, comparison_index, index, metric_index] += values[metric]
                    group_counts[seed_index, comparison_index, index, metric_index] += 1.0
            for chain_id, final_step in final_step_by_chain.items():
                left_row = trace[(chain_id, final_step, left)]
                right_row = trace[(chain_id, final_step, right)]
                index = aoi_index[str(left_row["aoi_id"])]
                metric_index = comparison_metrics.index("mean_final_chain_raster_iou")
                group_sums[seed_index, comparison_index, index, metric_index] += (
                    left_row["final_raster_iou"] - right_row["final_raster_iou"]
                )
                group_counts[seed_index, comparison_index, index, metric_index] += 1.0
    rng = np.random.default_rng(seed)
    sampled_aois = rng.integers(0, len(aois), size=(repetitions, len(aois)))
    seed_draws = np.empty((len(seeds), repetitions, len(comparisons), len(comparison_metrics)))
    for seed_index in range(len(seeds)):
        for comparison_index in range(len(comparisons)):
            for metric_index in range(len(comparison_metrics)):
                counts = group_counts[seed_index, comparison_index, :, metric_index][sampled_aois].sum(axis=1)
                sums = group_sums[seed_index, comparison_index, :, metric_index][sampled_aois].sum(axis=1)
                if np.any(counts == 0):
                    raise ValueError("an AOI bootstrap draw has no paired support")
                seed_draws[seed_index, :, comparison_index, metric_index] = sums / counts
    sampled_seeds = rng.integers(0, len(seeds), size=(repetitions, len(seeds)))
    draws = np.moveaxis(seed_draws, 0, 1)[np.arange(repetitions)[:, None], sampled_seeds].mean(axis=1)
    for comparison_index, name in enumerate(comparisons):
        for metric_index, metric in enumerate(comparison_metrics):
            values = draws[:, comparison_index, metric_index]
            observed[name][metric] = {
                "observed_delta": observed[name][metric],
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
            }
    return {
        "schema_version": "activemap-online-persistent-maintenance-aggregate-v1",
        "split": "val",
        "test_assets_read": False,
        "seed_count": len(seeds),
        "aoi_count": len(aois),
        "chain_step_count": len(support) // len(BRANCHES),
        "per_seed": per_seed,
        "branches": summary,
        "paired_comparisons": observed,
        "bootstrap": {"unit": "updater seed then AOI", "repetitions": repetitions, "seed": seed},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--record", action="append", type=parse_record, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = dict(args.record)
    if len(records) != len(args.record):
        raise ValueError("duplicate updater seed")
    result = aggregate(
        {seed_label: load(path) for seed_label, path in records.items()},
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    result["sources"] = {seed_label: str(path.resolve()) for seed_label, path in records.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
