#!/usr/bin/env python3
"""Aggregate matched-cost SN7 post-tool causal runs across seeds and AOIs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

METRICS = (
    "terminal_correct",
    "false_edit",
    "missed_edit",
    "wrong_edit",
    "quality_gain",
    "quality_cost_utility",
    "tool_calls",
    "tool_belief_l1_delta",
)
VARIANTS = ("notool", "noadapter", "base", "updated", "tools")
COMPARISONS = (
    ("noadapter", "notool"),
    ("base", "noadapter"),
    ("updated", "noadapter"),
    ("updated", "base"),
    ("tools", "updated"),
    ("updated", "notool"),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty trace: {path}")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_rows(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    right_index = {row["sample_id"]: row for row in right}
    if len(right_index) != len(right):
        raise ValueError("duplicate sample IDs in reference trace")
    if {row["sample_id"] for row in left} != set(right_index):
        raise ValueError("paired traces do not contain identical samples")
    output = []
    for row in left:
        reference = right_index[row["sample_id"]]
        delta = {
            metric: float(row[metric]) - float(reference[metric])
            for metric in METRICS
        }
        output.append(
            {
                "sample_id": row["sample_id"],
                "aoi_id": row["aoi_id"],
                "action_flip": row["predicted_edit"] != reference["predicted_edit"],
                "tool_episode": int(row["tool_calls"]) > 0,
                "delta": delta,
            }
        )
    return output


def _hierarchical_bootstrap(
    paired_by_seed: dict[int, list[dict[str, Any]]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    seeds = np.asarray(sorted(paired_by_seed), dtype=np.int64)
    grouped: dict[int, dict[str, Any]] = {}
    for run_seed, rows in paired_by_seed.items():
        by_aoi: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_aoi[row["aoi_id"]].append(row)
        aoi_ids = sorted(by_aoi)
        grouped[run_seed] = {
            "counts": np.asarray(
                [len(by_aoi[aoi_id]) for aoi_id in aoi_ids], dtype=np.float64
            ),
            "sums": {
                metric: np.asarray(
                    [
                        sum(row["delta"][metric] for row in by_aoi[aoi_id])
                        for aoi_id in aoi_ids
                    ],
                    dtype=np.float64,
                )
                for metric in METRICS
            },
        }

    draws = {metric: np.empty(repetitions, dtype=np.float64) for metric in METRICS}
    for repetition in range(repetitions):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means = {metric: [] for metric in METRICS}
        for run_seed in sampled_seeds:
            clusters = grouped[int(run_seed)]
            counts = clusters["counts"]
            sampled_indices = rng.integers(0, len(counts), size=len(counts))
            denominator = float(counts[sampled_indices].sum())
            for metric in METRICS:
                seed_means[metric].append(
                    float(
                        clusters["sums"][metric][sampled_indices].sum()
                        / denominator
                    )
                )
        for metric in METRICS:
            draws[metric][repetition] = float(np.mean(seed_means[metric]))
    return {
        metric: {
            "mean_delta": float(
                np.mean(
                    [
                        row["delta"][metric]
                        for rows in paired_by_seed.values()
                        for row in rows
                    ]
                )
            ),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for metric, values in draws.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", default="20260730,20260731,20260801")
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--tag", default="wiring_v2")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 2 or args.bootstrap_repetitions <= 0:
        raise ValueError("at least two seeds and positive repetitions are required")

    traces: dict[int, dict[str, list[dict[str, Any]]]] = {}
    inputs = []
    per_seed = []
    for seed in seeds:
        traces[seed] = {}
        for variant in VARIANTS:
            root = (
                args.run_root
                / f"sn7_step0_causal_{variant}_n{args.limit}_seed{seed}_{args.tag}"
            )
            trace_path = root / "edit_utility.jsonl"
            summary_path = root / "summary.json"
            traces[seed][variant] = _read_jsonl(trace_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = summary["policies"]["edit_utility"]["metrics"]
            per_seed.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "terminal_accuracy": metrics["terminal_accuracy"],
                    "false_edit_rate": metrics["false_edit_rate"],
                    "missed_edit_rate": metrics["missed_edit_rate"],
                    "mean_quality_gain": metrics["mean_quality_gain"],
                    "mean_quality_cost_utility": metrics[
                        "mean_quality_cost_utility"
                    ],
                    "mean_tool_calls": metrics["mean_tool_calls"],
                    "mean_tool_belief_l1_delta": metrics[
                        "mean_tool_belief_l1_delta"
                    ],
                }
            )
            inputs.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "trace": str(trace_path.resolve()),
                    "trace_sha256": _sha256(trace_path),
                    "summary": str(summary_path.resolve()),
                    "summary_sha256": _sha256(summary_path),
                }
            )

    comparisons = {}
    for left, right in COMPARISONS:
        paired_by_seed = {
            seed: _paired_rows(traces[seed][left], traces[seed][right])
            for seed in seeds
        }
        comparisons[f"{left}_vs_{right}"] = {
            "per_seed": [
                {
                    "seed": seed,
                    "episodes": len(rows),
                    "tool_episodes": sum(row["tool_episode"] for row in rows),
                    "action_flips": sum(row["action_flip"] for row in rows),
                    "correct_delta_count": int(
                        sum(row["delta"]["terminal_correct"] for row in rows)
                    ),
                    "false_edit_delta_count": int(
                        sum(row["delta"]["false_edit"] for row in rows)
                    ),
                }
                for seed, rows in paired_by_seed.items()
            ],
            "hierarchical_seed_aoi_bootstrap": _hierarchical_bootstrap(
                paired_by_seed,
                repetitions=args.bootstrap_repetitions,
                seed=args.bootstrap_seed,
            ),
        }

    aggregate = {
        "schema_version": "sn7-post-tool-causal-three-seed-v1",
        "seeds": seeds,
        "variants": list(VARIANTS),
        "comparisons": comparisons,
        "per_seed": per_seed,
        "inputs": inputs,
        "protocol": {
            "matched_samples_costs_tools_and_budgets": True,
            "hierarchical_bootstrap_unit": "seed_then_aoi",
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.bootstrap_seed,
            "test_assets_read": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate["comparisons"], indent=2))


if __name__ == "__main__":
    main()
