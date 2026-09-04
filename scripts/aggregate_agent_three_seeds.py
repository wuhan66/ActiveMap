#!/usr/bin/env python3
"""Hierarchical seed-and-task bootstrap for MUNO21 Agent rollouts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.compare_agent_rollouts import _load, _metrics


def aggregate(
    root: Path,
    *,
    seeds: list[int],
    candidate: str,
    baselines: list[str],
    bootstrap: int,
    rng_seed: int,
) -> dict[str, Any]:
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two unique model seeds are required")
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    pairing = json.loads((root / "seed_pairing.json").read_text(encoding="utf-8"))
    paired_agent_seeds = [int(row["agent_seed"]) for row in pairing["pairs"]]
    if paired_agent_seeds != seeds or pairing.get("test_assets_read") is not False:
        raise ValueError("seed pairing does not match requested train/validation seeds")

    loaded: dict[int, dict[str, dict[tuple[str, float], dict[str, Any]]]] = {}
    methods = [candidate, *baselines]
    reference_keys: set[tuple[str, float]] | None = None
    for seed in seeds:
        loaded[seed] = {
            method: _load(root / f"seed{seed}" / f"{method}.jsonl")
            for method in methods
        }
        keys = set(loaded[seed][candidate])
        if reference_keys is None:
            reference_keys = keys
        if keys != reference_keys:
            raise ValueError("model seeds do not share identical task-budget keys")
        for method in baselines:
            if set(loaded[seed][method]) != keys:
                raise ValueError(f"seed {seed} method {method} is not paired")
            for key in keys:
                if loaded[seed][method][key]["target"] != loaded[seed][candidate][key]["target"]:
                    raise ValueError(f"target mismatch for seed {seed}, key {key}")

    assert reference_keys is not None
    keys = sorted(reference_keys)
    task_keys: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for key in keys:
        task_keys[key[0]].append(key)
    task_ids = sorted(task_keys)
    rng = np.random.default_rng(rng_seed)
    comparisons: dict[str, Any] = {}

    for baseline in baselines:
        per_seed = []
        for seed in seeds:
            left = _metrics([loaded[seed][baseline][key] for key in keys])
            right = _metrics([loaded[seed][candidate][key] for key in keys])
            per_seed.append(
                {
                    "seed": seed,
                    "baseline": left,
                    "candidate": right,
                    "delta": {name: right[name] - left[name] for name in left},
                }
            )
        metric_names = list(per_seed[0]["delta"])
        draws = {name: [] for name in metric_names}
        for _ in range(bootstrap):
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            seed_deltas = {name: [] for name in metric_names}
            for sampled_seed in sampled_seeds:
                seed = int(sampled_seed)
                sampled_tasks = rng.choice(task_ids, size=len(task_ids), replace=True)
                sampled_keys = [key for task in sampled_tasks for key in task_keys[str(task)]]
                left = _metrics([loaded[seed][baseline][key] for key in sampled_keys])
                right = _metrics([loaded[seed][candidate][key] for key in sampled_keys])
                for name in metric_names:
                    seed_deltas[name].append(right[name] - left[name])
            for name in metric_names:
                draws[name].append(float(np.mean(seed_deltas[name])))
        comparisons[baseline] = {
            "per_seed": per_seed,
            "paired_delta": {
                name: {
                    "delta": float(np.mean([row["delta"][name] for row in per_seed])),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                }
                for name, values in draws.items()
            },
        }

    return {
        "schema_version": "activemap-agent-three-seed-bootstrap-v2",
        "protocol": {
            "split": "val",
            "test_assets_read": False,
            "model_seeds": seeds,
            "resampling": "model seeds and complete task trajectories",
            "bootstrap": bootstrap,
            "rng_seed": rng_seed,
            "primary_metric": (
                "episode_utility_v2_balanced_auc"
                if all(
                    "episode_utility_v2_balanced_auc" in row["candidate"]
                    for comparison in comparisons.values()
                    for row in comparison["per_seed"]
                )
                else "quality_cost_utility_auc"
            ),
        },
        "candidate": candidate,
        "task_count": len(task_ids),
        "sample_count_per_seed": len(keys),
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", default="20260821,20260822,20260823")
    parser.add_argument("--candidate", default="qwen3_4b_sft_tool_to_belief")
    parser.add_argument(
        "--baselines",
        default="edit_conditioned_selector,generic_selector,qwen3_4b_sft_tools_no_belief,forced_tools",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=20260824)
    args = parser.parse_args()
    result = aggregate(
        args.root,
        seeds=[int(value) for value in args.seeds.split(",")],
        candidate=args.candidate,
        baselines=args.baselines.split(","),
        bootstrap=args.bootstrap,
        rng_seed=args.rng_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
