#!/usr/bin/env python3
"""Aggregate generated-OOF terminal critics with seed/task bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = {
    "terminal_accuracy": "terminal_correct",
    "false_edit_rate": "false_edit",
    "missed_edit_rate": "missed_edit",
    "balanced_utility": "episode_utility_v2_proxy_balanced",
    "safety_utility": "episode_utility_v2_proxy_safety",
}


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()

    runs = sorted(args.run_root.glob("critic*_agent*"))
    if len(runs) != 4:
        raise ValueError(f"expected four critic runs, found {len(runs)}")
    paired = []
    per_seed = []
    for run in runs:
        agent_seed = run.name.split("_agent", 1)[1]
        hybrid = load(run / "hybrid.jsonl")
        baseline = load(
            args.baseline_root
            / f"seed{agent_seed}"
            / "edit_conditioned_proactive_tools.jsonl"
        )
        baseline_by_id = {str(row["sample_id"]): row for row in baseline}
        if {str(row["sample_id"]) for row in hybrid} != baseline_by_id.keys():
            raise ValueError(f"sample mismatch for {run.name}")
        rows = []
        for candidate in hybrid:
            base = baseline_by_id[str(candidate["sample_id"])]
            rows.append(
                {
                    "task_id": str(candidate["task_id"]),
                    **{
                        metric: float(candidate[key]) - float(base[key])
                        for metric, key in METRICS.items()
                    },
                }
            )
        paired.append(rows)
        per_seed.append(
            {
                "run": run.name,
                **{
                    metric: float(np.mean([row[metric] for row in rows]))
                    for metric in METRICS
                },
            }
        )

    point = {
        metric: float(np.mean([row[metric] for rows in paired for row in rows]))
        for metric in METRICS
    }
    rng = np.random.default_rng(args.seed)
    draws = {metric: [] for metric in METRICS}
    for _ in range(args.bootstrap):
        seed_indices = rng.integers(0, len(paired), size=len(paired))
        sampled = []
        for seed_index in seed_indices:
            rows = paired[int(seed_index)]
            tasks = sorted({row["task_id"] for row in rows})
            selected_tasks = rng.choice(tasks, size=len(tasks), replace=True)
            by_task = {
                task: [row for row in rows if row["task_id"] == task]
                for task in tasks
            }
            for task in selected_tasks:
                sampled.extend(by_task[str(task)])
        for metric in METRICS:
            draws[metric].append(float(np.mean([row[metric] for row in sampled])))
    intervals = {
        metric: {
            "delta": point[metric],
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for metric, values in draws.items()
    }
    promotion = {
        "balanced_utility_positive": intervals["balanced_utility"]["ci95_low"] > 0,
        "safety_not_worse": intervals["safety_utility"]["ci95_low"] >= 0,
        "false_edit_not_worse": intervals["false_edit_rate"]["ci95_high"] <= 0,
        "accuracy_not_worse": intervals["terminal_accuracy"]["ci95_low"] >= 0,
    }
    promotion["passed"] = all(promotion.values())
    result = {
        "schema_version": "muno21-generated-oof-terminal-critic-aggregate-v1",
        "protocol": {
            "critic_seeds": len(runs),
            "bootstrap": args.bootstrap,
            "resampling": "critic-seed then opaque task_id",
            "fit_split": "train genuine generated OOF",
            "evaluation_split": "validation frozen",
            "test_assets_read": False,
        },
        "per_seed_delta": per_seed,
        "paired_delta": intervals,
        "promotion_gate": promotion,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
