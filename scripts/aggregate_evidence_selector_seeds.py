#!/usr/bin/env python3
"""Aggregate validation-only evidence-selector metrics across random seeds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", action="append", required=True, metavar="SEED=JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows_by_budget: dict[float, list[dict[str, object]]] = defaultdict(list)
    seeds = []
    for item in args.evaluation:
        seed, separator, raw_path = item.partition("=")
        if not separator:
            parser.error(f"invalid --evaluation value: {item}")
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        protocol = payload["protocol"]
        if protocol["split"] != "val" or protocol["test_evaluation"]:
            raise ValueError(f"seed {seed} is not a validation-only evaluation")
        learned = [row for row in payload["overall"] if row["method"] == "learned"]
        if not learned:
            raise ValueError(f"seed {seed} lacks learned-selector rows")
        seeds.append(seed)
        for row in learned:
            rows_by_budget[float(row["budget"])].append(row)

    metrics = (
        "mean_utility",
        "mean_oracle_utility",
        "mean_regret",
        "top1_accuracy",
        "stop_accuracy",
        "mean_cost",
    )
    budgets = []
    for budget, rows in sorted(rows_by_budget.items()):
        if len(rows) != len(seeds):
            raise ValueError(f"budget {budget:g} is missing one or more seeds")
        aggregate: dict[str, object] = {"budget": budget, "seed_count": len(rows)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            aggregate[f"{metric}_mean"] = float(np.mean(values))
            aggregate[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        budgets.append(aggregate)
    result = {
        "status": "validation_only",
        "seeds": seeds,
        "test_evaluation": None,
        "budgets": budgets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
