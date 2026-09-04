#!/usr/bin/env python3
"""Aggregate selector seed evaluations with paired AOI bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.evaluate_selector_checkpoint import aggregate_rows
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from evaluate_selector_checkpoint import aggregate_rows


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def aggregate_seeds(
    seed_rows: list[list[dict[str, Any]]], *, draws: int, seed: int
) -> dict[str, Any]:
    if len(seed_rows) < 2:
        raise ValueError("multi-seed aggregation requires at least two seeds")
    reference_ids = [str(row["sample_id"]) for row in seed_rows[0]]
    if any([str(row["sample_id"]) for row in rows] != reference_ids for rows in seed_rows[1:]):
        raise ValueError("seed evaluations must contain identical ordered samples")
    per_seed = [aggregate_rows(rows) for rows in seed_rows]
    metric_names = [name for name in per_seed[0] if name != "sample_count"]
    mean = {
        name: float(np.mean([metrics[name] for metrics in per_seed]))
        for name in metric_names
    }
    sample_sd = {
        name: float(np.std([metrics[name] for metrics in per_seed], ddof=1))
        for name in metric_names
    }
    grouped = []
    for rows in seed_rows:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row["aoi_id"]), []).append(row)
        grouped.append(groups)
    aoi_names = sorted(grouped[0])
    if any(sorted(groups) != aoi_names for groups in grouped[1:]):
        raise ValueError("seed evaluations must contain identical AOI support")
    rng = np.random.default_rng(seed)
    bootstrap: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(draws):
        sampled = rng.choice(aoi_names, size=len(aoi_names), replace=True)
        draw_metrics = []
        for groups in grouped:
            rows = [row for name in sampled for row in groups[str(name)]]
            draw_metrics.append(aggregate_rows(rows))
        for name in metric_names:
            bootstrap[name].append(
                float(np.mean([metrics[name] for metrics in draw_metrics]))
            )
    ci = {
        name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for name, values in bootstrap.items()
    }
    return {
        "seed_count": len(seed_rows),
        "sample_count_per_seed": len(seed_rows[0]),
        "aoi_count": len(aoi_names),
        "mean": mean,
        "sample_sd": sample_sd,
        "paired_aoi_bootstrap_95_ci": ci,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("per_sample", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260721)
    args = parser.parse_args()
    summary = {
        "schema_version": "selector-three-seed-evaluation-v1",
        "inputs": [str(path.resolve()) for path in args.per_sample],
        "test_assets_read": False,
        "bootstrap_draws": args.bootstrap_draws,
        **aggregate_seeds(
            [load_rows(path) for path in args.per_sample],
            draws=args.bootstrap_draws,
            seed=args.bootstrap_seed,
        ),
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
