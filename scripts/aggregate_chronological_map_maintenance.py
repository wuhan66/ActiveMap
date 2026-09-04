#!/usr/bin/env python3
"""Aggregate chronological maintenance traces across independently trained seeds."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _parse(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("record must be seed=trace.jsonl")
    seed, path = value.split("=", 1)
    return seed, Path(path)


def _load(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty chronological trace: {path}")
    keys = [(row["task_id"], row["budget"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate chronological identities: {path}")
    if any(row.get("test_assets_read") for row in rows):
        raise ValueError("validation aggregate cannot include test-access rows")
    return rows


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "independent_iou": float(np.mean([row["independent_iou"] for row in rows])),
        "carry_iou": float(np.mean([row["carry_iou"] for row in rows])),
        "risk_gated_iou": float(np.mean([row["risk_gated_iou"] for row in rows])),
        "carry_minus_independent": float(
            np.mean([row["carry_iou"] - row["independent_iou"] for row in rows])
        ),
        "risk_gated_minus_carry": float(
            np.mean([row["risk_gated_iou"] - row["carry_iou"] for row in rows])
        ),
        "risk_gated_minus_independent": float(
            np.mean(
                [row["risk_gated_iou"] - row["independent_iou"] for row in rows]
            )
        ),
        "risk_gated_intervention_rate": float(
            np.mean([row["risk_gate_intervened"] for row in rows])
        ),
        "requested_intervention_rate": float(
            np.mean([row["requested_intervention"] for row in rows])
        ),
    }


def aggregate(
    records: dict[str, list[dict[str, Any]]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    identities = {
        label: {(row["task_id"], float(row["budget"])) for row in rows}
        for label, rows in records.items()
    }
    reference = next(iter(identities.values()))
    if any(value != reference for value in identities.values()):
        raise ValueError("three-seed chronological support is not identical")
    per_seed = {label: _metrics(rows) for label, rows in records.items()}
    metric_names = tuple(next(iter(per_seed.values())))
    aggregate_metrics = {
        name: {
            "mean": float(np.mean([values[name] for values in per_seed.values()])),
            "seed_std": float(
                np.std([values[name] for values in per_seed.values()], ddof=1)
                if len(per_seed) > 1
                else 0.0
            ),
        }
        for name in metric_names
    }
    rng = random.Random(seed)
    seed_labels = sorted(records)
    bootstrap = {name: [] for name in metric_names}
    grouped = {}
    for label, rows in records.items():
        by_aoi: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_aoi[str(row["aoi_id"])].append(row)
        grouped[label] = by_aoi
    for _ in range(repetitions):
        sampled_seed_metrics = []
        for label in rng.choices(seed_labels, k=len(seed_labels)):
            by_aoi = grouped[label]
            aoi_ids = sorted(by_aoi)
            selected = [
                row
                for aoi in rng.choices(aoi_ids, k=len(aoi_ids))
                for row in by_aoi[aoi]
            ]
            sampled_seed_metrics.append(_metrics(selected))
        for name in metric_names:
            bootstrap[name].append(
                float(np.mean([values[name] for values in sampled_seed_metrics]))
            )
    for name, values in bootstrap.items():
        aggregate_metrics[name].update(
            {
                "hierarchical_ci95_low": float(np.quantile(values, 0.025)),
                "hierarchical_ci95_high": float(np.quantile(values, 0.975)),
            }
        )
    return {
        "schema_version": "activemap-chronological-maintenance-three-seed-v1",
        "seed_count": len(records),
        "transition_count_per_seed": len(next(iter(records.values()))),
        "test_assets_read": False,
        "per_seed": per_seed,
        "aggregate": aggregate_metrics,
        "bootstrap": {
            "unit": "seed then AOI",
            "repetitions": repetitions,
            "seed": seed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--record", action="append", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    parsed = dict(_parse(value) for value in args.record)
    if len(parsed) != len(args.record):
        raise ValueError("duplicate seed label")
    result = aggregate(
        {label: _load(path) for label, path in parsed.items()},
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    result["sources"] = {label: str(path.resolve()) for label, path in parsed.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
