#!/usr/bin/env python3
"""Aggregate paired executable writebacks across seeds with shared AOI bootstrap."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.compare_agent_writebacks import _grouped_delta_draws, _load, _metrics


def _seed_path(value: str) -> tuple[int, Path]:
    raw_seed, separator, raw_path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("input must be SEED=PATH")
    return int(raw_seed), Path(raw_path)


def aggregate(
    baselines: dict[int, Path],
    candidates: dict[int, Path],
    *,
    repetitions: int,
    seed: int,
    split: str = "val",
    frozen_test: bool = False,
) -> dict[str, Any]:
    test_assets_read = split == "test"
    if test_assets_read:
        if not frozen_test:
            raise PermissionError("test writeback aggregation requires --frozen-test")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    elif frozen_test:
        raise ValueError("--frozen-test is valid only for the test split")
    if baselines.keys() != candidates.keys() or len(baselines) < 2:
        raise ValueError("matching baseline/candidate inputs for at least two seeds are required")
    rows = {
        model_seed: {
            "baseline": _load(baselines[model_seed]),
            "candidate": _load(candidates[model_seed]),
        }
        for model_seed in sorted(baselines)
    }
    for model_seed, pair in rows.items():
        for label, values in pair.items():
            if any(
                row.get("split", "val") != split
                or row.get("test_assets_read", False) is not test_assets_read
                for row in values.values()
            ):
                raise ValueError(f"writeback split mismatch for {model_seed}/{label}")
    reference_seed = next(iter(rows))
    reference = rows[reference_seed]["candidate"]
    protocol = {
        key: (str(row.get("aoi_id", key[0])), str(row["target"]))
        for key, row in reference.items()
    }
    for model_seed, pair in rows.items():
        for label, values in pair.items():
            current = {
                key: (str(row.get("aoi_id", key[0])), str(row["target"]))
                for key, row in values.items()
            }
            if current != protocol:
                raise ValueError(f"writeback protocol mismatch for {model_seed}/{label}")
    groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for key, row in reference.items():
        groups[str(row.get("aoi_id", key[0]))].append(key)
    if len(groups) < 2 or repetitions < 1:
        raise ValueError("multiple AOIs and positive repetitions are required")
    ordered_keys = sorted(reference)
    group_ids = sorted(groups)

    per_seed = {
        model_seed: {
            label: _metrics([values[key] for key in ordered_keys])
            for label, values in pair.items()
        }
        for model_seed, pair in rows.items()
    }
    names = per_seed[reference_seed]["candidate"].keys()
    baseline_mean = {
        name: statistics.fmean(per_seed[s]["baseline"][name] for s in rows) for name in names
    }
    candidate_mean = {
        name: statistics.fmean(per_seed[s]["candidate"][name] for s in rows) for name in names
    }
    observed = {name: candidate_mean[name] - baseline_mean[name] for name in names}
    rng = random.Random(seed)
    group_index = {group_id: index for index, group_id in enumerate(group_ids)}
    sampled_counts = np.zeros((repetitions, len(group_ids)), dtype=np.float64)
    for repetition in range(repetitions):
        for group_id in rng.choices(group_ids, k=len(group_ids)):
            sampled_counts[repetition, group_index[group_id]] += 1.0
    seed_draws = [
        _grouped_delta_draws(
            rows[s]["baseline"],
            rows[s]["candidate"],
            groups,
            group_ids,
            sampled_counts,
            list(names),
        )
        for s in rows
    ]
    draws = {
        name: np.mean([seed_result[name] for seed_result in seed_draws], axis=0)
        for name in names
    }
    return {
        "schema_version": "active-catalog-fixed-seed-writeback-bootstrap-v1",
        "model_seeds": sorted(rows),
        "seed_count": len(rows),
        "group_key": "aoi_id",
        "group_count": len(groups),
        "shared_aoi_resampling_across_seeds": True,
        "baseline": baseline_mean,
        "candidate": candidate_mean,
        "per_seed": per_seed,
        "paired_delta": {
            name: {
                "delta": observed[name],
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
            }
            for name, values in draws.items()
        },
        "split": split,
        "test_assets_read": test_assets_read,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", action="append", type=_seed_path, required=True)
    parser.add_argument("--candidate", action="append", type=_seed_path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--frozen-test", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = aggregate(
        dict(args.baseline),
        dict(args.candidate),
        repetitions=args.repetitions,
        seed=args.seed,
        split=args.split,
        frozen_test=args.frozen_test,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
