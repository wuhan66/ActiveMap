#!/usr/bin/env python3
"""Aggregate recurrent tool branches across model seeds with shared AOI bootstrap."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.compare_active_catalog_closed_loop import COMPARISON_METRICS, load_rows, metric_delta
from scripts.evaluate_active_catalog_closed_loop import metrics


def _record(value: str) -> tuple[int, str, Path]:
    head, separator, raw_path = value.partition("=")
    seed, colon, label = head.partition(":")
    if not separator or not colon or not seed or not label or not raw_path:
        raise argparse.ArgumentTypeError("record must be SEED:LABEL=PATH")
    return int(seed), label, Path(raw_path)


def aggregate(
    records: list[tuple[int, str, Path]],
    *,
    repetitions: int,
    seed: int,
    split: str = "val",
    frozen_test: bool = False,
) -> dict[str, Any]:
    test_assets_read = split == "test"
    required_labels = {"selective", "forced", "no_tool"}
    optional_labels = {"selective_frozen_prior"}
    paths: dict[int, dict[str, Path]] = defaultdict(dict)
    for model_seed, label, path in records:
        if label in paths[model_seed]:
            raise ValueError(f"duplicate seed/label: {model_seed}/{label}")
        paths[model_seed][label] = path
    if len(paths) < 2:
        raise ValueError("at least two model seeds are required")
    label_sets = {frozenset(branches) for branches in paths.values()}
    if len(label_sets) != 1:
        raise ValueError("every seed must contain the same tool branches")
    labels = set(next(iter(label_sets)))
    if not required_labels.issubset(labels) or not labels.issubset(
        required_labels | optional_labels
    ):
        raise ValueError(
            "branches must contain selective, forced, no_tool, and optionally "
            "selective_frozen_prior"
        )
    rows = {
        model_seed: {
            label: load_rows(path, split=split, frozen_test=frozen_test)
            for label, path in branches.items()
        }
        for model_seed, branches in sorted(paths.items())
    }
    reference_seed = next(iter(rows))
    reference = rows[reference_seed]["selective"]
    protocol = {
        key: (str(row["aoi_id"]), str(row["target_edit"])) for key, row in reference.items()
    }
    for model_seed, branches in rows.items():
        for label, values in branches.items():
            current = {
                key: (str(row["aoi_id"]), str(row["target_edit"])) for key, row in values.items()
            }
            if current != protocol:
                raise ValueError(f"protocol mismatch for {model_seed}/{label}")

    groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for key, row in reference.items():
        groups[str(row["aoi_id"])].append(key)
    if len(groups) < 2 or repetitions < 1:
        raise ValueError("multiple AOIs and positive repetitions are required")
    group_ids = sorted(groups)
    ordered_keys = sorted(reference)

    per_seed_metrics = {
        model_seed: {
            label: metrics([values[key] for key in ordered_keys])
            for label, values in branches.items()
        }
        for model_seed, branches in rows.items()
    }
    metric_names = per_seed_metrics[reference_seed]["selective"].keys()
    mean_metrics = {
        label: {
            name: statistics.fmean(per_seed_metrics[s][label][name] for s in rows)
            for name in metric_names
        }
        for label in labels
    }
    reference_labels = sorted(labels - {"selective"})
    samples = {
        reference_label: {name: [] for name in COMPARISON_METRICS}
        for reference_label in reference_labels
    }
    observed = {
        reference_label: {
            name: statistics.fmean(
                metric_delta(
                    [rows[s]["selective"][key] for key in ordered_keys],
                    [rows[s][reference_label][key] for key in ordered_keys],
                )[name]
                for s in rows
            )
            for name in COMPARISON_METRICS
        }
        for reference_label in samples
    }
    rng = random.Random(seed)
    for _ in range(repetitions):
        selected = rng.choices(group_ids, k=len(group_ids))
        keys = [key for group in selected for key in groups[group]]
        for reference_label in samples:
            seed_deltas = [
                metric_delta(
                    [rows[s]["selective"][key] for key in keys],
                    [rows[s][reference_label][key] for key in keys],
                )
                for s in rows
            ]
            for name in COMPARISON_METRICS:
                samples[reference_label][name].append(
                    statistics.fmean(value[name] for value in seed_deltas)
                )
    comparisons = {}
    for reference_label, values in samples.items():
        intervals = {
            name: {
                "observed_delta": observed[reference_label][name],
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
            }
            for name, draws in values.items()
        }
        comparisons[f"selective_minus_{reference_label}"] = {
            "group_key": "aoi_id",
            "group_count": len(groups),
            "repetitions": repetitions,
            "intervals": intervals,
        }
    return {
        "schema_version": "active-catalog-closed-loop-paired-comparison-v1",
        "comparison_orientation": {
            "candidate": "selective",
            "references": reference_labels,
        },
        "model_seeds": sorted(rows),
        "seed_count": len(rows),
        "record_count_per_seed": len(reference),
        "shared_aoi_resampling_across_seeds": True,
        "metrics": mean_metrics,
        "per_seed_metrics": per_seed_metrics,
        "paired_aoi_comparisons": comparisons,
        "split": split,
        "test_assets_read": test_assets_read,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--records", action="append", type=_record, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--frozen-test", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = aggregate(
        args.records,
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
