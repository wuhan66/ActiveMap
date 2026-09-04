#!/usr/bin/env python3
"""Aggregate deployable pre-call semantic-tool selectors across model seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

VARIANTS = (
    "no_tool",
    "forced_tool",
    "selective_tool",
    "oracle_tool",
    "cross_task_mismatch",
)
METRICS = (
    "accuracy",
    "macro_f1",
    "false_edit_rate",
    "missed_edit_rate",
    "tool_call_rate",
    "mean_terminal_reward",
    "mean_tool_cost",
    "mean_utility",
)
FROZEN_KEYS = (
    "schema_version",
    "selection_protocol",
    "selector_feature_protocol",
    "tool_cost",
    "train_examples",
    "train_tasks",
    "validation_examples",
    "validation_tasks",
)


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(paths: list[Path], expected_seeds: tuple[int, ...]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two seed results are required")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(row.get("test_assets_read") is not False for row in payloads):
        raise ValueError("all seed results must keep test assets frozen")
    seeds = [row.get("model_training_seed") for row in payloads]
    if any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("each result must record model_training_seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("model-training seeds must be unique")
    if tuple(sorted(seeds)) != tuple(sorted(expected_seeds)):
        raise ValueError(f"seed mismatch: found={sorted(seeds)} expected={sorted(expected_seeds)}")

    protocol = {key: payloads[0].get(key) for key in FROZEN_KEYS}
    for row in payloads[1:]:
        if {key: row.get(key) for key in FROZEN_KEYS} != protocol:
            raise ValueError("seed results do not share a frozen protocol")

    per_seed = []
    for path, row in sorted(
        zip(paths, payloads, strict=True), key=lambda item: item[1]["model_training_seed"]
    ):
        validation = row.get("validation", {})
        for variant in VARIANTS:
            if variant not in validation:
                raise ValueError(f"missing validation variant {variant}: {path}")
            if any(metric not in validation[variant] for metric in METRICS):
                raise ValueError(f"incomplete validation metrics for {variant}: {path}")
        per_seed.append(
            {
                "model_training_seed": int(row["model_training_seed"]),
                "selector_seed": int(row["seed"]),
                "promotion_passed": bool(row.get("promotion_gate", {}).get("passed")),
                "train_beneficial_tool_rate": float(row["train_beneficial_tool_rate"]),
                "selected_C": row["selected_gate"].get("C"),
                "selected_threshold": float(row["selected_gate"]["threshold"]),
                "validation": {
                    variant: {metric: float(validation[variant][metric]) for metric in METRICS}
                    for variant in VARIANTS
                },
                "source": str(path.resolve()),
                "sha256": _sha256(path),
            }
        )

    aggregate_metrics = {
        variant: {
            metric: _mean_std([row["validation"][variant][metric] for row in per_seed])
            for metric in METRICS
        }
        for variant in VARIANTS
    }
    promotion_count = sum(row["promotion_passed"] for row in per_seed)
    oracle_gaps = {
        "macro_f1": _mean_std(
            [
                row["validation"]["oracle_tool"]["macro_f1"]
                - row["validation"]["selective_tool"]["macro_f1"]
                for row in per_seed
            ]
        ),
        "mean_utility": _mean_std(
            [
                row["validation"]["oracle_tool"]["mean_utility"]
                - row["validation"]["selective_tool"]["mean_utility"]
                for row in per_seed
            ]
        ),
    }
    return {
        "schema_version": "selective-semantic-tool-seed-aggregate-v1",
        "model_training_seeds": sorted(seeds),
        "seed_semantics": "upstream_model_training",
        "seed_count": len(per_seed),
        "promotion_passed_count": promotion_count,
        "all_seeds_promoted": promotion_count == len(per_seed),
        "selective_tool_ready": promotion_count == len(per_seed),
        "paper_claim_ready": False,
        "frozen_protocol": protocol,
        "per_seed": per_seed,
        "aggregate_metrics": aggregate_metrics,
        "oracle_gaps": oracle_gaps,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--expected-seeds", default="20260716,20260719,20260722")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = aggregate(
        args.result,
        tuple(int(value) for value in args.expected_seeds.split(",")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
