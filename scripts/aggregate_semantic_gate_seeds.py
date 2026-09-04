#!/usr/bin/env python3
"""Aggregate aligned semantic Belief-gate results across model-training seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

VARIANTS = (
    "belief_only",
    "belief_plus_weak_tools",
    "belief_plus_semantic",
    "task_deranged_semantic_mismatch",
    "belief_plus_all",
    "all_with_task_deranged_semantic_mismatch",
)
METRICS = ("accuracy", "macro_f1", "false_edit_rate", "missed_edit_rate")
FROZEN_KEYS = (
    "schema_version",
    "selection_protocol",
    "semantic_mismatch_protocol",
    "validation_adaptation_stage",
    "guard_protocol",
    "fixed_threshold",
    "C_grid",
    "train_examples",
    "train_tasks",
    "val_examples",
    "commit_threshold_grid",
    "train_false_edit_cap",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate(
    paths: list[Path],
    expected_seeds: tuple[int, ...],
    *,
    validation_key: str = "validation",
    gate_key: str = "gate",
    interaction_gate_key: str = "interaction_gate",
) -> dict[str, Any]:
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
        validation = row.get(validation_key, {})
        for variant in VARIANTS:
            if variant not in validation:
                raise ValueError(f"missing validation variant {variant}: {path}")
            if any(metric not in validation[variant] for metric in METRICS):
                raise ValueError(f"incomplete validation metrics for {variant}: {path}")
        per_seed.append(
            {
                "model_training_seed": row["model_training_seed"],
                "cv_seed": row.get("cv_seed"),
                "direct_gate_passed": bool(row.get(gate_key, {}).get("passed")),
                "interaction_gate_passed": bool(row.get(interaction_gate_key, {}).get("passed")),
                "validation": {
                    variant: {metric: float(validation[variant][metric]) for metric in METRICS}
                    for variant in VARIANTS
                },
                "deltas": {
                    key: float(value) for key, value in validation.get("deltas", {}).items()
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
    delta_keys = tuple(per_seed[0]["deltas"])
    if any(tuple(row["deltas"]) != delta_keys for row in per_seed[1:]):
        raise ValueError("seed results expose different delta metrics")
    aggregate_deltas = {
        key: _mean_std([row["deltas"][key] for row in per_seed]) for key in delta_keys
    }
    direct_count = sum(row["direct_gate_passed"] for row in per_seed)
    interaction_count = sum(row["interaction_gate_passed"] for row in per_seed)
    return {
        "schema_version": "aligned-semantic-gate-seed-aggregate-v1",
        "model_training_seeds": sorted(seeds),
        "seed_semantics": "upstream_model_training",
        "seed_count": len(per_seed),
        "validation_key": validation_key,
        "gate_key": gate_key,
        "interaction_gate_key": interaction_gate_key,
        "direct_gate_passed_count": direct_count,
        "interaction_gate_passed_count": interaction_count,
        "all_direct_gates_passed": direct_count == len(per_seed),
        "all_interaction_gates_passed": interaction_count == len(per_seed),
        "semantic_gate_ready": direct_count == len(per_seed),
        "paper_claim_ready": False,
        "frozen_protocol": protocol,
        "per_seed": per_seed,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_deltas": aggregate_deltas,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--expected-seeds", default="20260716,20260719,20260722")
    parser.add_argument("--validation-key", default="validation")
    parser.add_argument("--gate-key", default="gate")
    parser.add_argument("--interaction-gate-key", default="interaction_gate")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    seeds = tuple(int(value) for value in args.expected_seeds.split(","))
    payload = aggregate(
        args.result,
        seeds,
        validation_key=args.validation_key,
        gate_key=args.gate_key,
        interaction_gate_key=args.interaction_gate_key,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
