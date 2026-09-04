#!/usr/bin/env python3
"""Stratify paired executable writebacks by target edit operation."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.compare_agent_writebacks import _load


METRICS = (
    "raster_iou",
    "episode_utility_v2_balanced",
    "episode_utility_v2_safety",
    "false_edit",
    "missed_edit",
    "spent_cost",
)
DIAGNOSTICS = (
    "terminal_accuracy",
    "action_flip",
    "corrected",
    "regressed",
)
DELTA_METRICS = METRICS + DIAGNOSTICS
TARGETS = ("KEEP", "ADD", "DELETE", "RESHAPE")


def _target_operation(value: Any) -> str:
    target = str(value)
    if target == "REJECT":
        return "KEEP"
    if target.startswith("COMMIT:"):
        target = target.split(":", 1)[1]
    if target not in TARGETS:
        raise ValueError(f"unsupported target action: {value!r}")
    return target


def _parse_pair(value: str) -> tuple[int, Path, Path]:
    seed_text, separator, paths = value.partition("=")
    baseline, comma, candidate = paths.partition(",")
    if not separator or not comma:
        raise argparse.ArgumentTypeError(
            "pair must be SEED=BASELINE.jsonl,CANDIDATE.jsonl"
        )
    return int(seed_text), Path(baseline), Path(candidate)


def stratify(
    pairs: list[tuple[int, Path, Path]],
    *,
    repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if len({seed for seed, _, _ in pairs}) != len(pairs):
        raise ValueError("model seeds must be unique")
    loaded = {
        seed: (_load(baseline), _load(candidate))
        for seed, baseline, candidate in pairs
    }
    blocks: dict[str, dict[str, list[dict[str, float]]]] = {
        target: defaultdict(list) for target in TARGETS
    }
    counts = {target: 0 for target in TARGETS}
    absolute: dict[str, dict[str, list[float]]] = {
        target: {
            f"{branch}_{metric}": []
            for branch in ("baseline", "candidate")
            for metric in METRICS
        }
        for target in TARGETS
    }
    for seed, (baseline, candidate) in sorted(loaded.items()):
        if baseline.keys() != candidate.keys():
            raise ValueError(f"support mismatch for seed {seed}")
        per_block: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
        for key in baseline:
            left = baseline[key]
            right = candidate[key]
            target = _target_operation(left["target"])
            if target != _target_operation(right["target"]):
                raise ValueError(f"target mismatch for seed {seed}, key {key}")
            if str(left["aoi_id"]) != str(right["aoi_id"]):
                raise ValueError(f"AOI mismatch for seed {seed}, key {key}")
            counts[target] += 1
            delta = {
                metric: float(right[metric]) - float(left[metric])
                for metric in METRICS
            }
            baseline_correct = left["prediction"] == left["target"]
            candidate_correct = right["prediction"] == right["target"]
            delta.update(
                {
                    "terminal_accuracy": float(candidate_correct)
                    - float(baseline_correct),
                    "action_flip": float(
                        left["prediction"] != right["prediction"]
                    ),
                    "corrected": float(
                        not baseline_correct and candidate_correct
                    ),
                    "regressed": float(
                        baseline_correct and not candidate_correct
                    ),
                }
            )
            per_block[(target, str(left["aoi_id"]))].append(delta)
            for metric in METRICS:
                absolute[target][f"baseline_{metric}"].append(float(left[metric]))
                absolute[target][f"candidate_{metric}"].append(float(right[metric]))
        for (target, aoi_id), rows in per_block.items():
            blocks[target][f"{seed}:{aoi_id}"].append(
                {
                    metric: statistics.fmean(row[metric] for row in rows)
                    for metric in DELTA_METRICS
                }
            )

    rng = np.random.default_rng(bootstrap_seed)
    results = {}
    for target in TARGETS:
        block_values = [
            rows[0] for _, rows in sorted(blocks[target].items())
        ]
        if not block_values:
            continue
        matrix = np.asarray(
            [[row[metric] for metric in DELTA_METRICS] for row in block_values],
            dtype=np.float64,
        )
        sampled = rng.integers(
            0, len(matrix), size=(repetitions, len(matrix))
        )
        draws = matrix[sampled].mean(axis=1)
        deltas = {}
        for index, metric in enumerate(DELTA_METRICS):
            deltas[metric] = {
                "observed_delta": float(matrix[:, index].mean()),
                "ci95_low": float(np.quantile(draws[:, index], 0.025)),
                "ci95_high": float(np.quantile(draws[:, index], 0.975)),
            }
        results[target] = {
            "record_count": counts[target],
            "seed_aoi_block_count": len(matrix),
            "absolute": {
                key: statistics.fmean(values)
                for key, values in absolute[target].items()
            },
            "candidate_minus_baseline": deltas,
        }
    return {
        "schema_version": "agent-writeback-operation-stratified-v1",
        "model_seeds": sorted(loaded),
        "seed_count": len(loaded),
        "metrics": list(METRICS),
        "diagnostics": list(DIAGNOSTICS),
        "bootstrap": {
            "unit": "seed-AOI block",
            "repetitions": repetitions,
            "seed": bootstrap_seed,
        },
        "operations": results,
        "split": "val",
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--pair", action="append", type=_parse_pair, required=True)
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = stratify(
        args.pair,
        repetitions=args.repetitions,
        bootstrap_seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
