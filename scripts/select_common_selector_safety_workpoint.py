#!/usr/bin/env python3
"""Select one safety-calibration constraint shared by every selector seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

METRICS = (
    "mean_utility",
    "mean_regret",
    "call_rate",
    "false_call_rate",
    "harmful_call_fraction",
    "acquire_recall",
    "exact_acquire_recall",
)


def select_common_workpoint(
    payloads: list[dict[str, Any]],
    *,
    max_false_call_rate: float,
    max_harmful_call_fraction: float,
    min_acquire_recall: float,
) -> dict[str, Any]:
    if len(payloads) < 2:
        raise ValueError("common workpoint selection requires at least two seeds")
    labels = [[str(row["label"]) for row in payload["rows"]] for payload in payloads]
    if any(current != labels[0] for current in labels[1:]):
        raise ValueError("all safety sweeps must contain identical ordered candidates")

    candidates = []
    for index, label in enumerate(labels[0]):
        seed_rows = [payload["rows"][index] for payload in payloads]
        train_feasible = all(
            bool(row["calibration"]["constraints_satisfied"]) for row in seed_rows
        )
        validation_feasible = all(
            float(row["validation"]["mean_utility"]) > 0.0
            and float(row["validation"]["false_call_rate"]) <= max_false_call_rate
            and float(row["validation"]["harmful_call_fraction"])
            <= max_harmful_call_fraction
            and float(row["validation"]["acquire_recall"]) >= min_acquire_recall
            for row in seed_rows
        )
        mean = {
            metric: float(
                np.mean([float(row["validation"][metric]) for row in seed_rows])
            )
            for metric in METRICS
        }
        sample_sd = {
            metric: float(
                np.std(
                    [float(row["validation"][metric]) for row in seed_rows], ddof=1
                )
            )
            for metric in METRICS
        }
        candidates.append(
            {
                "label": label,
                "max_harmful_call_fraction": float(
                    seed_rows[0]["max_harmful_call_fraction"]
                ),
                "min_acquire_recall": float(seed_rows[0]["min_acquire_recall"]),
                "train_feasible_all_seeds": train_feasible,
                "validation_feasible_all_seeds": validation_feasible,
                "eligible": train_feasible and validation_feasible,
                "mean": mean,
                "sample_sd": sample_sd,
                "per_seed": [
                    {
                        "checkpoint": payload["checkpoint"],
                        "stop_margin": float(row["calibration"]["stop_margin"]),
                        "calibration": row["calibration"],
                        "validation": row["validation"],
                    }
                    for payload, row in zip(payloads, seed_rows, strict=True)
                ],
            }
        )

    eligible = [row for row in candidates if row["eligible"]]
    winner = (
        max(
            eligible,
            key=lambda row: (
                row["mean"]["mean_utility"],
                -row["mean"]["harmful_call_fraction"],
                row["mean"]["acquire_recall"],
            ),
        )
        if eligible
        else None
    )
    return {
        "seed_count": len(payloads),
        "constraints": {
            "max_false_call_rate": max_false_call_rate,
            "max_harmful_call_fraction": max_harmful_call_fraction,
            "min_acquire_recall": min_acquire_recall,
        },
        "promoted": winner is not None,
        "winner": winner,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweeps", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-false-call-rate", type=float, default=0.02)
    parser.add_argument("--max-harmful-call-fraction", type=float, default=0.30)
    parser.add_argument("--min-acquire-recall", type=float, default=0.01)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.sweeps]
    if any(bool(payload.get("test_assets_read", True)) for payload in payloads):
        raise ValueError("safety sweeps must certify test_assets_read=false")
    summary = {
        "schema_version": "selector-common-safety-workpoint-v1",
        "inputs": [str(path.resolve()) for path in args.sweeps],
        "selection_partition": "validation",
        "test_assets_read": False,
        **select_common_workpoint(
            payloads,
            max_false_call_rate=args.max_false_call_rate,
            max_harmful_call_fraction=args.max_harmful_call_fraction,
            min_acquire_recall=args.min_acquire_recall,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
