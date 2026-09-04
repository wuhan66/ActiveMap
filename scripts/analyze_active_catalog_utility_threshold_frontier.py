#!/usr/bin/env python3
"""Screen stricter utility thresholds from completed validation rollouts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from scripts.compare_active_catalog_closed_loop import load_rows
from scripts.evaluate_active_catalog_closed_loop import metrics


def parse_candidate(value: str) -> tuple[int, Path]:
    seed, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("candidate must be SEED=PATH")
    return int(seed), Path(path)


def first_utility(row: dict[str, Any]) -> float:
    events = row.get("events", [])
    if not events:
        raise ValueError(f"trace has no events: {row.get('sample_id')}")
    return float(events[0]["predicted_acquire_utility"])


def threshold_grid(
    scores: list[float], call_rates: tuple[float, ...]
) -> list[float]:
    values = np.asarray(scores, dtype=np.float64)
    thresholds = {
        float(np.quantile(values, 1.0 - call_rate, method="higher"))
        for call_rate in call_rates
    }
    return sorted(thresholds)


def analyze(
    candidates: list[tuple[int, Path]],
    reference_path: Path,
    *,
    call_rates: tuple[float, ...],
) -> dict[str, Any]:
    candidate_rows = {
        seed: load_rows(path) for seed, path in sorted(candidates)
    }
    if len(candidate_rows) < 2:
        raise ValueError("at least two candidate seeds are required")
    reference = load_rows(reference_path)
    for seed, rows in candidate_rows.items():
        if rows.keys() != reference.keys():
            raise ValueError(f"protocol mismatch for seed {seed}")

    ordered_keys = sorted(reference)
    scores = [
        first_utility(row)
        for rows in candidate_rows.values()
        for row in rows.values()
    ]
    thresholds = threshold_grid(scores, call_rates)
    reference_metrics = metrics([reference[key] for key in ordered_keys])
    frontier = []
    for threshold in thresholds:
        per_seed = {}
        for seed, rows in candidate_rows.items():
            replayed = []
            retained = 0
            retained_multi_acquisition = 0
            for key in ordered_keys:
                row = rows[key]
                if first_utility(row) >= threshold:
                    replayed.append(row)
                    retained += 1
                    retained_multi_acquisition += int(row["acquisitions"] > 1)
                else:
                    replayed.append(reference[key])
            current = metrics(replayed)
            current["retained_episode_rate"] = retained / len(ordered_keys)
            current["retained_multi_acquisition_rate"] = (
                retained_multi_acquisition / len(ordered_keys)
            )
            per_seed[str(seed)] = current
        mean = {
            name: statistics.fmean(row[name] for row in per_seed.values())
            for name in (
                "terminal_accuracy",
                "false_edit_rate",
                "missed_edit_rate",
                "mean_acquisitions",
                "mean_cost",
                "mean_quality_gain",
                "mean_quality_cost_utility",
                "retained_episode_rate",
                "retained_multi_acquisition_rate",
            )
        }
        delta = {
            name: mean[name] - reference_metrics[name]
            for name in (
                "terminal_accuracy",
                "false_edit_rate",
                "missed_edit_rate",
                "mean_cost",
                "mean_quality_gain",
                "mean_quality_cost_utility",
            )
        }
        frontier.append(
            {
                "threshold": threshold,
                "mean_metrics": mean,
                "candidate_minus_always_stop": delta,
                "per_seed_metrics": per_seed,
            }
        )

    feasible = [
        row
        for row in frontier
        if row["candidate_minus_always_stop"]["false_edit_rate"] <= 0.0
        and row["candidate_minus_always_stop"]["mean_quality_cost_utility"] > 0.0
    ]
    selected = (
        max(
            feasible,
            key=lambda row: (
                row["candidate_minus_always_stop"]["mean_quality_cost_utility"],
                row["candidate_minus_always_stop"]["terminal_accuracy"],
                -row["mean_metrics"]["mean_cost"],
            ),
        )
        if feasible
        else None
    )
    return {
        "schema_version": "active-catalog-utility-threshold-frontier-v1",
        "split": "val",
        "test_assets_read": False,
        "reference": str(reference_path),
        "reference_metrics": reference_metrics,
        "candidate_seeds": sorted(candidate_rows),
        "screening_call_rates": call_rates,
        "replay_assumption": (
            "Episodes whose first utility score clears the stricter threshold retain "
            "their completed rollout; all others replay always-stop."
        ),
        "approximation_warning": (
            "Retained rollouts with more than one acquisition may differ under exact "
            "re-execution. Use this frontier only to choose thresholds for exact runs."
        ),
        "frontier": frontier,
        "selected_exact_run_candidate": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--call-rates",
        default="0.001,0.0025,0.005,0.0075,0.01,0.015,0.02,0.03",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    call_rates = tuple(float(value) for value in args.call_rates.split(","))
    if not call_rates or any(value <= 0.0 or value >= 1.0 for value in call_rates):
        raise ValueError("call rates must be in (0, 1)")
    result = analyze(
        args.candidate,
        args.reference,
        call_rates=call_rates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
