#!/usr/bin/env python3
"""Summarize frozen SN7 prior-rendering corruption evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

METRICS = (
    "committed_map_iou",
    "map_iou_delta",
    "operation_accuracy",
    "false_edit_rate",
    "missed_edit_rate",
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    stable = [row for row in rows if row["target_edit"] == "KEEP"]
    updates = [row for row in rows if row["target_edit"] != "KEEP"]
    return {
        "committed_map_iou": float(
            np.mean([row["committed_map_iou"] for row in rows])
        ),
        "map_iou_delta": float(np.mean([row["map_iou_delta"] for row in rows])),
        "operation_accuracy": float(
            np.mean([row["operation_correct"] for row in rows])
        ),
        "false_edit_rate": float(np.mean([row["false_edit"] for row in stable])),
        "missed_edit_rate": float(
            np.mean([row["missed_edit"] for row in updates])
        ),
    }


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["aoi_id"])].append(row)
    return groups


def _group_sufficient_statistics(
    groups: dict[str, list[dict[str, Any]]],
    aoi_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    numerators = np.zeros((len(aoi_ids), len(METRICS)), dtype=np.float64)
    denominators = np.zeros_like(numerators)
    for aoi_index, aoi_id in enumerate(aoi_ids):
        for row in groups[aoi_id]:
            stable = row["target_edit"] == "KEEP"
            update = not stable
            numerators[aoi_index] += (
                float(row["committed_map_iou"]),
                float(row["map_iou_delta"]),
                float(row["operation_correct"]),
                float(row["false_edit"]) if stable else 0.0,
                float(row["missed_edit"]) if update else 0.0,
            )
            denominators[aoi_index] += (
                1.0,
                1.0,
                1.0,
                float(stable),
                float(update),
            )
    return numerators, denominators


def _bootstrap_delta(
    reference: dict[int, list[dict[str, Any]]],
    candidate: dict[int, list[dict[str, Any]]],
    *,
    repetitions: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, float]]:
    seeds = sorted(reference)
    aoi_ids = sorted(
        set.intersection(
            *(set(_group_rows(reference[seed])) for seed in seeds),
            *(set(_group_rows(candidate[seed])) for seed in seeds),
        )
    )
    if len(aoi_ids) < 2:
        raise ValueError("at least two shared AOIs are required")
    grouped = {
        seed: {
            "reference": _group_rows(reference[seed]),
            "candidate": _group_rows(candidate[seed]),
        }
        for seed in seeds
    }
    rng = np.random.default_rng(bootstrap_seed)
    counts = rng.multinomial(
        len(aoi_ids),
        np.full(len(aoi_ids), 1.0 / len(aoi_ids)),
        size=repetitions,
    ).astype(np.float64)
    seed_draws = []
    for seed in seeds:
        reference_num, reference_den = _group_sufficient_statistics(
            grouped[seed]["reference"], aoi_ids
        )
        candidate_num, candidate_den = _group_sufficient_statistics(
            grouped[seed]["candidate"], aoi_ids
        )
        reference_values = (counts @ reference_num) / (counts @ reference_den)
        candidate_values = (counts @ candidate_num) / (counts @ candidate_den)
        seed_draws.append(candidate_values - reference_values)
    draws = np.mean(seed_draws, axis=0)
    return {
        metric: {
            "ci95_low": float(np.quantile(draws[:, index], 0.025)),
            "ci95_high": float(np.quantile(draws[:, index], 0.975)),
        }
        for index, metric in enumerate(METRICS)
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SN7 Prior-Input Corruption Robustness",
        "",
        "The intervention translates only the rendered prior-map model input. "
        "The editable prior used for writeback, target geometry, and operation "
        "labels remain unchanged.",
        "",
        "| Max shift (px) | Committed IoU | Delta vs 0 | Operation acc. | "
        "False edit | Missed edit |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for severity in payload["severities"]:
        aggregate = severity["aggregate"]
        delta = severity["delta_vs_zero"]["committed_map_iou"]
        lines.append(
            f"| {severity['max_pixels']} | "
            f"{aggregate['committed_map_iou']['mean']:.6f} | "
            f"{delta['observed']:+.6f} "
            f"[{delta['ci95_low']:+.6f}, {delta['ci95_high']:+.6f}] | "
            f"{aggregate['operation_accuracy']['mean']:.6f} | "
            f"{aggregate['false_edit_rate']['mean']:.6f} | "
            f"{aggregate['missed_edit_rate']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "All intervals use shared AOI bootstrap resampling across the three "
            "model seeds. Validation only; test assets were not read.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(
    root: Path,
    *,
    severities: list[int],
    model_seeds: list[int],
    repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    data: dict[int, dict[int, list[dict[str, Any]]]] = {}
    summaries: dict[int, dict[int, dict[str, Any]]] = {}
    for severity in severities:
        data[severity] = {}
        summaries[severity] = {}
        for model_seed in model_seeds:
            run = root / f"severity{severity}_model_seed{model_seed}"
            summary_path = run / "summary.json"
            rows_path = run / "per_sample.jsonl"
            if not summary_path.is_file() or not rows_path.is_file():
                raise FileNotFoundError(f"incomplete run: {run}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary["split"] != "val" or summary["test_assets_read"]:
                raise ValueError(f"invalid validation contract: {run}")
            corruption = summary["prior_input_corruption"]
            if corruption["max_pixels"] != severity:
                raise ValueError(f"severity mismatch: {run}")
            if (
                corruption["writeback_prior_corrupted"]
                or corruption["target_geometry_corrupted"]
            ):
                raise ValueError(f"invalid corruption scope: {run}")
            rows = _load_rows(rows_path)
            if len(rows) != summary["sample_count"]:
                raise ValueError(f"sample count mismatch: {run}")
            data[severity][model_seed] = rows
            summaries[severity][model_seed] = summary
    reference = data[severities[0]]
    results = []
    for severity in severities:
        per_seed = {
            seed: _aggregate(data[severity][seed]) for seed in model_seeds
        }
        aggregate = {
            metric: {
                "mean": float(np.mean([per_seed[seed][metric] for seed in model_seeds])),
                "std": float(
                    np.std(
                        [per_seed[seed][metric] for seed in model_seeds],
                        ddof=1,
                    )
                ),
            }
            for metric in METRICS
        }
        intervals = _bootstrap_delta(
            reference,
            data[severity],
            repetitions=repetitions,
            bootstrap_seed=bootstrap_seed + severity,
        )
        delta_vs_zero = {}
        for metric in METRICS:
            observed = float(
                np.mean(
                    [
                        per_seed[seed][metric]
                        - _aggregate(reference[seed])[metric]
                        for seed in model_seeds
                    ]
                )
            )
            delta_vs_zero[metric] = {"observed": observed, **intervals[metric]}
        results.append(
            {
                "max_pixels": severity,
                "per_seed": per_seed,
                "aggregate": aggregate,
                "delta_vs_zero": delta_vs_zero,
            }
        )
    return {
        "schema_version": "sn7-prior-input-corruption-summary-v1",
        "protocol": {
            "split": "val",
            "model_seeds": model_seeds,
            "severities_pixels": severities,
            "corruption_scope": "rendered prior-map model input only",
            "writeback_prior_corrupted": False,
            "target_geometry_corrupted": False,
            "bootstrap_unit": "aoi_id shared across model seeds",
            "bootstrap_repetitions": repetitions,
            "bootstrap_seed": bootstrap_seed,
            "test_assets_read": False,
        },
        "severities": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--severities", type=int, nargs="+", default=[0, 4, 8, 16])
    parser.add_argument(
        "--model-seeds",
        type=int,
        nargs="+",
        default=[20260725, 20260726, 20260727],
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    args = parser.parse_args()
    payload = summarize(
        args.root,
        severities=args.severities,
        model_seeds=args.model_seeds,
        repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    json_path = args.root / "summary.json"
    markdown_path = args.root / "summary.md"
    csv_path = args.root / "summary.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "max_pixels",
                *METRICS,
                *(f"delta_{metric}" for metric in METRICS),
            ]
        )
        for severity in payload["severities"]:
            writer.writerow(
                [
                    severity["max_pixels"],
                    *(
                        severity["aggregate"][metric]["mean"]
                        for metric in METRICS
                    ),
                    *(
                        severity["delta_vs_zero"][metric]["observed"]
                        for metric in METRICS
                    ),
                ]
            )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
