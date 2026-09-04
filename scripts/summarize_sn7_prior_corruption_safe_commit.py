#!/usr/bin/env python3
"""Summarize frozen Safe Commit under SN7 prior-input corruption."""

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


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _metric_values(rows: list[dict[str, Any]], policy: str) -> dict[str, float]:
    if policy not in {"direct", "safe"}:
        raise ValueError(f"unsupported policy: {policy}")
    stable = [row for row in rows if row["target_edit"] == "KEEP"]
    updates = [row for row in rows if row["target_edit"] != "KEEP"]
    if policy == "direct":
        final_iou = [float(row["committed_map_iou"]) for row in rows]
        predicted = [row["predicted_edit"] for row in rows]
    else:
        final_iou = [float(row["safe_commit_map_iou"]) for row in rows]
        predicted = [row["safe_commit_predicted_edit"] for row in rows]
    prior_iou = [float(row["prior_map_iou"]) for row in rows]
    return {
        "committed_map_iou": float(np.mean(final_iou)),
        "map_iou_delta": float(
            np.mean(np.asarray(final_iou) - np.asarray(prior_iou))
        ),
        "operation_accuracy": float(
            np.mean(
                [
                    predicted[index] == row["target_edit"]
                    for index, row in enumerate(rows)
                ]
            )
        ),
        "false_edit_rate": float(
            np.mean(
                [
                    (
                        row["predicted_edit"]
                        if policy == "direct"
                        else row["safe_commit_predicted_edit"]
                    )
                    != "KEEP"
                    for row in stable
                ]
            )
        ),
        "missed_edit_rate": float(
            np.mean(
                [
                    (
                        row["predicted_edit"]
                        if policy == "direct"
                        else row["safe_commit_predicted_edit"]
                    )
                    == "KEEP"
                    for row in updates
                ]
            )
        ),
    }


def _group_statistics(
    rows: list[dict[str, Any]],
    aoi_ids: list[str],
    policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["aoi_id"])].append(row)
    numerators = np.zeros((len(aoi_ids), len(METRICS)), dtype=np.float64)
    denominators = np.zeros_like(numerators)
    for aoi_index, aoi_id in enumerate(aoi_ids):
        for row in groups[aoi_id]:
            stable = row["target_edit"] == "KEEP"
            update = not stable
            predicted = (
                row["predicted_edit"]
                if policy == "direct"
                else row["safe_commit_predicted_edit"]
            )
            final_iou = float(
                row["committed_map_iou"]
                if policy == "direct"
                else row["safe_commit_map_iou"]
            )
            numerators[aoi_index] += (
                final_iou,
                final_iou - float(row["prior_map_iou"]),
                float(predicted == row["target_edit"]),
                float(stable and predicted != "KEEP"),
                float(update and predicted == "KEEP"),
            )
            denominators[aoi_index] += (
                1.0,
                1.0,
                1.0,
                float(stable),
                float(update),
            )
    return numerators, denominators


def _shared_bootstrap(
    reference: dict[int, list[dict[str, Any]]],
    candidate: dict[int, list[dict[str, Any]]],
    *,
    reference_policy: str,
    candidate_policy: str,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    model_seeds = sorted(reference)
    aoi_ids = sorted(
        set.intersection(
            *(
                {str(row["aoi_id"]) for row in reference[model_seed]}
                for model_seed in model_seeds
            ),
            *(
                {str(row["aoi_id"]) for row in candidate[model_seed]}
                for model_seed in model_seeds
            ),
        )
    )
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(aoi_ids),
        np.full(len(aoi_ids), 1.0 / len(aoi_ids)),
        size=repetitions,
    ).astype(np.float64)
    seed_deltas = []
    for model_seed in model_seeds:
        ref_num, ref_den = _group_statistics(
            reference[model_seed], aoi_ids, reference_policy
        )
        cand_num, cand_den = _group_statistics(
            candidate[model_seed], aoi_ids, candidate_policy
        )
        seed_deltas.append(
            (counts @ cand_num) / (counts @ cand_den)
            - (counts @ ref_num) / (counts @ ref_den)
        )
    draws = np.mean(seed_deltas, axis=0)
    return {
        metric: {
            "ci95_low": float(np.quantile(draws[:, index], 0.025)),
            "ci95_high": float(np.quantile(draws[:, index], 0.975)),
        }
        for index, metric in enumerate(METRICS)
    }


def summarize(
    root: Path,
    *,
    severities: list[int],
    model_seeds: list[int],
    repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    rows: dict[int, dict[int, list[dict[str, Any]]]] = {}
    thresholds = {}
    for severity in severities:
        rows[severity] = {}
        for model_seed in model_seeds:
            run = root / f"severity{severity}_seed{model_seed}"
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            if summary["test_assets_read"] or summary["split"] != "val":
                raise ValueError(f"invalid validation contract: {run}")
            current_rows = _read_rows(run / "per_sample.jsonl")
            if len(current_rows) != summary["sample_count"]:
                raise ValueError(f"sample count mismatch: {run}")
            rows[severity][model_seed] = current_rows
            thresholds[model_seed] = summary["selected_threshold"]
    results = []
    clean = rows[severities[0]]
    for severity in severities:
        per_seed = {
            model_seed: {
                policy: _metric_values(rows[severity][model_seed], policy)
                for policy in ("direct", "safe")
            }
            for model_seed in model_seeds
        }
        aggregate = {
            policy: {
                metric: {
                    "mean": float(
                        np.mean(
                            [
                                per_seed[model_seed][policy][metric]
                                for model_seed in model_seeds
                            ]
                        )
                    ),
                    "std": float(
                        np.std(
                            [
                                per_seed[model_seed][policy][metric]
                                for model_seed in model_seeds
                            ],
                            ddof=1,
                        )
                    ),
                }
                for metric in METRICS
            }
            for policy in ("direct", "safe")
        }
        safe_minus_direct_ci = _shared_bootstrap(
            rows[severity],
            rows[severity],
            reference_policy="direct",
            candidate_policy="safe",
            repetitions=repetitions,
            seed=bootstrap_seed + severity,
        )
        safe_corrupt_minus_clean_ci = _shared_bootstrap(
            clean,
            rows[severity],
            reference_policy="safe",
            candidate_policy="safe",
            repetitions=repetitions,
            seed=bootstrap_seed + 100 + severity,
        )
        results.append(
            {
                "max_pixels": severity,
                "per_seed": per_seed,
                "aggregate": aggregate,
                "safe_minus_direct": {
                    metric: {
                        "observed": (
                            aggregate["safe"][metric]["mean"]
                            - aggregate["direct"][metric]["mean"]
                        ),
                        **safe_minus_direct_ci[metric],
                    }
                    for metric in METRICS
                },
                "safe_corrupt_minus_clean": {
                    metric: {
                        "observed": (
                            aggregate["safe"][metric]["mean"]
                            - np.mean(
                                [
                                    _metric_values(clean[model_seed], "safe")[metric]
                                    for model_seed in model_seeds
                                ]
                            )
                        ),
                        **safe_corrupt_minus_clean_ci[metric],
                    }
                    for metric in METRICS
                },
            }
        )
    return {
        "schema_version": "sn7-prior-corruption-frozen-safe-commit-summary-v1",
        "protocol": {
            "split": "val",
            "model_seeds": model_seeds,
            "severities_pixels": severities,
            "selected_thresholds": thresholds,
            "calibration_source": "train predictions only",
            "threshold_recalibrated_on_corruption": False,
            "bootstrap_unit": "aoi_id shared across model seeds",
            "bootstrap_repetitions": repetitions,
            "test_assets_read": False,
        },
        "severities": results,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Frozen Safe Commit under Prior-Input Corruption",
        "",
        "| Shift | Direct IoU | Safe IoU | Safe-Direct IoU | Direct FE | "
        "Safe FE | Safe-Direct FE | Safe missed edit |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["severities"]:
        direct = row["aggregate"]["direct"]
        safe = row["aggregate"]["safe"]
        iou = row["safe_minus_direct"]["committed_map_iou"]
        false_edit = row["safe_minus_direct"]["false_edit_rate"]
        lines.append(
            f"| {row['max_pixels']} | "
            f"{direct['committed_map_iou']['mean']:.6f} | "
            f"{safe['committed_map_iou']['mean']:.6f} | "
            f"{iou['observed']:+.6f} "
            f"[{iou['ci95_low']:+.6f},{iou['ci95_high']:+.6f}] | "
            f"{direct['false_edit_rate']['mean']:.6f} | "
            f"{safe['false_edit_rate']['mean']:.6f} | "
            f"{false_edit['observed']:+.6f} "
            f"[{false_edit['ci95_low']:+.6f},{false_edit['ci95_high']:+.6f}] | "
            f"{safe['missed_edit_rate']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The logistic model and threshold are fitted on train predictions "
            "only and frozen across all corruption severities. Test assets "
            "were not read.",
            "",
        ]
    )
    return "\n".join(lines)


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
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    args = parser.parse_args()
    payload = summarize(
        args.root,
        severities=args.severities,
        model_seeds=args.model_seeds,
        repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    (args.root / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.root / "summary.md").write_text(_markdown(payload), encoding="utf-8")
    with (args.root / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "max_pixels",
                "direct_iou",
                "safe_iou",
                "safe_minus_direct_iou",
                "direct_false_edit",
                "safe_false_edit",
                "safe_minus_direct_false_edit",
                "safe_missed_edit",
            ]
        )
        for row in payload["severities"]:
            writer.writerow(
                [
                    row["max_pixels"],
                    row["aggregate"]["direct"]["committed_map_iou"]["mean"],
                    row["aggregate"]["safe"]["committed_map_iou"]["mean"],
                    row["safe_minus_direct"]["committed_map_iou"]["observed"],
                    row["aggregate"]["direct"]["false_edit_rate"]["mean"],
                    row["aggregate"]["safe"]["false_edit_rate"]["mean"],
                    row["safe_minus_direct"]["false_edit_rate"]["observed"],
                    row["aggregate"]["safe"]["missed_edit_rate"]["mean"],
                ]
            )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
