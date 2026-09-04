#!/usr/bin/env python3
"""Audit threshold recalibration across every eligible target-train AOI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.calibrate_sn7_changemamba_safe_commit import (
    choose_threshold,
    feature_matrix,
    policy_metrics,
    predict_logistic,
    read_rows,
)
from scripts.evaluate_sn7_safe_commit_recalibration_curve import source_gate

METRICS = (
    "map_iou_delta",
    "committed_map_iou",
    "operation_accuracy",
    "false_edit_rate",
    "missed_edit_rate",
    "wrong_edit_rate",
    "accepted_commit_rate",
)


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def eligible_aoi_rows(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["aoi_id"]), []).append(row)
    return {
        aoi_id: values
        for aoi_id, values in grouped.items()
        if {row["target_edit"] == "KEEP" for row in values} == {False, True}
    }


def evaluate(
    source_paths: list[Path],
    target_train_paths: list[Path],
    target_val_paths: list[Path],
    *,
    source_backend: str,
    target_backend: str,
    folds: int,
    l2: float,
) -> dict[str, Any]:
    if not (
        len(source_paths) == len(target_train_paths) == len(target_val_paths)
        and len(source_paths) >= 2
    ):
        raise ValueError("source, target-train, and target-val runs must align")
    runs = []
    evaluations = []
    for run_index, (source_path, target_train_path, target_val_path) in enumerate(
        zip(source_paths, target_train_paths, target_val_paths, strict=True)
    ):
        source_rows = read_rows(source_path, expected_split="train")
        target_train = read_rows(target_train_path, expected_split="train")
        target_val = read_rows(target_val_path, expected_split="val")
        model, source_threshold = source_gate(source_rows, folds=folds, l2=l2)
        val_scores = predict_logistic(model, feature_matrix(target_val))
        baseline, _ = policy_metrics(
            target_val, val_scores, threshold=source_threshold
        )
        grouped = eligible_aoi_rows(target_train)
        run_evaluations = []
        for aoi_id in sorted(grouped):
            calibration_rows = grouped[aoi_id]
            calibration_scores = predict_logistic(
                model, feature_matrix(calibration_rows)
            )
            threshold, _ = choose_threshold(
                calibration_rows, calibration_scores
            )
            metrics, _ = policy_metrics(
                target_val, val_scores, threshold=threshold
            )
            row = {
                "run_index": run_index,
                "calibration_aoi_id": aoi_id,
                "calibration_samples": len(calibration_rows),
                "threshold": threshold,
                "baseline": baseline,
                "metrics": metrics,
                "delta": {
                    metric: metrics[metric] - baseline[metric]
                    for metric in METRICS
                },
            }
            run_evaluations.append(row)
            evaluations.append(row)
        runs.append(
            {
                "run_index": run_index,
                "source_train": str(source_path),
                "target_train": str(target_train_path),
                "target_val": str(target_val_path),
                "source_threshold": source_threshold,
                "baseline": baseline,
                "eligible_calibration_aois": len(grouped),
                "evaluations": run_evaluations,
            }
        )
    summary = {
        "threshold": distribution([row["threshold"] for row in evaluations]),
        "calibration_samples": distribution(
            [float(row["calibration_samples"]) for row in evaluations]
        ),
        "metrics": {
            metric: distribution(
                [row["metrics"][metric] for row in evaluations]
            )
            for metric in METRICS
        },
        "deltas": {
            metric: distribution(
                [row["delta"][metric] for row in evaluations]
            )
            for metric in METRICS
        },
        "rates": {
            "positive_map_iou_gain": float(
                np.mean(
                    [row["metrics"]["map_iou_delta"] > 0.0 for row in evaluations]
                )
            ),
            "map_iou_improved_vs_source_only": float(
                np.mean(
                    [row["delta"]["map_iou_delta"] > 0.0 for row in evaluations]
                )
            ),
            "false_edit_reduced_vs_source_only": float(
                np.mean(
                    [row["delta"]["false_edit_rate"] < 0.0 for row in evaluations]
                )
            ),
            "joint_quality_and_false_edit_improvement": float(
                np.mean(
                    [
                        row["delta"]["map_iou_delta"] > 0.0
                        and row["delta"]["false_edit_rate"] < 0.0
                        for row in evaluations
                    ]
                )
            ),
            "false_edit_at_most_0p05": float(
                np.mean(
                    [
                        row["metrics"]["false_edit_rate"] <= 0.05
                        for row in evaluations
                    ]
                )
            ),
        },
    }
    return {
        "schema_version": "sn7-safe-commit-single-aoi-robustness-v1",
        "source_backend": source_backend,
        "target_backend": target_backend,
        "adaptation": "source weights frozen; each target train AOI selects threshold alone",
        "run_count": len(runs),
        "evaluation_count": len(evaluations),
        "folds": folds,
        "l2": l2,
        "runs": runs,
        "summary": summary,
        "test_assets_read": False,
    }


def markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "| Statistic | Mean | Median | 5% | 95% | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, values in (
        ("Map-IoU gain", summary["metrics"]["map_iou_delta"]),
        ("False edit", summary["metrics"]["false_edit_rate"]),
        ("Missed edit", summary["metrics"]["missed_edit_rate"]),
        ("Map-IoU delta vs source-only", summary["deltas"]["map_iou_delta"]),
    ):
        lines.append(
            f"| {label} | {values['mean']:+.6f} | {values['median']:+.6f} | "
            f"{values['q05']:+.6f} | {values['q95']:+.6f} | "
            f"{values['min']:+.6f} | {values['max']:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Evaluations: {result['evaluation_count']}",
            "- Positive map-IoU gain: "
            f"{summary['rates']['positive_map_iou_gain']:.2%}",
            "- Joint quality + false-edit improvement: "
            f"{summary['rates']['joint_quality_and_false_edit_improvement']:.2%}",
            "- False edit <= 0.05: "
            f"{summary['rates']['false_edit_at_most_0p05']:.2%}",
        ]
    )
    return "\n".join(lines) + "\n"


def plot(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        row for run in result["runs"] for row in run["evaluations"]
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].scatter(
        [row["metrics"]["false_edit_rate"] for row in rows],
        [row["metrics"]["map_iou_delta"] for row in rows],
        alpha=0.55,
        s=18,
    )
    axes[0].set_xlabel("False-edit rate")
    axes[0].set_ylabel("Map-IoU gain")
    axes[1].hist(
        [row["delta"]["map_iou_delta"] for row in rows],
        bins=24,
        edgecolor="white",
    )
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Map-IoU delta vs source-only")
    axes[1].set_ylabel("Single-AOI calibrations")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"Single-AOI robustness: {result['source_backend']} -> "
        f"{result['target_backend']}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--source-train", nargs="+", type=Path, required=True)
    parser.add_argument("--target-train", nargs="+", type=Path, required=True)
    parser.add_argument("--target-val", nargs="+", type=Path, required=True)
    parser.add_argument("--source-backend", required=True)
    parser.add_argument("--target-backend", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--output-figure", type=Path)
    args = parser.parse_args()
    result = evaluate(
        args.source_train,
        args.target_train,
        args.target_val,
        source_backend=args.source_backend,
        target_backend=args.target_backend,
        folds=args.folds,
        l2=args.l2,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if args.output_markdown is not None:
        args.output_markdown.write_text(markdown(result), encoding="utf-8")
    if args.output_figure is not None:
        plot(result, args.output_figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
