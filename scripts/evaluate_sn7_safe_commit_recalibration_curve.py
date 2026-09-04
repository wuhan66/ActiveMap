#!/usr/bin/env python3
"""Measure target-AOI threshold recalibration for a frozen source-backend gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.calibrate_sn7_changemamba_safe_commit import (
    choose_threshold,
    feature_matrix,
    fit_logistic,
    grouped_folds,
    policy_metrics,
    predict_logistic,
    read_rows,
)

METRICS = (
    "committed_map_iou",
    "map_iou_delta",
    "operation_accuracy",
    "false_edit_rate",
    "missed_edit_rate",
    "wrong_edit_rate",
    "accepted_commit_rate",
    "beneficial_commit_precision",
    "beneficial_commit_recall",
)


def source_gate(
    rows: list[dict[str, Any]], *, folds: int, l2: float
) -> tuple[dict[str, np.ndarray], float]:
    features = feature_matrix(rows)
    labels = np.asarray(
        [row["map_iou_delta"] > 1e-8 for row in rows], dtype=np.float64
    )
    fold_ids = grouped_folds(rows, folds)
    oof_scores = np.zeros(len(rows), dtype=np.float64)
    for fold in range(folds):
        held_out = fold_ids == fold
        model = fit_logistic(features[~held_out], labels[~held_out], l2=l2)
        oof_scores[held_out] = predict_logistic(model, features[held_out])
    threshold, _ = choose_threshold(rows, oof_scores)
    return fit_logistic(features, labels, l2=l2), threshold


def ordered_calibration_aois(
    rows: list[dict[str, Any]], *, selection_seed: int
) -> list[str]:
    by_aoi: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_aoi.setdefault(str(row["aoi_id"]), []).append(row)
    eligible = [
        aoi_id
        for aoi_id, values in by_aoi.items()
        if {row["target_edit"] == "KEEP" for row in values} == {False, True}
    ]
    return sorted(
        eligible,
        key=lambda aoi_id: hashlib.sha256(
            f"{selection_seed}:{aoi_id}".encode("utf-8")
        ).hexdigest(),
    )


def parse_counts(value: str) -> list[int | str]:
    result: list[int | str] = []
    for item in value.split(","):
        item = item.strip().lower()
        if item == "all":
            result.append(item)
        else:
            count = int(item)
            if count < 0:
                raise ValueError("target AOI counts must be nonnegative")
            result.append(count)
    if not result or result[0] != 0 or len(set(result)) != len(result):
        raise ValueError("counts must be unique and start with zero")
    return result


def run_curve(
    source_paths: list[Path],
    target_train_paths: list[Path],
    target_val_paths: list[Path],
    *,
    source_backend: str,
    target_backend: str,
    counts: list[int | str],
    folds: int,
    l2: float,
    selection_seed: int,
) -> dict[str, Any]:
    if not (
        len(source_paths) == len(target_train_paths) == len(target_val_paths)
        and len(source_paths) >= 2
    ):
        raise ValueError("source, target-train, and target-val runs must align")
    runs = []
    for index, (source_path, train_path, val_path) in enumerate(
        zip(source_paths, target_train_paths, target_val_paths, strict=True)
    ):
        source_rows = read_rows(source_path, expected_split="train")
        target_train = read_rows(train_path, expected_split="train")
        target_val = read_rows(val_path, expected_split="val")
        model, source_threshold = source_gate(source_rows, folds=folds, l2=l2)
        train_scores = predict_logistic(model, feature_matrix(target_train))
        val_scores = predict_logistic(model, feature_matrix(target_val))
        ordered_aois = ordered_calibration_aois(
            target_train, selection_seed=selection_seed + index
        )
        points = []
        for count in counts:
            if count == 0:
                threshold = source_threshold
                used_aois: list[str] = []
            else:
                used_aois = (
                    sorted({str(row["aoi_id"]) for row in target_train})
                    if count == "all"
                    else ordered_aois[: int(count)]
                )
                if not used_aois:
                    raise ValueError("target calibration AOI subset is empty")
                keep = np.asarray(
                    [str(row["aoi_id"]) in set(used_aois) for row in target_train]
                )
                subset_rows = [
                    row for row, selected in zip(target_train, keep, strict=True)
                    if selected
                ]
                threshold, _ = choose_threshold(
                    subset_rows, train_scores[keep]
                )
            metrics, _ = policy_metrics(
                target_val, val_scores, threshold=threshold
            )
            points.append(
                {
                    "requested_target_aois": count,
                    "used_target_aois": len(used_aois),
                    "target_calibration_samples": sum(
                        str(row["aoi_id"]) in set(used_aois)
                        for row in target_train
                    ),
                    "threshold": threshold,
                    "metrics": metrics,
                }
            )
        runs.append(
            {
                "run_index": index,
                "source_train": str(source_path),
                "target_train": str(train_path),
                "target_val": str(val_path),
                "eligible_target_calibration_aois": len(ordered_aois),
                "points": points,
            }
        )

    aggregate = []
    for point_index, count in enumerate(counts):
        aggregate.append(
            {
                "requested_target_aois": count,
                "used_target_aois": {
                    "mean": float(
                        np.mean(
                            [
                                run["points"][point_index]["used_target_aois"]
                                for run in runs
                            ]
                        )
                    )
                },
                "threshold": {
                    "mean": float(
                        np.mean(
                            [
                                run["points"][point_index]["threshold"]
                                for run in runs
                            ]
                        )
                    ),
                    "std": float(
                        np.std(
                            [
                                run["points"][point_index]["threshold"]
                                for run in runs
                            ],
                            ddof=1,
                        )
                    ),
                },
                "metrics": {
                    metric: {
                        "mean": float(
                            np.mean(
                                [
                                    run["points"][point_index]["metrics"][metric]
                                    for run in runs
                                ]
                            )
                        ),
                        "std": float(
                            np.std(
                                [
                                    run["points"][point_index]["metrics"][metric]
                                    for run in runs
                                ],
                                ddof=1,
                            )
                        ),
                    }
                    for metric in METRICS
                },
            }
        )
    return {
        "schema_version": "sn7-safe-commit-threshold-recalibration-curve-v1",
        "source_backend": source_backend,
        "target_backend": target_backend,
        "adaptation": "source weights frozen; target train AOIs select threshold only",
        "run_count": len(runs),
        "folds": folds,
        "l2": l2,
        "selection_seed": selection_seed,
        "runs": runs,
        "aggregate": aggregate,
        "test_assets_read": False,
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "| Target AOIs | Threshold | Map-IoU gain | False edit | Missed edit |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in result["aggregate"]:
        metrics = point["metrics"]
        lines.append(
            f"| {point['requested_target_aois']} | "
            f"{point['threshold']['mean']:.4f} | "
            f"{metrics['map_iou_delta']['mean']:+.6f} | "
            f"{metrics['false_edit_rate']['mean']:.6f} | "
            f"{metrics['missed_edit_rate']['mean']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def plot(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = result["aggregate"]
    x = list(range(len(points)))
    labels = [str(point["requested_target_aois"]) for point in points]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].plot(
        x,
        [point["metrics"]["map_iou_delta"]["mean"] for point in points],
        marker="o",
    )
    axes[0].set_ylabel("Map-IoU gain")
    axes[1].plot(
        x,
        [point["metrics"]["false_edit_rate"]["mean"] for point in points],
        marker="o",
        label="False edit",
    )
    axes[1].plot(
        x,
        [point["metrics"]["missed_edit_rate"]["mean"] for point in points],
        marker="s",
        label="Missed edit",
    )
    axes[1].set_ylabel("Error rate")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Target calibration AOIs")
        axis.grid(alpha=0.25)
    figure.suptitle(
        f"{result['source_backend']} gate -> {result['target_backend']}"
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
    parser.add_argument("--target-aoi-counts", default="0,1,2,4,8,16,all")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--selection-seed", type=int, default=20260727)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--output-figure", type=Path)
    args = parser.parse_args()
    result = run_curve(
        args.source_train,
        args.target_train,
        args.target_val,
        source_backend=args.source_backend,
        target_backend=args.target_backend,
        counts=parse_counts(args.target_aoi_counts),
        folds=args.folds,
        l2=args.l2,
        selection_seed=args.selection_seed,
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
