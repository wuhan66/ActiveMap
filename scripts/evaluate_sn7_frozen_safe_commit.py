#!/usr/bin/env python3
"""Apply a predeclared safe-commit calibration protocol to frozen SN7 test rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from activemap.frozen_test import assert_frozen_test_access
from scripts.calibrate_sn7_changemamba_safe_commit import (
    FEATURE_NAMES,
    choose_threshold,
    feature_matrix,
    fit_logistic,
    grouped_folds,
    paper_prediction,
    policy_metrics,
    predict_logistic,
    read_rows,
)


def fit_source_gate(
    rows: list[dict[str, Any]], *, folds: int, l2: float
) -> tuple[dict[str, np.ndarray], float, dict[str, Any]]:
    features = feature_matrix(rows)
    labels = np.asarray(
        [row["map_iou_delta"] > 1e-8 for row in rows], dtype=np.float64
    )
    fold_ids = grouped_folds(rows, folds)
    oof_scores = np.zeros(len(rows), dtype=np.float64)
    for fold in range(folds):
        held_out = fold_ids == fold
        fold_model = fit_logistic(
            features[~held_out], labels[~held_out], l2=l2
        )
        oof_scores[held_out] = predict_logistic(
            fold_model, features[held_out]
        )
    threshold, _ = choose_threshold(rows, oof_scores)
    model = fit_logistic(features, labels, l2=l2)
    metrics, _ = policy_metrics(rows, oof_scores, threshold=threshold)
    return model, threshold, metrics


def evaluate_frozen(
    source_rows: list[dict[str, Any]],
    target_train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    source_backend: str,
    target_backend: str,
    threshold_mode: str,
    folds: int,
    l2: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, source_threshold, source_oof_metrics = fit_source_gate(
        source_rows, folds=folds, l2=l2
    )
    target_train_scores = predict_logistic(
        model, feature_matrix(target_train_rows)
    )
    target_threshold, _ = choose_threshold(
        target_train_rows, target_train_scores
    )
    thresholds = {
        "source_oof": source_threshold,
        "all_target_train": target_threshold,
    }
    if threshold_mode not in thresholds:
        raise ValueError(f"unknown threshold mode: {threshold_mode}")
    threshold = thresholds[threshold_mode]

    test_scores = predict_logistic(model, feature_matrix(test_rows))
    always_metrics, _ = policy_metrics(test_rows, test_scores, threshold=0.0)
    safe_metrics, accepted = policy_metrics(
        test_rows, test_scores, threshold=threshold
    )
    output_rows = [
        {
            **row,
            "safe_commit_score": float(test_scores[index]),
            "safe_commit_threshold": threshold,
            "safe_commit_accepted": bool(accepted[index]),
            "safe_commit_predicted_edit": (
                row["predicted_edit"] if accepted[index] else "KEEP"
            ),
            "safe_commit_map_iou": (
                row["committed_map_iou"]
                if accepted[index]
                else row["prior_map_iou"]
            ),
        }
        for index, row in enumerate(test_rows)
    ]
    return (
        {
            "schema_version": "sn7-frozen-safe-commit-v1",
            "source_backend": source_backend,
            "target_backend": target_backend,
            "feature_names": FEATURE_NAMES,
            "protocol": {
                "gate_training": "source train only",
                "threshold_mode": threshold_mode,
                "threshold_selection_data": (
                    "source grouped OOF"
                    if threshold_mode == "source_oof"
                    else "all target train AOIs"
                ),
                "folds": folds,
                "l2": l2,
                "source_threshold": source_threshold,
                "target_train_threshold": target_threshold,
                "selected_threshold": threshold,
                "source_oof_metrics": source_oof_metrics,
            },
            "test": {
                "sample_count": len(test_rows),
                "aoi_count": len({str(row["aoi_id"]) for row in test_rows}),
                "always_commit": always_metrics,
                "safe_commit": safe_metrics,
                "safe_minus_always": {
                    key: safe_metrics[key] - always_metrics[key]
                    for key in always_metrics
                },
            },
            "test_assets_read": True,
        },
        output_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_train_predictions", type=Path)
    parser.add_argument("target_train_predictions", type=Path)
    parser.add_argument("target_test_predictions", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-backend", required=True)
    parser.add_argument("--target-backend", required=True)
    parser.add_argument(
        "--threshold-mode",
        choices=("source_oof", "all_target_train"),
        required=True,
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--l2", type=float, default=0.01)
    args = parser.parse_args()
    assert_frozen_test_access()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source_rows = read_rows(
        args.source_train_predictions, expected_split="train"
    )
    target_train_rows = read_rows(
        args.target_train_predictions, expected_split="train"
    )
    test_rows = read_rows(
        args.target_test_predictions, expected_split="test"
    )
    result, rows = evaluate_frozen(
        source_rows,
        target_train_rows,
        test_rows,
        source_backend=args.source_backend,
        target_backend=args.target_backend,
        threshold_mode=args.threshold_mode,
        folds=args.folds,
        l2=args.l2,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_sample.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with (args.output_dir / "predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(paper_prediction(row)) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
