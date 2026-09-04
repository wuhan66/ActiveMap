#!/usr/bin/env python3
"""Evaluate a safe-commit gate on a different frozen perception backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

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


def transfer(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    *,
    source_backend: str,
    target_backend: str,
    folds: int,
    l2: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_features = feature_matrix(source_rows)
    source_labels = np.asarray(
        [row["map_iou_delta"] > 1e-8 for row in source_rows], dtype=np.float64
    )
    fold_ids = grouped_folds(source_rows, folds)
    oof_scores = np.zeros(len(source_rows), dtype=np.float64)
    for fold in range(folds):
        held_out = fold_ids == fold
        model = fit_logistic(
            source_features[~held_out], source_labels[~held_out], l2=l2
        )
        oof_scores[held_out] = predict_logistic(model, source_features[held_out])
    threshold, candidates = choose_threshold(source_rows, oof_scores)
    source_baseline, _ = policy_metrics(source_rows, oof_scores, threshold=0.0)
    source_gated, _ = policy_metrics(
        source_rows, oof_scores, threshold=threshold
    )

    model = fit_logistic(source_features, source_labels, l2=l2)
    target_scores = predict_logistic(model, feature_matrix(target_rows))
    target_baseline, _ = policy_metrics(target_rows, target_scores, threshold=0.0)
    target_gated, accepted = policy_metrics(
        target_rows, target_scores, threshold=threshold
    )
    delta = {
        key: target_gated[key] - target_baseline[key]
        for key in target_baseline
    }
    output_rows = [
        {
            **row,
            "safe_commit_score": float(target_scores[index]),
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
        for index, row in enumerate(target_rows)
    ]
    return (
        {
            "schema_version": "sn7-safe-commit-cross-backend-transfer-v1",
            "feature_names": FEATURE_NAMES,
            "calibration": {
                "method": "source-backend AOI-grouped OOF balanced logistic regression",
                "source_backend": source_backend,
                "target_backend": target_backend,
                "folds": folds,
                "l2": l2,
                "beneficial_definition": "updater map_iou_delta > 1e-8",
                "threshold_selection": (
                    "maximize source OOF map_iou_delta subject to source "
                    "false_edit_rate <= always_commit"
                ),
                "selected_threshold": threshold,
                "train_sample_count": len(source_rows),
                "train_aoi_count": len(
                    {str(row["aoi_id"]) for row in source_rows}
                ),
                "oof_always_commit": source_baseline,
                "oof_safe_commit": source_gated,
                "threshold_candidates": candidates,
            },
            "validation": {
                "sample_count": len(target_rows),
                "aoi_count": len({str(row["aoi_id"]) for row in target_rows}),
                "always_commit": target_baseline,
                "safe_commit": target_gated,
                "safe_minus_always": delta,
            },
            "test_assets_read": False,
        },
        output_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_train_predictions", type=Path)
    parser.add_argument("target_val_predictions", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-backend", required=True)
    parser.add_argument("--target-backend", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--l2", type=float, default=0.01)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source_rows = read_rows(args.source_train_predictions, expected_split="train")
    target_rows = read_rows(args.target_val_predictions, expected_split="val")
    result, rows = transfer(
        source_rows,
        target_rows,
        source_backend=args.source_backend,
        target_backend=args.target_backend,
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
