#!/usr/bin/env python3
"""Calibrate an AOI-grouped safe-commit policy for ChangeMamba predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

EDIT_TYPES = ("KEEP", "ADD", "DELETE", "RESHAPE")
CONTINUOUS_FEATURES = (
    "confidence",
    "mean_change_probability",
    "max_change_probability",
    "p95_change_probability",
    "mean_predictive_entropy",
    "prior_foreground_fraction",
    "predicted_change_fraction",
)
FEATURE_NAMES = CONTINUOUS_FEATURES + tuple(
    f"predicted_edit_{edit}" for edit in EDIT_TYPES
)


def read_rows(path: Path, *, expected_split: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty prediction file: {path}")
    if {row["split"] for row in rows} != {expected_split}:
        raise ValueError(f"unexpected split in {path}")
    missing = [
        name
        for name in CONTINUOUS_FEATURES
        if any(name not in row for row in rows)
    ]
    if missing:
        raise ValueError(f"missing observable features {missing}: {path}")
    return rows


def feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    continuous = np.asarray(
        [[float(row[name]) for name in CONTINUOUS_FEATURES] for row in rows],
        dtype=np.float64,
    )
    categorical = np.asarray(
        [
            [float(row["predicted_edit"] == edit) for edit in EDIT_TYPES]
            for row in rows
        ],
        dtype=np.float64,
    )
    values = np.concatenate((continuous, categorical), axis=1)
    if not np.all(np.isfinite(values)):
        raise ValueError("safe-commit features must be finite")
    return values


def grouped_folds(rows: list[dict[str, Any]], folds: int) -> np.ndarray:
    counts = Counter(str(row["aoi_id"]) for row in rows)
    if folds < 2 or len(counts) < folds:
        raise ValueError("AOI-grouped calibration requires at least one AOI per fold")
    fold_sizes = [0] * folds
    assignments: dict[str, int] = {}
    ordered = sorted(
        counts,
        key=lambda value: (
            -counts[value],
            hashlib.sha256(value.encode("utf-8")).hexdigest(),
        ),
    )
    for aoi_id in ordered:
        fold = min(range(folds), key=lambda index: (fold_sizes[index], index))
        assignments[aoi_id] = fold
        fold_sizes[fold] += counts[aoi_id]
    return np.asarray([assignments[str(row["aoi_id"])] for row in rows])


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float,
    iterations: int = 50,
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.float64)
    if set(np.unique(labels)) != {0.0, 1.0}:
        raise ValueError("safe-commit fitting requires both target classes")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack(
        (np.ones(features.shape[0]), (features - mean) / scale)
    )
    positives = labels.sum()
    negatives = labels.size - positives
    sample_weight = np.where(
        labels > 0.5,
        labels.size / (2.0 * positives),
        labels.size / (2.0 * negatives),
    )
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * l2
    regularizer[0, 0] = 0.0
    for _ in range(iterations):
        logits = np.clip(design @ coefficients, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = (
            design.T @ (sample_weight * (probabilities - labels))
            / sample_weight.sum()
            + regularizer @ coefficients
        )
        curvature = sample_weight * probabilities * (1.0 - probabilities)
        hessian = (
            design.T @ (curvature[:, None] * design)
            / sample_weight.sum()
            + regularizer
        )
        step = np.linalg.solve(hessian + np.eye(hessian.shape[0]) * 1e-8, gradient)
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def predict_logistic(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    design = np.column_stack(
        (np.ones(features.shape[0]), (features - model["mean"]) / model["scale"])
    )
    logits = np.clip(design @ model["coefficients"], -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def serialize_logistic(model: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {
        name: np.asarray(model[name], dtype=np.float64).tolist()
        for name in ("mean", "scale", "coefficients")
    }


def deserialize_logistic(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    required = {"mean", "scale", "coefficients"}
    if set(payload) != required:
        raise ValueError("serialized safe-commit model has an invalid schema")
    model = {
        name: np.asarray(payload[name], dtype=np.float64) for name in required
    }
    feature_count = len(FEATURE_NAMES)
    if (
        model["mean"].shape != (feature_count,)
        or model["scale"].shape != (feature_count,)
        or model["coefficients"].shape != (feature_count + 1,)
        or not all(np.all(np.isfinite(value)) for value in model.values())
        or np.any(model["scale"] <= 0)
    ):
        raise ValueError("serialized safe-commit model has invalid parameters")
    return model


def policy_metrics(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(rows),):
        raise ValueError("safe-commit scores must align with prediction rows")
    candidate = np.asarray([row["predicted_edit"] != "KEEP" for row in rows])
    accepted = candidate & (scores >= threshold)
    predicted_edit = np.asarray(
        [
            row["predicted_edit"] if accepted[index] else "KEEP"
            for index, row in enumerate(rows)
        ]
    )
    target_edit = np.asarray([row["target_edit"] for row in rows])
    stable = target_edit == "KEEP"
    updates = ~stable
    prior_iou = np.asarray([row["prior_map_iou"] for row in rows], dtype=np.float64)
    updater_iou = np.asarray(
        [row["committed_map_iou"] for row in rows], dtype=np.float64
    )
    final_iou = np.where(accepted, updater_iou, prior_iou)
    beneficial = np.asarray(
        [row["map_iou_delta"] > 1e-8 for row in rows], dtype=bool
    )
    accepted_count = int(accepted.sum())
    return (
        {
            "committed_map_iou": float(final_iou.mean()),
            "map_iou_delta": float((final_iou - prior_iou).mean()),
            "operation_accuracy": float(np.mean(predicted_edit == target_edit)),
            "false_edit_rate": float(np.mean(predicted_edit[stable] != "KEEP")),
            "missed_edit_rate": float(np.mean(predicted_edit[updates] == "KEEP")),
            "wrong_edit_rate": float(
                np.mean(
                    (predicted_edit[updates] != "KEEP")
                    & (predicted_edit[updates] != target_edit[updates])
                )
            ),
            "candidate_rate": float(candidate.mean()),
            "accepted_commit_rate": float(accepted.mean()),
            "candidate_acceptance_rate": float(
                accepted_count / max(int(candidate.sum()), 1)
            ),
            "beneficial_commit_precision": float(
                beneficial[accepted].mean() if accepted_count else 0.0
            ),
            "beneficial_commit_recall": float(
                accepted[beneficial].mean() if beneficial.any() else 0.0
            ),
        },
        accepted,
    )


def choose_threshold(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    baseline, _ = policy_metrics(rows, scores, threshold=0.0)
    cap = baseline["false_edit_rate"] + 1e-12
    candidates: list[dict[str, Any]] = []
    for threshold in np.linspace(0.0, 1.0, 101):
        metrics, _ = policy_metrics(rows, scores, threshold=float(threshold))
        candidates.append({"threshold": float(threshold), "metrics": metrics})
    eligible = [
        candidate
        for candidate in candidates
        if candidate["metrics"]["false_edit_rate"] <= cap
    ]
    selected = max(
        eligible,
        key=lambda candidate: (
            candidate["metrics"]["map_iou_delta"],
            -candidate["metrics"]["false_edit_rate"],
            -candidate["metrics"]["missed_edit_rate"],
            candidate["metrics"]["candidate_acceptance_rate"],
        ),
    )
    return float(selected["threshold"]), candidates


def calibrate(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    *,
    folds: int,
    l2: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_features = feature_matrix(train_rows)
    train_labels = np.asarray(
        [row["map_iou_delta"] > 1e-8 for row in train_rows], dtype=np.float64
    )
    fold_ids = grouped_folds(train_rows, folds)
    oof_scores = np.zeros(len(train_rows), dtype=np.float64)
    for fold in range(folds):
        held_out = fold_ids == fold
        model = fit_logistic(
            train_features[~held_out],
            train_labels[~held_out],
            l2=l2,
        )
        oof_scores[held_out] = predict_logistic(model, train_features[held_out])
    threshold, candidates = choose_threshold(train_rows, oof_scores)
    oof_baseline, _ = policy_metrics(train_rows, oof_scores, threshold=0.0)
    oof_gated, _ = policy_metrics(train_rows, oof_scores, threshold=threshold)

    final_model = fit_logistic(train_features, train_labels, l2=l2)
    val_scores = predict_logistic(final_model, feature_matrix(val_rows))
    val_baseline, _ = policy_metrics(val_rows, val_scores, threshold=0.0)
    val_gated, val_accepted = policy_metrics(
        val_rows, val_scores, threshold=threshold
    )
    metric_delta = {
        key: val_gated[key] - val_baseline[key]
        for key in val_baseline
        if key in val_gated
    }
    output_rows = [
        {
            **row,
            "safe_commit_score": float(val_scores[index]),
            "safe_commit_threshold": threshold,
            "safe_commit_accepted": bool(val_accepted[index]),
            "safe_commit_predicted_edit": (
                row["predicted_edit"] if val_accepted[index] else "KEEP"
            ),
            "safe_commit_map_iou": (
                row["committed_map_iou"]
                if val_accepted[index]
                else row["prior_map_iou"]
            ),
        }
        for index, row in enumerate(val_rows)
    ]
    result = {
        "schema_version": "sn7-changemamba-safe-commit-v2",
        "feature_names": FEATURE_NAMES,
        "calibration": {
            "method": "AOI-grouped OOF balanced logistic regression",
            "folds": folds,
            "l2": l2,
            "beneficial_definition": "updater map_iou_delta > 1e-8",
            "threshold_selection": (
                "maximize OOF map_iou_delta subject to false_edit_rate "
                "<= always_commit"
            ),
            "selected_threshold": threshold,
            "frozen_logistic_model": serialize_logistic(final_model),
            "train_sample_count": len(train_rows),
            "train_aoi_count": len({row["aoi_id"] for row in train_rows}),
            "oof_always_commit": oof_baseline,
            "oof_safe_commit": oof_gated,
            "threshold_candidates": candidates,
        },
        "validation": {
            "sample_count": len(val_rows),
            "aoi_count": len({row["aoi_id"] for row in val_rows}),
            "always_commit": val_baseline,
            "safe_commit": val_gated,
            "safe_minus_always": metric_delta,
        },
        "test_assets_read": False,
    }
    return result, output_rows


def paper_prediction(row: dict[str, Any]) -> dict[str, Any]:
    accepted = bool(row["safe_commit_accepted"])
    commit_score = float(row["safe_commit_score"])
    return {
        "sample_id": row["sample_id"],
        "aoi_id": row["aoi_id"],
        "target_edit": row["target_edit"],
        "predicted_edit": row["predicted_edit"],
        "confidence": commit_score if accepted else 1.0 - commit_score,
        "committed": accepted,
        "raster_iou": row["safe_commit_map_iou"],
        "polygon_iou": None,
        "topology_valid": None,
        "metadata": {
            "split": row["split"],
            "safe_commit_threshold": row["safe_commit_threshold"],
            "safe_commit_score": commit_score,
            "updater_confidence": row["confidence"],
            "prior_map_iou": row["prior_map_iou"],
            "updater_map_iou": row["committed_map_iou"],
            "confidence_source": "train_oof_beneficial_commit_probability",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_predictions", type=Path)
    parser.add_argument("val_predictions", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--l2", type=float, default=0.01)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    train_rows = read_rows(args.train_predictions, expected_split="train")
    val_rows = read_rows(args.val_predictions, expected_split="val")
    result, output_rows = calibrate(
        train_rows,
        val_rows,
        folds=args.folds,
        l2=args.l2,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "per_sample.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")
    with (args.output_dir / "predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in output_rows:
            handle.write(json.dumps(paper_prediction(row)) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
