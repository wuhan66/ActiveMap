#!/usr/bin/env python3
"""Fit a conservative risk controller from aligned train-only OOF action scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scripts.train_sequential_set_utility_ranker import action_metrics, candidate_thresholds, task_bootstrap
except ModuleNotFoundError as error:
    if error.name is None or not error.name.startswith("scripts"):
        raise
    from train_sequential_set_utility_ranker import action_metrics, candidate_thresholds, task_bootstrap


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(run: Path) -> dict[str, Any]:
    payload = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if payload.get("test_assets_read") is not False:
        raise ValueError("source run violates frozen-test isolation")
    return payload


def _rows(path: Path) -> list[dict[str, Any]]:
    result = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not result:
        raise ValueError(f"empty score file: {path}")
    return result


def build_examples(primary_rows: list[dict[str, Any]], safety_rows: list[dict[str, Any]], primary_margin: float, safety_margin: float) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if len(primary_rows) != len(safety_rows):
        raise ValueError("primary and safety rows have different support")
    features, labels, examples = [], [], []
    for primary, safety in zip(primary_rows, safety_rows, strict=True):
        if (
            str(primary.get("task_id")) != str(safety.get("task_id"))
            or list(primary.get("candidate_ids", [])) != list(safety.get("candidate_ids", []))
            or list(primary.get("advantages", [])) != list(safety.get("advantages", []))
        ):
            raise ValueError("source rows are not aligned")
        candidates = [str(value) for value in primary["candidate_ids"]]
        advantages = np.asarray(primary["advantages"], dtype=np.float64)
        primary_scores = np.asarray(primary["relative_action_scores"], dtype=np.float64)
        safety_scores = np.asarray(safety["relative_action_scores"], dtype=np.float64)
        if len(candidates) < 2 or primary_scores.shape != advantages.shape or safety_scores.shape != advantages.shape:
            raise ValueError("invalid candidate score dimensions")
        primary_index = int(np.argmax(primary_scores))
        safety_index = int(np.argmax(safety_scores))
        primary_order = np.sort(primary_scores)[::-1]
        safety_order = np.sort(safety_scores)[::-1]
        primary_call = primary_scores[primary_index] >= primary_margin
        safety_call = safety_scores[safety_index] >= safety_margin
        features.append(
            [
                float(primary_scores[primary_index]),
                float(safety_scores[safety_index]),
                float(primary_scores[primary_index] - primary_order[1]),
                float(safety_scores[safety_index] - safety_order[1]),
                float(primary_scores[primary_index] - primary_margin),
                float(safety_scores[safety_index] - safety_margin),
                float(primary_index == safety_index),
                float(primary_call),
                float(safety_call),
            ]
        )
        labels.append(int(advantages[primary_index] > 0.0))
        examples.append(
            {
                "task_id": str(primary["task_id"]),
                "candidate_ids": candidates,
                "advantages": advantages,
                "primary_index": primary_index,
                "primary_evidence_id": candidates[primary_index],
                "primary_selected_utility": float(advantages[primary_index]),
            }
        )
    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if not np.all(np.isfinite(values)) or len(np.unique(target)) != 2:
        raise ValueError("risk controller needs finite features and both safety labels")
    return values, target, examples


def _model() -> Any:
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, random_state=0))


def cross_fit_probabilities(features: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    folds = min(5, int(labels.sum()), int((1 - labels).sum()))
    if folds < 2:
        raise ValueError("insufficient task labels for gate cross-fit")
    result = np.full(len(labels), np.nan, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_indices, holdout_indices in splitter.split(features, labels):
        classifier = _model().fit(features[train_indices], labels[train_indices])
        result[holdout_indices] = classifier.predict_proba(features[holdout_indices])[:, 1]
    if not np.all(np.isfinite(result)):
        raise RuntimeError("cross-fit left gate scores unfilled")
    return result


def traces(examples: list[dict[str, Any]], probabilities: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    if len(examples) != len(probabilities):
        raise ValueError("example/probability count differs")
    result = []
    for example, probability in zip(examples, probabilities, strict=True):
        advantages = example["advantages"]
        target_index = int(np.argmax(advantages))
        target_call = advantages[target_index] > 0.0
        acquire = float(probability) >= threshold
        index = int(example["primary_index"])
        result.append(
            {
                "task_id": example["task_id"],
                "candidate_count": len(advantages),
                "target_selection": "ACQUIRE" if target_call else "STOP",
                "target_evidence_id": example["candidate_ids"][target_index] if target_call else None,
                "predicted_selection": "ACQUIRE" if acquire else "STOP",
                "predicted_evidence_id": example["primary_evidence_id"] if acquire else None,
                "realized_utility": float(example["primary_selected_utility"]) if acquire else 0.0,
                "oracle_utility": max(float(np.max(advantages)), 0.0),
                "risk_positive_probability": float(probability),
                "test_assets_read": False,
            }
        )
    return result


def select_threshold(examples: list[dict[str, Any]], probabilities: np.ndarray, maximum_false_call_rate: float) -> tuple[float, dict[str, float]]:
    thresholds = candidate_thresholds(
        [np.asarray([value]) for value in probabilities],
        (0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50),
    )
    candidates = [(threshold, action_metrics(traces(examples, probabilities, threshold))) for threshold in thresholds]
    safe = [entry for entry in candidates if entry[1]["false_call_rate"] <= maximum_false_call_rate and entry[1]["call_rate"] > 0.0]
    if not safe:
        raise RuntimeError("no OOF threshold meets the false-call constraint")
    return max(safe, key=lambda entry: (entry[1]["utility_mean"], entry[1]["exact_candidate_recall"], -entry[1]["false_call_rate"], -entry[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("primary_run", type=Path)
    parser.add_argument("safety_run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--maximum-false-call-rate", type=float, default=0.05)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not 0.0 < args.maximum_false_call_rate <= 0.10:
        raise ValueError("maximum false-call rate must be in (0, 0.10]")
    primary_summary = _summary(args.primary_run)
    safety_summary = _summary(args.safety_run)
    primary_margin = float(primary_summary["oof"]["safety_margin"])
    safety_margin = float(safety_summary["oof"]["safety_margin"])
    train_x, train_y, train_examples = build_examples(
        _rows(args.primary_run / "oof_raw_scores.jsonl"),
        _rows(args.safety_run / "oof_raw_scores.jsonl"),
        primary_margin,
        safety_margin,
    )
    val_x, _, val_examples = build_examples(
        _rows(args.primary_run / "validation_raw_scores.jsonl"),
        _rows(args.safety_run / "validation_raw_scores.jsonl"),
        primary_margin,
        safety_margin,
    )
    oof_probabilities = cross_fit_probabilities(train_x, train_y, args.seed)
    threshold, oof_metrics = select_threshold(
        train_examples, oof_probabilities, args.maximum_false_call_rate
    )
    classifier = _model().fit(train_x, train_y)
    val_probabilities = classifier.predict_proba(val_x)[:, 1]
    oof_traces = traces(train_examples, oof_probabilities, threshold)
    validation_traces = traces(val_examples, val_probabilities, threshold)
    validation_metrics = action_metrics(validation_traces)
    bootstrap = task_bootstrap(validation_traces, repetitions=args.bootstrap_repetitions, seed=args.seed)
    promotion = {
        "nonzero_calls": validation_metrics["call_rate"] > 0.0,
        "bounded_call_rate": validation_metrics["call_rate"] <= 0.50,
        "positive_utility": validation_metrics["utility_mean"] > 0.0,
        "task_utility_ci_above_zero": bootstrap["utility_mean"]["ci95_low"] > 0.0,
        "false_call_rate_at_most_0_10": validation_metrics["false_call_rate"] <= 0.10,
        "exact_candidate_recall_above_zero": validation_metrics["exact_candidate_recall"] > 0.0,
    }
    args.output.mkdir(parents=True)
    with (args.output / "oof_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in oof_traces:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (args.output / "validation_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in validation_traces:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "structured-vla-oof-risk-controller-v1",
        "role": "diagnostic-only-aligned-train-only-oof-risk-controller",
        "controller_interface": "STOP-or-primary-public-evidence-id",
        "primary_candidate": "full visual+structured SET_SELECT head argmax",
        "safety_features": [
            "primary_best_score", "safety_best_score", "primary_top1_margin",
            "safety_top1_margin", "primary_margin_distance", "safety_margin_distance",
            "same_candidate", "primary_calls", "safety_calls",
        ],
        "risk_target": "primary selected candidate has positive realized utility",
        "gate_protocol": "second-level stratified task cross-fit on train-only base OOF scores",
        "maximum_oof_false_call_rate": args.maximum_false_call_rate,
        "threshold": threshold,
        "oof": {"metrics": oof_metrics},
        "validation": {"metrics": validation_metrics},
        "validation_task_bootstrap": bootstrap,
        "promotion_gate": {**promotion, "passed": all(promotion.values())},
        "sources": {
            "primary_summary": _sha256(args.primary_run / "summary.json"),
            "primary_oof_scores": _sha256(args.primary_run / "oof_raw_scores.jsonl"),
            "primary_validation_scores": _sha256(args.primary_run / "validation_raw_scores.jsonl"),
            "safety_summary": _sha256(args.safety_run / "summary.json"),
            "safety_oof_scores": _sha256(args.safety_run / "oof_raw_scores.jsonl"),
            "safety_validation_scores": _sha256(args.safety_run / "validation_raw_scores.jsonl"),
        },
        "test_assets_read": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
