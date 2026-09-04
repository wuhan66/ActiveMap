#!/usr/bin/env python3
"""Add a train-only task-level STOP risk gate to a frozen SET_SELECT ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scripts.train_sequential_set_utility_ranker import (
        _audit,
        _fit_predict,
        _model,
        action_metrics,
        candidate_thresholds,
        grouped_oof_scores,
        load_task_examples,
        task_bootstrap,
    )
except ModuleNotFoundError as error:
    if error.name is None or not error.name.startswith("scripts"):
        raise
    from train_sequential_set_utility_ranker import (
        _audit,
        _fit_predict,
        _model,
        action_metrics,
        candidate_thresholds,
        grouped_oof_scores,
        load_task_examples,
        task_bootstrap,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_score_features(scores: list[np.ndarray]) -> np.ndarray:
    """Pool only public candidate scores into task-level STOP-risk features."""
    rows = []
    for score in scores:
        values = np.sort(np.asarray(score, dtype=np.float64))[::-1]
        if len(values) < 2 or not np.all(np.isfinite(values)):
            raise ValueError("each task needs at least two finite candidate scores")
        rows.append(
            [
                float(values[0]),
                float(values.mean()),
                float(values.std()),
                float(values[0] - values[1]),
                float(values[-1]),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _gate_model(c_value: float) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, max_iter=5000, random_state=0),
    )


def oof_gate_probabilities(
    features: np.ndarray, labels: np.ndarray, *, c_value: float, seed: int
) -> np.ndarray:
    if len(features) != len(labels) or len(np.unique(labels)) != 2:
        raise ValueError("gate requires aligned positive and negative task labels")
    folds = min(5, int(labels.sum()), int((1 - labels).sum()))
    if folds < 2:
        raise ValueError("insufficient task labels for gate OOF")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    for train_indices, holdout_indices in splitter.split(features, labels):
        model = _gate_model(c_value).fit(features[train_indices], labels[train_indices])
        probabilities[holdout_indices] = model.predict_proba(features[holdout_indices])[:, 1]
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("gate OOF left tasks without a probability")
    return probabilities


def gated_action_traces(
    tasks: list[dict[str, Any]],
    candidate_scores: list[np.ndarray],
    gate_probabilities: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    if len(tasks) != len(candidate_scores) or len(tasks) != len(gate_probabilities):
        raise ValueError("task, candidate score, and gate probability counts differ")
    traces = []
    for task, score, gate_probability in zip(
        tasks, candidate_scores, gate_probabilities, strict=True
    ):
        advantages = np.asarray(task["advantages"], dtype=np.float64)
        score = np.asarray(score, dtype=np.float64)
        if advantages.shape != score.shape:
            raise ValueError("candidate score shape differs from utility shape")
        target_index = int(np.argmax(advantages))
        target_call = bool(advantages[target_index] > 0.0)
        predicted_index = int(np.argmax(score))
        predicted_call = bool(gate_probability >= threshold)
        realized = float(advantages[predicted_index]) if predicted_call else 0.0
        traces.append(
            {
                "task_id": str(task["task_id"]),
                "candidate_count": len(advantages),
                "target_selection": "ACQUIRE" if target_call else "STOP",
                "target_evidence_id": (
                    str(task["candidate_ids"][target_index]) if target_call else None
                ),
                "predicted_selection": "ACQUIRE" if predicted_call else "STOP",
                "predicted_evidence_id": (
                    str(task["candidate_ids"][predicted_index]) if predicted_call else None
                ),
                "realized_utility": realized,
                "oracle_utility": float(max(float(np.max(advantages)), 0.0)),
                "predicted_candidate_score": float(score[predicted_index]),
                "predicted_acquire_probability": float(gate_probability),
                "test_assets_read": False,
            }
        )
    return traces


def select_gate_configuration(
    tasks: list[dict[str, Any]],
    candidate_scores: list[np.ndarray],
    *,
    c_values: tuple[float, ...],
    call_rates: tuple[float, ...],
    max_call_rate: float,
    min_recall: float,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    labels = np.asarray([np.max(task["advantages"]) > 0.0 for task in tasks], dtype=np.int64)
    features = task_score_features(candidate_scores)
    grid = []
    probabilities_by_c = {}
    for c_value in c_values:
        probabilities = oof_gate_probabilities(features, labels, c_value=c_value, seed=seed)
        probabilities_by_c[c_value] = probabilities
        for threshold in candidate_thresholds(
            [np.asarray([value]) for value in probabilities], call_rates
        ):
            metrics = action_metrics(
                gated_action_traces(tasks, candidate_scores, probabilities, threshold)
            )
            grid.append({"c_value": c_value, "threshold": threshold, "metrics": metrics})
    feasible = [
        row
        for row in grid
        if 0.0 < row["metrics"]["call_rate"] <= max_call_rate
        and row["metrics"]["acquire_recall"] >= min_recall
    ]
    if not feasible:
        raise RuntimeError("no OOF risk gate satisfies recall and call-rate constraints")
    selected = max(
        feasible,
        key=lambda row: (
            row["metrics"]["utility_mean"],
            -row["metrics"]["false_call_rate"],
            row["metrics"]["exact_candidate_recall"],
            row["metrics"]["macro_f1"],
            -row["metrics"]["call_rate"],
            -row["c_value"],
        ),
    )
    return selected, grid, probabilities_by_c[float(selected["c_value"])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_features", type=Path)
    parser.add_argument("val_features", type=Path)
    parser.add_argument("train_manifest", type=Path)
    parser.add_argument("val_manifest", type=Path)
    parser.add_argument("train_context_audit", type=Path)
    parser.add_argument("val_context_audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ranker-alpha", type=float, default=1000.0)
    parser.add_argument("--gate-c-values", default="0.01,0.1,1,10")
    parser.add_argument(
        "--candidate-call-rates",
        default="0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.30,0.40,0.50",
    )
    parser.add_argument("--max-call-rate", type=float, default=0.50)
    parser.add_argument("--min-recall", type=float, default=0.10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    train_audit = _audit(args.train_context_audit, "train")
    val_audit = _audit(args.val_context_audit, "val")
    train_tasks, train_summary = load_task_examples(
        args.train_features, args.train_manifest, "train"
    )
    val_tasks, val_summary = load_task_examples(args.val_features, args.val_manifest, "val")
    if train_summary.get("pooling") != val_summary.get("pooling"):
        raise ValueError("train and validation feature pooling differs")

    train_candidate_scores = grouped_oof_scores(train_tasks, args.ranker_alpha)
    selected, grid, train_gate_probabilities = select_gate_configuration(
        train_tasks,
        train_candidate_scores,
        c_values=tuple(float(value) for value in args.gate_c_values.split(",")),
        call_rates=tuple(float(value) for value in args.candidate_call_rates.split(",")),
        max_call_rate=args.max_call_rate,
        min_recall=args.min_recall,
        seed=args.seed,
    )
    labels = np.asarray(
        [np.max(task["advantages"]) > 0.0 for task in train_tasks], dtype=np.int64
    )
    gate_features = task_score_features(train_candidate_scores)
    gate = _gate_model(float(selected["c_value"])).fit(gate_features, labels)
    candidate_ranker = _model(args.ranker_alpha).fit(
        np.concatenate([task["features"] for task in train_tasks], axis=0),
        np.concatenate([task["advantages"] for task in train_tasks], axis=0),
    )
    val_candidate_scores = _fit_predict(train_tasks, val_tasks, args.ranker_alpha)
    val_gate_probabilities = gate.predict_proba(task_score_features(val_candidate_scores))[:, 1]
    traces = gated_action_traces(
        val_tasks,
        val_candidate_scores,
        val_gate_probabilities,
        float(selected["threshold"]),
    )
    metrics = action_metrics(traces)
    bootstrap = task_bootstrap(
        traces, repetitions=args.bootstrap_repetitions, seed=args.seed
    )
    promotion = {
        "nonzero_calls": metrics["call_rate"] > 0.0,
        "bounded_call_rate": metrics["call_rate"] <= args.max_call_rate,
        "positive_utility": metrics["utility_mean"] > 0.0,
        "task_utility_ci_above_zero": bootstrap["utility_mean"]["ci95_low"] > 0.0,
        "false_call_rate_at_most_0_10": metrics["false_call_rate"] <= 0.10,
        "exact_candidate_recall_above_zero": metrics["exact_candidate_recall"] > 0.0,
    }
    args.output.mkdir(parents=True)
    dump(candidate_ranker, args.output / "candidate_ranker.joblib")
    dump(gate, args.output / "risk_gate.joblib")
    (args.output / "selection_grid.json").write_text(
        json.dumps(grid, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "validation_traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "sequential-set-risk-gate-v1",
        "role": "diagnostic-only-explicit-multicandidate-SET_SELECT",
        "selection_protocol": "candidate-ranker-OOF-then-task-risk-gate-OOF-train-only",
        "controller_interface": "explicit-multicandidate-SET_SELECT",
        "candidate_feature_pooling": train_summary.get("pooling"),
        "ranker_alpha": args.ranker_alpha,
        "risk_gate_features": ["max", "mean", "std", "top1_minus_top2", "min"],
        "train_task_count": len(train_tasks),
        "validation_task_count": len(val_tasks),
        "selected": selected,
        "validation": metrics,
        "validation_task_bootstrap": bootstrap,
        "promotion_gate": {**promotion, "passed": all(promotion.values())},
        "context_audits": {
            "train": train_audit["explicit_multicandidate_reformulation_task_rate"],
            "val": val_audit["explicit_multicandidate_reformulation_task_rate"],
        },
        "sources": {
            "train_features": _sha256(args.train_features / "summary.json"),
            "val_features": _sha256(args.val_features / "summary.json"),
            "train_manifest": _sha256(args.train_manifest),
            "val_manifest": _sha256(args.val_manifest),
            "train_context_audit": _sha256(args.train_context_audit / "summary.json"),
            "val_context_audit": _sha256(args.val_context_audit / "summary.json"),
        },
        "test_assets_read": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
