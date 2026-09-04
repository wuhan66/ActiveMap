#!/usr/bin/env python3
"""Diagnose an explicit multi-candidate SELECT interface on frozen VLM states.

The preceding single-candidate controller cannot score sibling candidates
jointly because they are absent from its prompt.  This script evaluates a new,
explicit SET_SELECT interface: a shared candidate scorer receives frozen
candidate-wise visual states and makes one STOP-or-candidate action per task.
It is a diagnostic-only bridge to a structured VLA action head, never a
promotion result by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from joblib import dump
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model(alpha: float) -> Any:
    return make_pipeline(
        StandardScaler(), Ridge(alpha=alpha, solver="lsqr", tol=1e-3, max_iter=2000)
    )


def _observation(row: dict[str, Any]) -> dict[str, Any]:
    content = row["messages"][1]["content"]
    texts = [part["text"] for part in content if part.get("type") == "text"]
    if len(texts) != 1:
        raise ValueError("SELECT row must expose exactly one state payload")
    state = json.loads(str(texts[0]))
    if state.get("controller_stage") != "SELECT":
        raise ValueError("manifest contains a non-SELECT prompt")
    if not isinstance(state.get("evidence_id"), str):
        raise ValueError("single-candidate SELECT prompt lacks evidence_id")
    return state


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def load_task_examples(
    feature_root: Path, manifest: Path, expected_split: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Align frozen per-candidate embeddings with a task-level action set."""
    feature_summary = json.loads((feature_root / "summary.json").read_text(encoding="utf-8"))
    if feature_summary.get("test_assets_read") is not False:
        raise ValueError("feature receipt violates frozen-test isolation")
    if feature_summary.get("controller_stage") != "SELECT":
        raise ValueError("set utility ranker requires frozen SELECT features")
    features = np.load(feature_root / "features.npy").astype(np.float32)
    records = _load_jsonl(feature_root / "records.jsonl")
    if len(features) != len(records):
        raise ValueError("feature and record counts differ")
    if {str(row.get("split")) for row in records} != {expected_split}:
        raise ValueError("feature records do not match requested split")

    manifest_rows = _load_jsonl(manifest)
    if {str(row.get("split")) for row in manifest_rows} != {expected_split}:
        raise ValueError("manifest does not match requested split")
    by_trajectory = {str(row["trajectory_id"]): row for row in manifest_rows}
    if len(by_trajectory) != len(manifest_rows):
        raise ValueError("manifest contains duplicate trajectories")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature, record in zip(features, records, strict=True):
        trajectory_id = str(record["example_id"])
        source = by_trajectory.get(trajectory_id)
        if source is None:
            raise ValueError(f"feature record lacks source trajectory: {trajectory_id}")
        if str(source["task_id"]) != str(record["task_id"]):
            raise ValueError("feature and manifest task IDs differ")
        state = _observation(source)
        grouped[str(record["task_id"])].append(
            {
                "trajectory_id": trajectory_id,
                "evidence_id": str(state["evidence_id"]),
                "feature": feature,
                "advantage": float(record["consensus_mean_utility_gain"]),
            }
        )
    if set(by_trajectory) != {str(row["example_id"]) for row in records}:
        raise ValueError("manifest and feature receipt do not have identical SELECT support")

    tasks = []
    for task_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["trajectory_id"])
        candidate_ids = [str(row["evidence_id"]) for row in rows]
        if len(candidate_ids) < 2 or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"task {task_id} lacks a unique explicit candidate set")
        tasks.append(
            {
                "task_id": task_id,
                "candidate_ids": candidate_ids,
                "features": np.stack([row["feature"] for row in rows]),
                "advantages": np.asarray([row["advantage"] for row in rows], dtype=np.float64),
            }
        )
    if not tasks:
        raise ValueError("no task groups were assembled")
    return tasks, feature_summary


def _fit_predict(
    train_tasks: list[dict[str, Any]], test_tasks: list[dict[str, Any]], alpha: float
) -> list[np.ndarray]:
    features = np.concatenate([task["features"] for task in train_tasks], axis=0)
    targets = np.concatenate([task["advantages"] for task in train_tasks], axis=0)
    model = _model(alpha).fit(features, targets)
    return [model.predict(task["features"]) for task in test_tasks]


def grouped_oof_scores(tasks: list[dict[str, Any]], alpha: float) -> list[np.ndarray]:
    labels = np.asarray([np.max(task["advantages"]) > 0.0 for task in tasks], dtype=np.int64)
    groups = np.asarray([task["task_id"] for task in tasks])
    if len(tasks) < 4 or len(np.unique(labels)) != 2:
        raise ValueError("insufficient positive and negative task groups for OOF selection")
    splitter = GroupKFold(n_splits=min(5, len(tasks)))
    scores: list[np.ndarray | None] = [None] * len(tasks)
    for train_indices, holdout_indices in splitter.split(np.zeros(len(tasks)), labels, groups):
        predictions = _fit_predict(
            [tasks[index] for index in train_indices],
            [tasks[index] for index in holdout_indices],
            alpha,
        )
        for index, prediction in zip(holdout_indices, predictions, strict=True):
            scores[int(index)] = prediction
    if any(score is None for score in scores):
        raise RuntimeError("grouped OOF left a task without a score")
    return [np.asarray(score) for score in scores if score is not None]


def candidate_thresholds(
    scores: list[np.ndarray], call_rates: tuple[float, ...]
) -> tuple[float, ...]:
    maxima = np.asarray([float(np.max(score)) for score in scores], dtype=np.float64)
    if not np.all(np.isfinite(maxima)):
        raise ValueError("candidate scores must be finite")
    if not call_rates or any(not 0.0 < rate <= 1.0 for rate in call_rates):
        raise ValueError("candidate call rates must be in (0, 1]")
    return tuple(
        sorted(
            {
                float(np.quantile(maxima, 1.0 - rate, method="higher"))
                for rate in call_rates
            },
            reverse=True,
        )
    )


def action_traces(
    tasks: list[dict[str, Any]], scores: list[np.ndarray], threshold: float
) -> list[dict[str, Any]]:
    if len(tasks) != len(scores):
        raise ValueError("task and score counts differ")
    traces = []
    for task, score in zip(tasks, scores, strict=True):
        advantages = np.asarray(task["advantages"], dtype=np.float64)
        score = np.asarray(score, dtype=np.float64)
        if score.shape != advantages.shape:
            raise ValueError("candidate score shape differs from utility shape")
        target_index = int(np.argmax(advantages))
        target_call = bool(advantages[target_index] > 0.0)
        predicted_index = int(np.argmax(score))
        predicted_call = bool(score[predicted_index] >= threshold)
        selected_index = predicted_index if predicted_call else None
        realized = float(advantages[selected_index]) if selected_index is not None else 0.0
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
                "predicted_score": float(score[predicted_index]),
                "test_assets_read": False,
            }
        )
    return traces


def action_metrics(traces: list[dict[str, Any]]) -> dict[str, float]:
    if not traces:
        raise ValueError("task action traces are empty")
    target = [row["target_selection"] == "ACQUIRE" for row in traces]
    predicted = [row["predicted_selection"] == "ACQUIRE" for row in traces]
    tp = sum(left and right for left, right in zip(target, predicted, strict=True))
    fp = sum(not left and right for left, right in zip(target, predicted, strict=True))
    fn = sum(left and not right for left, right in zip(target, predicted, strict=True))
    tn = len(traces) - tp - fp - fn
    acquire_precision = tp / max(tp + fp, 1)
    acquire_recall = tp / max(tp + fn, 1)
    stop_precision = tn / max(tn + fn, 1)
    stop_recall = tn / max(tn + fp, 1)
    acquire_f1 = 2.0 * acquire_precision * acquire_recall / max(
        acquire_precision + acquire_recall, 1e-12
    )
    stop_f1 = 2.0 * stop_precision * stop_recall / max(stop_precision + stop_recall, 1e-12)
    target_calls = sum(target)
    predicted_calls = sum(predicted)
    exact = sum(
        row["target_selection"] == "ACQUIRE"
        and row["predicted_selection"] == "ACQUIRE"
        and row["target_evidence_id"] == row["predicted_evidence_id"]
        for row in traces
    )
    utilities = [float(row["realized_utility"]) for row in traces]
    oracle = [float(row["oracle_utility"]) for row in traces]
    always_stop_macro_f1 = (2.0 * (1.0 - target_calls / len(traces))) / (
        2.0 - target_calls / len(traces)
    ) / 2.0
    return {
        "task_count": float(len(traces)),
        "utility_sum": float(sum(utilities)),
        "utility_mean": float(sum(utilities) / len(traces)),
        "oracle_utility_mean": float(sum(oracle) / len(traces)),
        "mean_regret": float((sum(oracle) - sum(utilities)) / len(traces)),
        "selection_accuracy": float((tp + tn) / len(traces)),
        "macro_f1": float((acquire_f1 + stop_f1) / 2.0),
        "macro_f1_delta_vs_always_stop": float(
            (acquire_f1 + stop_f1) / 2.0 - always_stop_macro_f1
        ),
        "acquire_precision": float(acquire_precision),
        "acquire_recall": float(acquire_recall),
        "acquire_f1": float(acquire_f1),
        "stop_f1": float(stop_f1),
        "call_rate": float(predicted_calls / len(traces)),
        "false_call_rate": float(fp / max(len(traces) - target_calls, 1)),
        "missed_call_rate": float(fn / max(target_calls, 1)),
        "exact_candidate_recall": float(exact / max(target_calls, 1)),
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def task_bootstrap(
    traces: list[dict[str, Any]], *, repetitions: int, seed: int
) -> dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    rng = random.Random(seed)
    values = []
    for _ in range(repetitions):
        sampled = rng.choices(traces, k=len(traces))
        values.append(action_metrics(sampled)["utility_mean"])
    observed = action_metrics(traces)["utility_mean"]
    return {
        "grouping_unit": "task_id",
        "task_count": len(traces),
        "repetitions": repetitions,
        "seed": seed,
        "utility_mean": {
            "observed": observed,
            "ci95_low": _quantile(values, 0.025),
            "ci95_high": _quantile(values, 0.975),
            "bootstrap_probability_gt_zero": sum(value > 0.0 for value in values)
            / repetitions,
        },
    }


def choose_oof_configuration(
    tasks: list[dict[str, Any]],
    *,
    alphas: tuple[float, ...],
    call_rates: tuple[float, ...],
    max_call_rate: float,
    min_recall: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grid = []
    for alpha in alphas:
        scores = grouped_oof_scores(tasks, alpha)
        for threshold in candidate_thresholds(scores, call_rates):
            metrics = action_metrics(action_traces(tasks, scores, threshold))
            grid.append({"alpha": alpha, "threshold": threshold, "metrics": metrics})
    feasible = [
        row
        for row in grid
        if 0.0 < row["metrics"]["call_rate"] <= max_call_rate
        and row["metrics"]["acquire_recall"] >= min_recall
    ]
    if not feasible:
        raise RuntimeError("no OOF set policy satisfies recall and call-rate constraints")
    selected = max(
        feasible,
        key=lambda row: (
            row["metrics"]["utility_mean"],
            -row["metrics"]["false_call_rate"],
            row["metrics"]["exact_candidate_recall"],
            row["metrics"]["macro_f1"],
            -row["metrics"]["call_rate"],
            -row["alpha"],
        ),
    )
    return selected, grid


def _audit(path: Path, expected_split: str) -> dict[str, Any]:
    report = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if report.get("split") != expected_split or report.get("test_assets_read") is not False:
        raise ValueError("invalid public-context audit receipt")
    if not bool(report.get("explicit_multicandidate_reformulation_task_rate") == 1.0):
        raise ValueError("all tasks must support an explicit multicandidate reformulation")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_features", type=Path)
    parser.add_argument("val_features", type=Path)
    parser.add_argument("train_manifest", type=Path)
    parser.add_argument("val_manifest", type=Path)
    parser.add_argument("train_context_audit", type=Path)
    parser.add_argument("val_context_audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000")
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
    if train_tasks[0]["features"].shape[1] != val_tasks[0]["features"].shape[1]:
        raise ValueError("train and validation feature dimensions differ")
    if train_summary.get("pooling") != val_summary.get("pooling"):
        raise ValueError("train and validation feature pooling differs")

    selected, grid = choose_oof_configuration(
        train_tasks,
        alphas=tuple(float(value) for value in args.alphas.split(",")),
        call_rates=tuple(float(value) for value in args.candidate_call_rates.split(",")),
        max_call_rate=args.max_call_rate,
        min_recall=args.min_recall,
    )
    alpha = float(selected["alpha"])
    threshold = float(selected["threshold"])
    model = _model(alpha).fit(
        np.concatenate([task["features"] for task in train_tasks], axis=0),
        np.concatenate([task["advantages"] for task in train_tasks], axis=0),
    )
    val_scores = [model.predict(task["features"]) for task in val_tasks]
    traces = action_traces(val_tasks, val_scores, threshold)
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
    dump(model, args.output / "ranker.joblib")
    (args.output / "selection_grid.json").write_text(
        json.dumps(grid, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "validation_traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "sequential-set-utility-ranker-v1",
        "role": "diagnostic-only-explicit-multicandidate-SET_SELECT",
        "selection_protocol": "task-grouped-OOF-train-only",
        "feature_protocol": "frozen-Qwen3-VL-SELECT-candidate-wise-embeddings",
        "controller_interface": "explicit-multicandidate-SET_SELECT",
        "candidate_action": "STOP-or-one-public-evidence_id",
        "candidate_feature_pooling": train_summary.get("pooling"),
        "train_task_count": len(train_tasks),
        "validation_task_count": len(val_tasks),
        "feature_dim": int(train_tasks[0]["features"].shape[1]),
        "selected": selected,
        "validation": metrics,
        "validation_task_bootstrap": bootstrap,
        "promotion_gate": {**promotion, "passed": all(promotion.values())},
        "context_audits": {
            "train": {
                "current_prompt_listwise_eligible": train_audit[
                    "current_prompt_listwise_eligible"
                ],
                "explicit_multicandidate_reformulation_task_rate": train_audit[
                    "explicit_multicandidate_reformulation_task_rate"
                ],
            },
            "val": {
                "current_prompt_listwise_eligible": val_audit[
                    "current_prompt_listwise_eligible"
                ],
                "explicit_multicandidate_reformulation_task_rate": val_audit[
                    "explicit_multicandidate_reformulation_task_rate"
                ],
            },
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
