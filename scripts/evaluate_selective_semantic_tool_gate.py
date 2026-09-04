#!/usr/bin/env python3
"""Evaluate train-selected sparse use of prior-conditioned semantic evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from activemap.agent.tool_belief_data import PostAcquisitionToolPairExample
from activemap.agent.tool_features import encode_tool_result
from activemap.agent.tool_sft import terminal_reward
from activemap.evaluation_controls import grouped_derangement, permutation_sha256
from activemap.models import EditOperation

EDIT_ORDER = list(EditOperation)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> list[PostAcquisitionToolPairExample]:
    rows = [
        PostAcquisitionToolPairExample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def _probabilities(row: PostAcquisitionToolPairExample) -> tuple[np.ndarray, np.ndarray]:
    if row.semantic_result is None:
        raise ValueError(f"missing semantic result: {row.example_id}")
    belief = np.asarray(row.post_acquisition_belief.edit_probabilities, dtype=np.float64)
    semantic = np.asarray(row.semantic_result.outputs.get("edit_probabilities"), dtype=np.float64)
    expected = (len(EDIT_ORDER),)
    if belief.shape != expected or semantic.shape != expected:
        raise ValueError(f"invalid operation probabilities: {row.example_id}")
    for name, values in (("belief", belief), ("semantic", semantic)):
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or abs(float(values.sum()) - 1.0) > 1e-3
        ):
            raise ValueError(f"invalid {name} probabilities: {row.example_id}")
    return belief, semantic


def _semantic_prediction(row: PostAcquisitionToolPairExample) -> int:
    assert row.semantic_result is not None
    value = row.semantic_result.outputs.get("gated_edit")
    if value is None:
        raise ValueError(f"missing calibrated gated_edit: {row.example_id}")
    return EDIT_ORDER.index(EditOperation(str(value)))


def _pre_call_features(row: Any) -> np.ndarray:
    belief = row.post_acquisition_belief
    return np.asarray(
        [
            *belief.edit_probabilities,
            belief.confidence,
            belief.uncertainty,
            *belief.geometry_delta,
            *encode_tool_result(row.quality_result),
            *encode_tool_result(row.temporal_result),
        ],
        dtype=np.float64,
    )


def _observable_features(
    rows: list[PostAcquisitionToolPairExample],
    semantic_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if semantic_indices is None:
        semantic_indices = np.arange(len(rows))
    if semantic_indices.shape != (len(rows),):
        raise ValueError("semantic index shape mismatch")
    features = []
    baseline_prediction = []
    semantic_prediction = []
    target = []
    costs = []
    for index, row in enumerate(rows):
        source = rows[int(semantic_indices[index])]
        belief, _ = _probabilities(row)
        assert source.semantic_result is not None
        baseline_prediction.append(int(np.argmax(belief)))
        semantic_prediction.append(_semantic_prediction(source))
        target.append(EDIT_ORDER.index(row.gt_edit))
        costs.append(float(source.semantic_result.cost))
        features.append(_pre_call_features(row))
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(baseline_prediction, dtype=np.int64),
        np.asarray(semantic_prediction, dtype=np.int64),
        np.asarray(target, dtype=np.int64),
        np.asarray(costs, dtype=np.float64),
    )


def _rewards(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            terminal_reward(EDIT_ORDER[int(expected)], EDIT_ORDER[int(observed)])
            for expected, observed in zip(target, prediction, strict=True)
        ],
        dtype=np.float64,
    )


def _metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    call: np.ndarray,
    costs: np.ndarray,
) -> dict[str, Any]:
    keep = EDIT_ORDER.index(EditOperation.KEEP)
    rewards = _rewards(target, prediction)
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "false_edit_rate": float(
            np.sum((target == keep) & (prediction != keep)) / max(np.sum(target == keep), 1)
        ),
        "missed_edit_rate": float(
            np.sum((target != keep) & (prediction == keep)) / max(np.sum(target != keep), 1)
        ),
        "tool_call_rate": float(np.mean(call)),
        "mean_terminal_reward": float(np.mean(rewards)),
        "mean_tool_cost": float(np.mean(call * costs)),
        "mean_utility": float(np.mean(rewards - call * costs)),
        "confusion_matrix": confusion_matrix(
            target, prediction, labels=np.arange(len(EDIT_ORDER))
        ).tolist(),
    }


def _policy_metrics(
    target: np.ndarray,
    baseline: np.ndarray,
    semantic: np.ndarray,
    costs: np.ndarray,
    call: np.ndarray,
) -> dict[str, Any]:
    prediction = np.where(call, semantic, baseline)
    return _metrics(target, prediction, call, costs)


def _task_permutation_indices(task_ids: list[str], *, seed: int = 0) -> np.ndarray:
    return grouped_derangement(np.asarray(task_ids), seed=seed)


def _fit_gate(c_value: float, seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        ),
    )


def _select_gate(
    features: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray,
    semantic: np.ndarray,
    costs: np.ndarray,
    groups: np.ndarray,
    *,
    c_values: tuple[float, ...],
    thresholds: tuple[float, ...],
    seed: int,
    false_edit_delta: float,
) -> tuple[dict[str, Any], np.ndarray]:
    baseline_reward = _rewards(target, baseline)
    semantic_utility = _rewards(target, semantic) - costs
    beneficial = (semantic_utility > baseline_reward).astype(np.int64)
    baseline_metrics = _policy_metrics(
        target, baseline, semantic, costs, np.zeros(len(target), dtype=bool)
    )
    if len(np.unique(beneficial)) < 2:
        selected = {
            "C": None,
            "threshold": 1.01,
            "metrics": baseline_metrics,
            "reason": "single_class_benefit_labels",
        }
        return selected, beneficial

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    grid = []
    for c_value in c_values:
        probability = np.zeros(len(target), dtype=np.float64)
        for train_indices, holdout_indices in splitter.split(features, beneficial, groups):
            model = _fit_gate(c_value, seed)
            model.fit(features[train_indices], beneficial[train_indices])
            probability[holdout_indices] = model.predict_proba(features[holdout_indices])[:, 1]
        for threshold in thresholds:
            metrics = _policy_metrics(target, baseline, semantic, costs, probability >= threshold)
            grid.append({"C": c_value, "threshold": threshold, "metrics": metrics})
    false_cap = baseline_metrics["false_edit_rate"] + false_edit_delta
    selected = max(
        grid,
        key=lambda row: (
            row["metrics"]["false_edit_rate"] <= false_cap + 1e-12,
            row["metrics"]["mean_utility"],
            row["metrics"]["macro_f1"],
            -row["metrics"]["tool_call_rate"],
            -row["C"],
        ),
    )
    return {
        **selected,
        "false_edit_cap": false_cap,
        "grid": grid,
        "baseline_metrics": baseline_metrics,
    }, beneficial


def evaluate(
    train_rows: list[PostAcquisitionToolPairExample],
    val_rows: list[PostAcquisitionToolPairExample],
    *,
    c_values: tuple[float, ...],
    thresholds: tuple[float, ...],
    seed: int,
    false_edit_delta: float,
) -> dict[str, Any]:
    if {row.split for row in train_rows} != {"train"}:
        raise ValueError("train rows must have split=train")
    if {row.split for row in val_rows} != {"val"}:
        raise ValueError("validation rows must have split=val")
    train = _observable_features(train_rows)
    val = _observable_features(val_rows)
    selected, benefit = _select_gate(
        train[0],
        train[3],
        train[1],
        train[2],
        train[4],
        np.asarray([row.task_id for row in train_rows]),
        c_values=c_values,
        thresholds=thresholds,
        seed=seed,
        false_edit_delta=false_edit_delta,
    )
    if selected["C"] is None:
        gate_probability = np.zeros(len(val_rows), dtype=np.float64)
    else:
        gate = _fit_gate(float(selected["C"]), seed)
        gate.fit(train[0], benefit)
        gate_probability = gate.predict_proba(val[0])[:, 1]
    threshold = float(selected["threshold"])
    selective_call = gate_probability >= threshold
    no_call = np.zeros(len(val_rows), dtype=bool)
    forced_call = np.ones(len(val_rows), dtype=bool)
    oracle_call = _rewards(val[3], val[2]) - val[4] > _rewards(val[3], val[1])

    mismatch_indices = _task_permutation_indices([row.task_id for row in val_rows], seed=seed)
    mismatch = _observable_features(val_rows, mismatch_indices)
    if selected["C"] is None:
        mismatch_call = np.zeros(len(val_rows), dtype=bool)
    else:
        mismatch_call = gate.predict_proba(mismatch[0])[:, 1] >= threshold
    no_tool_metrics = _policy_metrics(val[3], val[1], val[2], val[4], no_call)
    selective_metrics = _policy_metrics(val[3], val[1], val[2], val[4], selective_call)
    mismatch_metrics = _policy_metrics(
        mismatch[3], mismatch[1], mismatch[2], mismatch[4], mismatch_call
    )
    promotion = {
        "utility_above_no_tool": selective_metrics["mean_utility"]
        > no_tool_metrics["mean_utility"],
        "macro_f1_not_below_no_tool": selective_metrics["macro_f1"] >= no_tool_metrics["macro_f1"],
        "false_edit_within_delta": selective_metrics["false_edit_rate"]
        <= no_tool_metrics["false_edit_rate"] + false_edit_delta,
        "utility_above_cross_task_mismatch": selective_metrics["mean_utility"]
        > mismatch_metrics["mean_utility"],
    }
    return {
        "schema_version": "selective-semantic-tool-gate-v1",
        "selection_protocol": "five-fold-stratified-group-OOF-on-train-only",
        "selector_feature_protocol": "pre-call-belief-quality-temporal-only",
        "seed": seed,
        "tool_cost": sorted(
            {float(row.semantic_result.cost) for row in train_rows if row.semantic_result}
        ),
        "train_examples": len(train_rows),
        "train_tasks": len({row.task_id for row in train_rows}),
        "train_beneficial_tool_rate": float(np.mean(benefit)),
        "selected_gate": selected,
        "validation": {
            "no_tool": no_tool_metrics,
            "forced_tool": _policy_metrics(val[3], val[1], val[2], val[4], forced_call),
            "selective_tool": selective_metrics,
            "oracle_tool": _policy_metrics(val[3], val[1], val[2], val[4], oracle_call),
            "cross_task_mismatch": mismatch_metrics,
        },
        "mismatch_control": {
            "all_rows_cross_task": bool(
                np.all(
                    np.asarray([row.task_id for row in val_rows])
                    != np.asarray([row.task_id for row in val_rows])[mismatch_indices]
                )
            ),
            "permutation_sha256": permutation_sha256(mismatch_indices),
        },
        "promotion_gate": {**promotion, "passed": all(promotion.values())},
        "validation_examples": len(val_rows),
        "validation_tasks": len({row.task_id for row in val_rows}),
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--C", dest="c_values", default="0.01,0.1,1,10")
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--model-training-seed", type=int, required=True)
    parser.add_argument("--false-edit-delta", type=float, default=0.02)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = evaluate(
        _read(args.train_jsonl),
        _read(args.val_jsonl),
        c_values=tuple(float(value) for value in args.c_values.split(",")),
        thresholds=tuple(float(value) for value in args.thresholds.split(",")),
        seed=args.seed,
        false_edit_delta=args.false_edit_delta,
    )
    payload["model_training_seed"] = args.model_training_seed
    payload["sources"] = {
        "train": {"path": str(args.train_jsonl.resolve()), "sha256": _sha256(args.train_jsonl)},
        "val": {"path": str(args.val_jsonl.resolve()), "sha256": _sha256(args.val_jsonl)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
