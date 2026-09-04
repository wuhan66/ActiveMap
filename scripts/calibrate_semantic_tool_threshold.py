#!/usr/bin/env python3
"""Select a semantic threshold by grouped train CV and evaluate validation once."""

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
from activemap.agent.tool_belief_model import encode_belief
from activemap.agent.tool_features import encode_semantic_tool_result, encode_tool_result
from activemap.evaluation_controls import grouped_derangement, permutation_sha256
from activemap.models import EditOperation

EDIT_ORDER = list(EditOperation)


def operation_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    keep = EDIT_ORDER.index(EditOperation.KEEP)
    keep_count = max(int(np.sum(target == keep)), 1)
    update_count = max(int(np.sum(target != keep)), 1)
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "false_edit_rate": float(
            np.sum((target == keep) & (prediction != keep)) / keep_count
        ),
        "missed_edit_rate": float(
            np.sum((target != keep) & (prediction == keep)) / update_count
        ),
        "confusion_matrix": confusion_matrix(
            target, prediction, labels=np.arange(len(EDIT_ORDER))
        ).tolist(),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> list[PostAcquisitionToolPairExample]:
    rows = [
        PostAcquisitionToolPairExample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    return rows


def _threshold_entries(row: PostAcquisitionToolPairExample) -> dict[float, dict[str, Any]]:
    if row.semantic_result is None:
        raise ValueError("semantic threshold calibration requires semantic results")
    entries = row.semantic_result.outputs.get("threshold_sweep")
    if not isinstance(entries, list) or not entries:
        raise ValueError("semantic result does not contain threshold_sweep")
    return {float(entry["threshold"]): entry for entry in entries}


def _semantic_matrix(
    rows: list[PostAcquisitionToolPairExample], threshold: float
) -> np.ndarray:
    features = []
    for row in rows:
        assert row.semantic_result is not None
        entries = _threshold_entries(row)
        matches = [value for key, value in entries.items() if abs(key - threshold) < 1e-8]
        if len(matches) != 1:
            raise ValueError(f"threshold {threshold} missing for {row.example_id}")
        result = row.semantic_result.model_copy(
            update={"outputs": {**row.semantic_result.outputs, **matches[0]}}
        )
        features.append(encode_semantic_tool_result(result))
    return np.asarray(features, dtype=np.float64)


def _belief_matrix(rows: list[PostAcquisitionToolPairExample]) -> np.ndarray:
    return np.asarray(
        [encode_belief(row.post_acquisition_belief) for row in rows],
        dtype=np.float64,
    )


def _weak_tool_matrix(rows: list[PostAcquisitionToolPairExample]) -> np.ndarray:
    return np.asarray(
        [
            encode_tool_result(row.quality_result)
            + encode_tool_result(row.temporal_result)
            for row in rows
        ],
        dtype=np.float64,
    )


def _targets(rows: list[PostAcquisitionToolPairExample]) -> np.ndarray:
    return np.asarray([EDIT_ORDER.index(row.gt_edit) for row in rows], dtype=np.int64)


def _model(c_value: float, seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        ),
    )


def _cross_validated_metrics(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    c_value: float,
    seed: int,
) -> dict[str, Any]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    prediction = np.full_like(target, -1)
    for train_indices, holdout_indices in splitter.split(features, target, groups):
        model = _model(c_value, seed)
        model.fit(features[train_indices], target[train_indices])
        prediction[holdout_indices] = model.predict(features[holdout_indices])
    if np.any(prediction < 0):
        raise RuntimeError("grouped cross-validation left examples unevaluated")
    return operation_metrics(target, prediction)


def calibrate(
    train_rows: list[PostAcquisitionToolPairExample],
    val_rows: list[PostAcquisitionToolPairExample],
    *,
    c_values: tuple[float, ...],
    seed: int,
    fixed_threshold: float | None = None,
) -> dict[str, Any]:
    if {row.split for row in train_rows} != {"train"}:
        raise ValueError("train rows must use split=train")
    if {row.split for row in val_rows} != {"val"}:
        raise ValueError("validation rows must use split=val")
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("C values must be positive")
    threshold_sets = [set(_threshold_entries(row)) for row in train_rows + val_rows]
    thresholds = sorted(set.intersection(*threshold_sets))
    if not thresholds:
        raise ValueError("train and validation have no common semantic thresholds")
    if fixed_threshold is not None:
        matches = [value for value in thresholds if abs(value - fixed_threshold) < 1e-8]
        if len(matches) != 1:
            raise ValueError(f"fixed threshold {fixed_threshold} is unavailable")
        thresholds = matches

    train_belief = _belief_matrix(train_rows)
    val_belief = _belief_matrix(val_rows)
    train_weak = _weak_tool_matrix(train_rows)
    val_weak = _weak_tool_matrix(val_rows)
    train_target, val_target = _targets(train_rows), _targets(val_rows)
    groups = np.asarray([row.task_id for row in train_rows])
    val_groups = np.asarray([row.task_id for row in val_rows])
    mismatch_indices = grouped_derangement(val_groups, seed=seed)

    baseline_grid = []
    for c_value in c_values:
        baseline_grid.append(
            {
                "C": c_value,
                "metrics": _cross_validated_metrics(
                    train_belief,
                    train_target,
                    groups,
                    c_value=c_value,
                    seed=seed,
                ),
            }
        )
    baseline_selected = max(
        baseline_grid,
        key=lambda row: (
            row["metrics"]["macro_f1"],
            -row["metrics"]["false_edit_rate"],
            -row["C"],
        ),
    )

    weak_grid = []
    train_belief_weak = np.concatenate([train_belief, train_weak], axis=1)
    val_belief_weak = np.concatenate([val_belief, val_weak], axis=1)
    for c_value in c_values:
        weak_grid.append(
            {
                "C": c_value,
                "metrics": _cross_validated_metrics(
                    train_belief_weak,
                    train_target,
                    groups,
                    c_value=c_value,
                    seed=seed,
                ),
            }
        )
    weak_selected = max(
        weak_grid,
        key=lambda row: (
            row["metrics"]["macro_f1"],
            -row["metrics"]["false_edit_rate"],
            -row["C"],
        ),
    )

    semantic_grid = []
    all_grid = []
    train_semantic_by_threshold = {}
    for threshold in thresholds:
        semantic = _semantic_matrix(train_rows, threshold)
        train_semantic_by_threshold[threshold] = semantic
        features = np.concatenate([train_belief, semantic], axis=1)
        for c_value in c_values:
            semantic_grid.append(
                {
                    "threshold": threshold,
                    "C": c_value,
                    "metrics": _cross_validated_metrics(
                        features,
                        train_target,
                        groups,
                        c_value=c_value,
                        seed=seed,
                    ),
                }
            )
            all_grid.append(
                {
                    "threshold": threshold,
                    "C": c_value,
                    "metrics": _cross_validated_metrics(
                        np.concatenate([train_belief_weak, semantic], axis=1),
                        train_target,
                        groups,
                        c_value=c_value,
                        seed=seed,
                    ),
                }
            )
    semantic_selected = max(
        semantic_grid,
        key=lambda row: (
            row["metrics"]["macro_f1"],
            -row["metrics"]["false_edit_rate"],
            -abs(row["threshold"] - 0.341),
            -row["C"],
        ),
    )
    all_selected = max(
        all_grid,
        key=lambda row: (
            row["metrics"]["macro_f1"],
            -row["metrics"]["false_edit_rate"],
            -abs(row["threshold"] - (fixed_threshold or 0.341)),
            -row["C"],
        ),
    )

    baseline_model = _model(float(baseline_selected["C"]), seed)
    baseline_model.fit(train_belief, train_target)
    baseline_metrics = operation_metrics(val_target, baseline_model.predict(val_belief))
    weak_model = _model(float(weak_selected["C"]), seed)
    weak_model.fit(train_belief_weak, train_target)
    weak_metrics = operation_metrics(
        val_target, weak_model.predict(val_belief_weak)
    )

    threshold = float(semantic_selected["threshold"])
    train_semantic = train_semantic_by_threshold[threshold]
    val_semantic = _semantic_matrix(val_rows, threshold)
    train_full = np.concatenate([train_belief, train_semantic], axis=1)
    val_full = np.concatenate([val_belief, val_semantic], axis=1)
    full_model = _model(float(semantic_selected["C"]), seed)
    full_model.fit(train_full, train_target)
    full_metrics = operation_metrics(val_target, full_model.predict(val_full))
    mismatch_metrics = operation_metrics(
        val_target,
        full_model.predict(
            np.concatenate([val_belief, val_semantic[mismatch_indices]], axis=1)
        ),
    )
    all_threshold = float(all_selected["threshold"])
    train_all_semantic = train_semantic_by_threshold[all_threshold]
    val_all_semantic = _semantic_matrix(val_rows, all_threshold)
    train_all = np.concatenate([train_belief_weak, train_all_semantic], axis=1)
    val_all = np.concatenate([val_belief_weak, val_all_semantic], axis=1)
    all_model = _model(float(all_selected["C"]), seed)
    all_model.fit(train_all, train_target)
    all_metrics = operation_metrics(val_target, all_model.predict(val_all))
    all_mismatch_metrics = operation_metrics(
        val_target,
        all_model.predict(
            np.concatenate(
                [val_belief_weak, val_all_semantic[mismatch_indices]], axis=1
            )
        ),
    )
    gate = {
        "beats_belief_only": full_metrics["macro_f1"] > baseline_metrics["macro_f1"],
        "beats_task_deranged_mismatch": full_metrics["macro_f1"]
        > mismatch_metrics["macro_f1"],
        "false_edit_within_delta_0_02": full_metrics["false_edit_rate"]
        <= baseline_metrics["false_edit_rate"] + 0.02,
    }
    interaction_gate = {
        "beats_belief_plus_weak_tools": all_metrics["macro_f1"]
        > weak_metrics["macro_f1"],
        "beats_task_deranged_semantic_mismatch": all_metrics["macro_f1"]
        > all_mismatch_metrics["macro_f1"],
        "false_edit_within_delta_0_02": all_metrics["false_edit_rate"]
        <= weak_metrics["false_edit_rate"] + 0.02,
    }
    return {
        "schema_version": "semantic-threshold-calibration-v1",
        "selection_protocol": "five-fold-stratified-group-CV-on-train-only",
        "semantic_mismatch_protocol": "seeded-task-derangement-on-validation",
        "semantic_mismatch_control": {
            "seed": seed,
            "all_rows_cross_task": bool(
                np.all(val_groups != val_groups[mismatch_indices])
            ),
            "permutation_sha256": permutation_sha256(mismatch_indices),
        },
        "cv_seed": seed,
        "fixed_threshold": fixed_threshold,
        "thresholds": thresholds,
        "C_grid": list(c_values),
        "train_examples": len(train_rows),
        "train_tasks": len(set(groups)),
        "val_examples": len(val_rows),
        "baseline_cv_grid": baseline_grid,
        "weak_tool_cv_grid": weak_grid,
        "semantic_cv_grid": semantic_grid,
        "all_tool_cv_grid": all_grid,
        "selected": {
            "belief_only": baseline_selected,
            "belief_plus_weak_tools": weak_selected,
            "belief_plus_semantic": semantic_selected,
            "belief_plus_all": all_selected,
        },
        "validation": {
            "belief_only": baseline_metrics,
            "belief_plus_weak_tools": weak_metrics,
            "belief_plus_semantic": full_metrics,
            "task_deranged_semantic_mismatch": mismatch_metrics,
            "belief_plus_all": all_metrics,
            "all_with_task_deranged_semantic_mismatch": all_mismatch_metrics,
            "deltas": {
                "full_minus_belief_macro_f1": full_metrics["macro_f1"]
                - baseline_metrics["macro_f1"],
                "full_minus_mismatch_macro_f1": full_metrics["macro_f1"]
                - mismatch_metrics["macro_f1"],
                "full_minus_belief_false_edit": full_metrics["false_edit_rate"]
                - baseline_metrics["false_edit_rate"],
                "all_minus_weak_macro_f1": all_metrics["macro_f1"]
                - weak_metrics["macro_f1"],
                "all_minus_task_deranged_semantic_mismatch_macro_f1": all_metrics[
                    "macro_f1"
                ]
                - all_mismatch_metrics["macro_f1"],
                "all_minus_weak_false_edit": all_metrics["false_edit_rate"]
                - weak_metrics["false_edit_rate"],
            },
        },
        "gate": {**gate, "passed": all(gate.values())},
        "interaction_gate": {
            **interaction_gate,
            "passed": all(interaction_gate.values()),
        },
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--C", dest="c_values", default="0.01,0.1,1,10")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--model-training-seed", type=int)
    parser.add_argument("--fixed-threshold", type=float)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = calibrate(
        _read(args.train_jsonl),
        _read(args.val_jsonl),
        c_values=tuple(float(value) for value in args.c_values.split(",")),
        seed=args.seed,
        fixed_threshold=args.fixed_threshold,
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
