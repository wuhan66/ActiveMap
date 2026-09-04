#!/usr/bin/env python3
"""Train-only safety calibration for the aligned semantic Belief gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from activemap.evaluation_controls import grouped_derangement, permutation_sha256
from activemap.models import EditOperation
from activemap.safe_commit import (
    apply_commit_threshold,
    apply_keep_preserving_guard,
    select_safety_candidate,
)
from scripts.calibrate_semantic_tool_threshold import (
    EDIT_ORDER,
    _belief_matrix,
    _model,
    _read,
    _semantic_matrix,
    _targets,
    _weak_tool_matrix,
    operation_metrics,
)

KEEP_INDEX = EDIT_ORDER.index(EditOperation.KEEP)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _oof_probabilities(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    c_value: float,
    seed: int,
) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    probabilities = np.full((len(target), len(EDIT_ORDER)), np.nan, dtype=np.float64)
    for train_indices, holdout_indices in splitter.split(features, target, groups):
        model = _model(c_value, seed)
        model.fit(features[train_indices], target[train_indices])
        fold = model.predict_proba(features[holdout_indices])
        probabilities[holdout_indices[:, None], model.classes_[None, :]] = fold
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("grouped cross-validation left probabilities unevaluated")
    return probabilities


def _select(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    c_values: tuple[float, ...],
    commit_thresholds: tuple[float, ...],
    false_edit_cap: float,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    for c_value in c_values:
        probabilities = _oof_probabilities(features, target, groups, c_value=c_value, seed=seed)
        for threshold in commit_thresholds:
            prediction = apply_commit_threshold(
                probabilities, keep_index=KEEP_INDEX, threshold=threshold
            )
            candidates.append(
                {
                    "C": c_value,
                    "commit_threshold": threshold,
                    "metrics": operation_metrics(target, prediction),
                }
            )
    return select_safety_candidate(candidates, false_edit_cap=false_edit_cap), candidates


def _evaluate(
    train_features: np.ndarray,
    train_target: np.ndarray,
    val_features: np.ndarray,
    val_target: np.ndarray,
    selected: dict[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], Any, np.ndarray]:
    model = _model(float(selected["C"]), seed)
    model.fit(train_features, train_target)
    probabilities = model.predict_proba(val_features)
    prediction = apply_commit_threshold(
        probabilities,
        keep_index=KEEP_INDEX,
        threshold=float(selected["commit_threshold"]),
    )
    return operation_metrics(val_target, prediction), model, prediction


def calibrate_safe(
    train_rows: list[Any],
    val_rows: list[Any],
    *,
    fixed_threshold: float,
    c_values: tuple[float, ...],
    commit_thresholds: tuple[float, ...],
    false_edit_cap: float,
    seed: int,
) -> dict[str, Any]:
    if {row.split for row in train_rows} != {"train"}:
        raise ValueError("train rows must use split=train")
    if {row.split for row in val_rows} != {"val"}:
        raise ValueError("validation rows must use split=val")
    train_target, val_target = _targets(train_rows), _targets(val_rows)
    train_groups = np.asarray([row.task_id for row in train_rows])
    val_groups = np.asarray([row.task_id for row in val_rows])
    mismatch_indices = grouped_derangement(val_groups, seed=seed)

    train_belief, val_belief = _belief_matrix(train_rows), _belief_matrix(val_rows)
    train_weak, val_weak = _weak_tool_matrix(train_rows), _weak_tool_matrix(val_rows)
    train_semantic = _semantic_matrix(train_rows, fixed_threshold)
    val_semantic = _semantic_matrix(val_rows, fixed_threshold)
    matrices = {
        "belief_only": (train_belief, val_belief),
        "belief_plus_weak_tools": (
            np.concatenate([train_belief, train_weak], axis=1),
            np.concatenate([val_belief, val_weak], axis=1),
        ),
        "belief_plus_semantic": (
            np.concatenate([train_belief, train_semantic], axis=1),
            np.concatenate([val_belief, val_semantic], axis=1),
        ),
        "belief_plus_all": (
            np.concatenate([train_belief, train_weak, train_semantic], axis=1),
            np.concatenate([val_belief, val_weak, val_semantic], axis=1),
        ),
    }

    selected, grids, validation, models, predictions = {}, {}, {}, {}, {}
    for name, (train_features, val_features) in matrices.items():
        selected[name], grids[name] = _select(
            train_features,
            train_target,
            train_groups,
            c_values=c_values,
            commit_thresholds=commit_thresholds,
            false_edit_cap=false_edit_cap,
            seed=seed,
        )
        validation[name], models[name], predictions[name] = _evaluate(
            train_features,
            train_target,
            val_features,
            val_target,
            selected[name],
            seed=seed,
        )

    semantic_selected = selected["belief_plus_semantic"]
    semantic_model = models["belief_plus_semantic"]
    mismatch_probabilities = semantic_model.predict_proba(
        np.concatenate([val_belief, val_semantic[mismatch_indices]], axis=1)
    )
    mismatch_prediction = apply_commit_threshold(
        mismatch_probabilities,
        keep_index=KEEP_INDEX,
        threshold=float(semantic_selected["commit_threshold"]),
    )
    validation["task_deranged_semantic_mismatch"] = operation_metrics(
        val_target, mismatch_prediction
    )
    all_selected = selected["belief_plus_all"]
    all_model = models["belief_plus_all"]
    all_mismatch_probabilities = all_model.predict_proba(
        np.concatenate([val_belief, val_weak, val_semantic[mismatch_indices]], axis=1)
    )
    all_mismatch_prediction = apply_commit_threshold(
        all_mismatch_probabilities,
        keep_index=KEEP_INDEX,
        threshold=float(all_selected["commit_threshold"]),
    )
    validation["all_with_task_deranged_semantic_mismatch"] = operation_metrics(
        val_target, all_mismatch_prediction
    )

    belief = validation["belief_only"]
    semantic = validation["belief_plus_semantic"]
    mismatch = validation["task_deranged_semantic_mismatch"]
    weak = validation["belief_plus_weak_tools"]
    all_tools = validation["belief_plus_all"]
    all_mismatch = validation["all_with_task_deranged_semantic_mismatch"]
    direct_gate = {
        "beats_belief_only": semantic["macro_f1"] > belief["macro_f1"],
        "beats_task_deranged_mismatch": semantic["macro_f1"] > mismatch["macro_f1"],
        "false_edit_within_delta_0_02": semantic["false_edit_rate"]
        <= belief["false_edit_rate"] + 0.02,
    }
    interaction_gate = {
        "beats_belief_plus_weak_tools": all_tools["macro_f1"] > weak["macro_f1"],
        "beats_task_deranged_semantic_mismatch": all_tools["macro_f1"] > all_mismatch["macro_f1"],
        "false_edit_within_delta_0_02": all_tools["false_edit_rate"]
        <= weak["false_edit_rate"] + 0.02,
    }
    validation["deltas"] = {
        "full_minus_belief_macro_f1": semantic["macro_f1"] - belief["macro_f1"],
        "full_minus_mismatch_macro_f1": semantic["macro_f1"] - mismatch["macro_f1"],
        "full_minus_belief_false_edit": semantic["false_edit_rate"] - belief["false_edit_rate"],
        "all_minus_weak_macro_f1": all_tools["macro_f1"] - weak["macro_f1"],
        "all_minus_task_deranged_semantic_mismatch_macro_f1": all_tools["macro_f1"]
        - all_mismatch["macro_f1"],
        "all_minus_weak_false_edit": all_tools["false_edit_rate"] - weak["false_edit_rate"],
    }

    guarded_predictions = {
        "belief_only": predictions["belief_only"],
        "belief_plus_weak_tools": predictions["belief_plus_weak_tools"],
        "belief_plus_semantic": apply_keep_preserving_guard(
            predictions["belief_only"],
            predictions["belief_plus_semantic"],
            keep_index=KEEP_INDEX,
        ),
        "task_deranged_semantic_mismatch": apply_keep_preserving_guard(
            predictions["belief_only"], mismatch_prediction, keep_index=KEEP_INDEX
        ),
        "belief_plus_all": apply_keep_preserving_guard(
            predictions["belief_plus_weak_tools"],
            predictions["belief_plus_all"],
            keep_index=KEEP_INDEX,
        ),
        "all_with_task_deranged_semantic_mismatch": apply_keep_preserving_guard(
            predictions["belief_plus_weak_tools"],
            all_mismatch_prediction,
            keep_index=KEEP_INDEX,
        ),
    }
    guarded_validation = {
        name: operation_metrics(val_target, prediction)
        for name, prediction in guarded_predictions.items()
    }
    guarded_belief = guarded_validation["belief_only"]
    guarded_semantic = guarded_validation["belief_plus_semantic"]
    guarded_mismatch = guarded_validation["task_deranged_semantic_mismatch"]
    guarded_weak = guarded_validation["belief_plus_weak_tools"]
    guarded_all = guarded_validation["belief_plus_all"]
    guarded_all_mismatch = guarded_validation["all_with_task_deranged_semantic_mismatch"]
    guarded_gate = {
        "beats_belief_only": guarded_semantic["macro_f1"] > guarded_belief["macro_f1"],
        "beats_task_deranged_mismatch": guarded_semantic["macro_f1"] > guarded_mismatch["macro_f1"],
        "false_edit_not_above_baseline": guarded_semantic["false_edit_rate"]
        <= guarded_belief["false_edit_rate"],
    }
    guarded_interaction_gate = {
        "beats_belief_plus_weak_tools": guarded_all["macro_f1"] > guarded_weak["macro_f1"],
        "beats_task_deranged_semantic_mismatch": guarded_all["macro_f1"]
        > guarded_all_mismatch["macro_f1"],
        "false_edit_not_above_baseline": guarded_all["false_edit_rate"]
        <= guarded_weak["false_edit_rate"],
    }
    guarded_validation["deltas"] = {
        "full_minus_belief_macro_f1": guarded_semantic["macro_f1"] - guarded_belief["macro_f1"],
        "full_minus_mismatch_macro_f1": guarded_semantic["macro_f1"] - guarded_mismatch["macro_f1"],
        "full_minus_belief_false_edit": guarded_semantic["false_edit_rate"]
        - guarded_belief["false_edit_rate"],
        "all_minus_weak_macro_f1": guarded_all["macro_f1"] - guarded_weak["macro_f1"],
        "all_minus_task_deranged_semantic_mismatch_macro_f1": guarded_all["macro_f1"]
        - guarded_all_mismatch["macro_f1"],
        "all_minus_weak_false_edit": guarded_all["false_edit_rate"]
        - guarded_weak["false_edit_rate"],
    }
    return {
        "schema_version": "safe-semantic-gate-calibration-v1",
        "selection_protocol": (
            "five-fold-stratified-group-CV-on-train-only-joint-C-and-commit-threshold"
        ),
        "validation_adaptation_stage": "post-argmax-diagnostic-protocol-revision",
        "guard_protocol": "keep-preserving-semantic-residual-no-parameters",
        "semantic_mismatch_protocol": "seeded-task-derangement-on-validation",
        "semantic_mismatch_control": {
            "seed": seed,
            "all_rows_cross_task": bool(np.all(val_groups != val_groups[mismatch_indices])),
            "permutation_sha256": permutation_sha256(mismatch_indices),
        },
        "cv_seed": seed,
        "fixed_threshold": fixed_threshold,
        "C_grid": list(c_values),
        "commit_threshold_grid": list(commit_thresholds),
        "train_false_edit_cap": false_edit_cap,
        "train_examples": len(train_rows),
        "train_tasks": len(set(train_groups)),
        "val_examples": len(val_rows),
        "selected": selected,
        "selection_grids": grids,
        "validation": validation,
        "guarded_validation": guarded_validation,
        "gate": {**direct_gate, "passed": all(direct_gate.values())},
        "interaction_gate": {
            **interaction_gate,
            "passed": all(interaction_gate.values()),
        },
        "guarded_gate": {**guarded_gate, "passed": all(guarded_gate.values())},
        "guarded_interaction_gate": {
            **guarded_interaction_gate,
            "passed": all(guarded_interaction_gate.values()),
        },
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fixed-threshold", type=float, required=True)
    parser.add_argument("--C", dest="c_values", default="0.01,0.1,1,10")
    parser.add_argument(
        "--commit-thresholds",
        default="0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95",
    )
    parser.add_argument("--train-false-edit-cap", type=float, default=0.10)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-training-seed", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = calibrate_safe(
        _read(args.train_jsonl),
        _read(args.val_jsonl),
        fixed_threshold=args.fixed_threshold,
        c_values=tuple(float(value) for value in args.c_values.split(",")),
        commit_thresholds=tuple(float(value) for value in args.commit_thresholds.split(",")),
        false_edit_cap=args.train_false_edit_cap,
        seed=args.seed,
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
