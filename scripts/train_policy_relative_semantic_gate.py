#!/usr/bin/env python3
"""Train a no-leak CALL/STOP gate against the current structured policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from activemap.agent.evidence_value_head import StructuredMapActionPredictor
from activemap.agent.active_catalog_tool_gate import (
    policy_relative_semantic_gate_features,
)
from activemap.agent.identifiers import public_task_id
from activemap.agent.post_tool_action_adapter import PostToolActionAdapterPredictor
from activemap.agent.tool_belief_data import PostAcquisitionToolPairExample
from activemap.agent.tool_sft import terminal_reward
from activemap.evaluation.episode_utility import UTILITY_PROFILES
from activemap.agent.tools import belief_from_features
from activemap.models import EditOperation
from activemap.selector_records import SelectorSample

EDIT_ORDER = list(EditOperation)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_states(path: Path) -> list[SelectorSample]:
    rows = [
        SelectorSample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("empty structured state file")
    return rows


def _semantic_by_task(
    path: Path,
) -> dict[str, PostAcquisitionToolPairExample]:
    result: dict[str, PostAcquisitionToolPairExample] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = PostAcquisitionToolPairExample.model_validate_json(line)
        if row.semantic_result is None:
            raise ValueError(f"missing semantic result: {row.example_id}")
        previous = result.get(row.task_id)
        if previous is not None:
            if previous.semantic_result.outputs != row.semantic_result.outputs:
                raise ValueError(f"task has inconsistent semantic results: {row.task_id}")
            continue
        result[row.task_id] = row
    if not result:
        raise ValueError("empty semantic result file")
    return result


def _fit(c_value: float, seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        ),
    )


def _policy_metrics(
    target: np.ndarray,
    direct: np.ndarray,
    semantic: np.ndarray,
    call: np.ndarray,
    raw_cost: np.ndarray,
    utility_cost: np.ndarray,
) -> dict[str, float]:
    prediction = np.where(call, semantic, direct)
    keep = EDIT_ORDER.index(EditOperation.KEEP)
    rewards = np.asarray(
        [
            terminal_reward(EDIT_ORDER[int(gt)], EDIT_ORDER[int(pred)])
            for gt, pred in zip(target, prediction, strict=True)
        ]
    )
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "macro_f1": float(
            f1_score(target, prediction, average="macro", zero_division=0)
        ),
        "false_edit_rate": float(
            np.mean((target == keep) & (prediction != keep))
        ),
        "missed_edit_rate": float(
            np.mean((target != keep) & (prediction == keep))
        ),
        "call_rate": float(np.mean(call)),
        "mean_cost": float(np.mean(call * raw_cost)),
        "mean_utility_cost": float(np.mean(call * utility_cost)),
        "mean_utility": float(np.mean(rewards - call * utility_cost)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("semantic_train", type=Path)
    parser.add_argument("semantic_val", type=Path)
    parser.add_argument("structured_checkpoint", type=Path)
    parser.add_argument("semantic_adapter", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--false-edit-delta", type=float, default=0.02)
    parser.add_argument(
        "--utility-profile",
        choices=tuple(UTILITY_PROFILES),
        default="balanced",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    structured = StructuredMapActionPredictor(
        str(args.structured_checkpoint), device="cpu"
    )
    semantic_adapter = PostToolActionAdapterPredictor(
        args.semantic_adapter, device="cpu"
    )
    semantic_rows = {
        **_semantic_by_task(args.semantic_train),
        **_semantic_by_task(args.semantic_val),
    }
    records: dict[str, list[Any]] = {
        key: []
        for key in (
            "features",
            "target",
            "direct",
            "semantic",
            "cost",
            "utility_cost",
            "group",
        )
    }
    split_indices: dict[str, list[int]] = {"train": [], "val": []}
    skipped = 0
    for sample in _read_states(args.states):
        task_id = public_task_id(str(sample.metadata["source_episode"]))
        semantic_row = semantic_rows.get(task_id)
        if semantic_row is None:
            skipped += 1
            continue
        _, direct_edit = structured.predict(sample)
        semantic_edit = semantic_adapter.predict(
            belief_from_features(sample),
            [semantic_row.semantic_result],
        )
        index = len(records["target"])
        split_indices[sample.split].append(index)
        records["features"].append(
            policy_relative_semantic_gate_features(sample)
        )
        records["target"].append(EDIT_ORDER.index(EditOperation(str(sample.metadata["gt_edit"]))))
        records["direct"].append(EDIT_ORDER.index(direct_edit))
        records["semantic"].append(EDIT_ORDER.index(semantic_edit))
        raw_cost = float(semantic_row.semantic_result.cost)
        budget = float(sample.metadata["budget"])
        records["cost"].append(raw_cost)
        records["utility_cost"].append(
            UTILITY_PROFILES[args.utility_profile].cost
            * min(raw_cost / budget, 1.0)
        )
        records["group"].append(task_id)

    arrays = {
        key: np.asarray(value)
        for key, value in records.items()
    }
    train_index = np.asarray(split_indices["train"], dtype=np.int64)
    val_index = np.asarray(split_indices["val"], dtype=np.int64)
    if not len(train_index) or not len(val_index):
        raise ValueError("gate data lacks train or validation support")
    train_reward_direct = np.asarray(
        [
            terminal_reward(EDIT_ORDER[int(gt)], EDIT_ORDER[int(pred)])
            for gt, pred in zip(
                arrays["target"][train_index],
                arrays["direct"][train_index],
                strict=True,
            )
        ]
    )
    train_reward_semantic = np.asarray(
        [
            terminal_reward(EDIT_ORDER[int(gt)], EDIT_ORDER[int(pred)])
            for gt, pred in zip(
                arrays["target"][train_index],
                arrays["semantic"][train_index],
                strict=True,
            )
        ]
    )
    beneficial = (
        train_reward_semantic
        - arrays["utility_cost"][train_index]
        > train_reward_direct
    ).astype(np.int64)
    if len(np.unique(beneficial)) != 2:
        raise ValueError("policy-relative labels contain only one class")

    c_values = (0.01, 0.1, 1.0, 10.0)
    thresholds = np.linspace(0.1, 0.9, 33)
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    best = None
    for c_value in c_values:
        probability = np.zeros(len(train_index), dtype=np.float64)
        for fit_index, holdout_index in splitter.split(
            arrays["features"][train_index],
            beneficial,
            arrays["group"][train_index],
        ):
            model = _fit(c_value, args.seed)
            model.fit(
                arrays["features"][train_index][fit_index],
                beneficial[fit_index],
            )
            probability[holdout_index] = model.predict_proba(
                arrays["features"][train_index][holdout_index]
            )[:, 1]
        direct_metrics = _policy_metrics(
            arrays["target"][train_index],
            arrays["direct"][train_index],
            arrays["semantic"][train_index],
            np.zeros(len(train_index), dtype=bool),
            arrays["cost"][train_index],
            arrays["utility_cost"][train_index],
        )
        for threshold in thresholds:
            call = probability >= threshold
            metric = _policy_metrics(
                arrays["target"][train_index],
                arrays["direct"][train_index],
                arrays["semantic"][train_index],
                call,
                arrays["cost"][train_index],
                arrays["utility_cost"][train_index],
            )
            feasible = (
                metric["false_edit_rate"]
                <= direct_metrics["false_edit_rate"] + args.false_edit_delta
            )
            candidate = (
                feasible,
                metric["mean_utility"],
                metric["macro_f1"],
                -metric["call_rate"],
                -c_value,
                -float(threshold),
                c_value,
                float(threshold),
                metric,
            )
            if best is None or candidate[:6] > best[:6]:
                best = candidate
    assert best is not None
    c_value, threshold, train_metrics = best[6], best[7], best[8]
    gate = _fit(c_value, args.seed)
    gate.fit(arrays["features"][train_index], beneficial)
    val_probability = gate.predict_proba(arrays["features"][val_index])[:, 1]
    val_call = val_probability >= threshold
    val_direct = _policy_metrics(
        arrays["target"][val_index],
        arrays["direct"][val_index],
        arrays["semantic"][val_index],
        np.zeros(len(val_index), dtype=bool),
        arrays["cost"][val_index],
        arrays["utility_cost"][val_index],
    )
    val_selective = _policy_metrics(
        arrays["target"][val_index],
        arrays["direct"][val_index],
        arrays["semantic"][val_index],
        val_call,
        arrays["cost"][val_index],
        arrays["utility_cost"][val_index],
    )
    args.output_dir.mkdir(parents=True)
    joblib.dump(gate, args.output_dir / "gate.joblib")
    scaler = gate.named_steps["standardscaler"]
    classifier = gate.named_steps["logisticregression"]
    portable_gate = {
        "schema_version": "linear-probability-gate-v1",
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficient": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
    }
    (args.output_dir / "gate.json").write_text(
        json.dumps(portable_gate, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "policy-relative-semantic-gate-v1",
        "train_examples": int(len(train_index)),
        "val_examples": int(len(val_index)),
        "skipped_states": skipped,
        "train_beneficial_rate": float(np.mean(beneficial)),
        "selected": {
            "C": c_value,
            "threshold": threshold,
            "train_oof_metrics": train_metrics,
        },
        "val": {
            "direct": val_direct,
            "selective": val_selective,
            "deltas": {
                key: val_selective[key] - val_direct[key]
                for key in val_direct
            },
        },
        "sources": {
            "states": _sha256(args.states),
            "semantic_train": _sha256(args.semantic_train),
            "semantic_val": _sha256(args.semantic_val),
            "structured_checkpoint": _sha256(args.structured_checkpoint),
            "semantic_adapter": _sha256(args.semantic_adapter),
        },
        "selection_protocol": "train-only-stratified-grouped-OOF",
        "utility_profile": args.utility_profile,
        "utility_cost_semantics": "profile.cost * min(raw_tool_cost / budget, 1)",
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
