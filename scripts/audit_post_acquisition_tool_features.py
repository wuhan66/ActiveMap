#!/usr/bin/env python3
"""Audit whether paired tool features add label information beyond map belief."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from activemap.agent.tool_belief_data import PostAcquisitionToolPairExample
from activemap.agent.tool_belief_model import encode_belief
from activemap.agent.tool_features import (
    encode_semantic_tool_result,
    encode_tool_result,
)
from activemap.models import EditOperation

EDIT_ORDER = list(EditOperation)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> list[PostAcquisitionToolPairExample]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(PostAcquisitionToolPairExample.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    return rows


def _matrices(
    rows: list[PostAcquisitionToolPairExample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    belief = np.asarray(
        [encode_belief(row.post_acquisition_belief) for row in rows], dtype=np.float64
    )
    tools = np.asarray(
        [
            encode_tool_result(row.quality_result)
            + encode_tool_result(row.temporal_result)
            for row in rows
        ],
        dtype=np.float64,
    )
    target = np.asarray([EDIT_ORDER.index(row.gt_edit) for row in rows], dtype=np.int64)
    semantic_presence = [row.semantic_result is not None for row in rows]
    if any(semantic_presence) and not all(semantic_presence):
        raise ValueError("dataset mixes examples with and without semantic tools")
    semantic = (
        np.asarray(
            [encode_semantic_tool_result(row.semantic_result) for row in rows],
            dtype=np.float64,
        )
        if all(semantic_presence)
        else None
    )
    return belief, tools, semantic, target


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


def cyclically_shuffle_tools(tools: np.ndarray) -> np.ndarray:
    if len(tools) < 2:
        raise ValueError("tool permutation requires at least two examples")
    return np.roll(tools, 1, axis=0)


def audit(
    train_rows: list[PostAcquisitionToolPairExample],
    val_rows: list[PostAcquisitionToolPairExample],
    *,
    c_values: tuple[float, ...],
    seed: int,
) -> dict[str, Any]:
    if not c_values or any(value <= 0.0 for value in c_values):
        raise ValueError("C values must be positive")
    train_belief, train_tools, train_semantic, train_target = _matrices(train_rows)
    val_belief, val_tools, val_semantic, val_target = _matrices(val_rows)
    if (train_semantic is None) != (val_semantic is None):
        raise ValueError("train and validation semantic-tool support differ")
    if {row.split for row in train_rows} != {"train"}:
        raise ValueError("train rows must use split=train")
    if {row.split for row in val_rows} != {"val"}:
        raise ValueError("validation rows must use split=val")
    feature_sets = {
        "belief_only": (train_belief, val_belief),
        "tool_only": (train_tools, val_tools),
        "belief_plus_tool": (
            np.concatenate([train_belief, train_tools], axis=1),
            np.concatenate([val_belief, val_tools], axis=1),
        ),
    }
    primary_full = "belief_plus_tool"
    if train_semantic is not None and val_semantic is not None:
        feature_sets.update(
            {
                "semantic_only": (train_semantic, val_semantic),
                "belief_plus_semantic": (
                    np.concatenate([train_belief, train_semantic], axis=1),
                    np.concatenate([val_belief, val_semantic], axis=1),
                ),
                "belief_plus_all": (
                    np.concatenate(
                        [train_belief, train_tools, train_semantic], axis=1
                    ),
                    np.concatenate([val_belief, val_tools, val_semantic], axis=1),
                ),
            }
        )
        primary_full = "belief_plus_all"
    grid = []
    fitted_full = {}
    for c_value in c_values:
        for name, (train_features, val_features) in feature_sets.items():
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=seed,
                ),
            )
            model.fit(train_features, train_target)
            metrics = operation_metrics(val_target, model.predict(val_features))
            grid.append({"feature_set": name, "C": c_value, "metrics": metrics})
            if name == primary_full:
                fitted_full[c_value] = model

    best = {}
    for name in feature_sets:
        candidates = [row for row in grid if row["feature_set"] == name]
        best[name] = max(
            candidates,
            key=lambda row: (
                row["metrics"]["macro_f1"],
                -row["metrics"]["false_edit_rate"],
                -row["C"],
            ),
        )
    full_best = best[primary_full]
    shuffled_parts = [val_belief, cyclically_shuffle_tools(val_tools)]
    if val_semantic is not None:
        shuffled_parts.append(cyclically_shuffle_tools(val_semantic))
    shuffled_features = np.concatenate(shuffled_parts, axis=1)
    shuffled_metrics = operation_metrics(
        val_target, fitted_full[full_best["C"]].predict(shuffled_features)
    )
    full_f1 = float(full_best["metrics"]["macro_f1"])
    belief_f1 = float(best["belief_only"]["metrics"]["macro_f1"])
    tool_f1 = float(best["tool_only"]["metrics"]["macro_f1"])
    shuffled_f1 = float(shuffled_metrics["macro_f1"])
    return {
        "schema_version": "post-acquisition-tool-feature-audit-v1",
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "seed": seed,
        "C_grid": list(c_values),
        "grid": grid,
        "best": best,
        "permutation": {
            "kind": "cyclic_tool_pair_shift_by_one",
            "metrics": shuffled_metrics,
        },
        "deltas": {
            "full_minus_belief_macro_f1": full_f1 - belief_f1,
            "full_minus_tool_macro_f1": full_f1 - tool_f1,
            "full_minus_shuffled_macro_f1": full_f1 - shuffled_f1,
            **(
                {
                    "full_minus_belief_plus_semantic_macro_f1": full_f1
                    - float(best["belief_plus_semantic"]["metrics"]["macro_f1"]),
                    "belief_plus_semantic_minus_belief_macro_f1": float(
                        best["belief_plus_semantic"]["metrics"]["macro_f1"]
                    )
                    - belief_f1,
                }
                if "belief_plus_semantic" in best
                else {}
            ),
        },
        "primary_full_feature_set": primary_full,
        "incremental_tool_signal_detected": (
            full_f1 > belief_f1 and full_f1 > shuffled_f1
        ),
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--C", dest="c_values", default="0.01,0.1,1,10")
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    c_values = tuple(float(item) for item in args.c_values.split(","))
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = audit(
        _read(args.train_jsonl),
        _read(args.val_jsonl),
        c_values=c_values,
        seed=args.seed,
    )
    payload["sources"] = {
        "train": {"path": str(args.train_jsonl.resolve()), "sha256": _sha256(args.train_jsonl)},
        "val": {"path": str(args.val_jsonl.resolve()), "sha256": _sha256(args.val_jsonl)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
