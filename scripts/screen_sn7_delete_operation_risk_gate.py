#!/usr/bin/env python3
"""Screen a train-only multi-feature risk gate for SN7 DELETE writebacks.

The gate only reads quantities available after fusion/vectorization and before
commit. Ground-truth targets and final-map metrics are used exclusively to
construct train labels and evaluate held-out validation behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from scripts.calibrate_sn7_v5_safe_commit import bool_value, target_operation, validate_row

POLICIES = ("direct", "selected")
FEATURE_NAMES = (
    "fused_confidence",
    "budget",
    "spent_cost",
    "semantic_tool_called",
    "fusion_count",
    "fusion_weight_max",
    "fusion_weight_entropy",
    "predicted_component_count",
    "prior_component_count",
    "raw_add_component_count",
    "raw_remove_component_count",
    "retained_add_component_count",
    "retained_remove_component_count",
    "vector_replay_iou",
    "topology_quality_before",
    "topology_quality_after",
    "vector_delta_topology_valid",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_record(value: str) -> tuple[str, Path]:
    policy, separator, raw_path = value.partition("=")
    if not separator or policy not in POLICIES or not raw_path:
        raise argparse.ArgumentTypeError(
            "record must be direct=WRITEBACK_JSONL or selected=WRITEBACK_JSONL"
        )
    return policy, Path(raw_path)


def _entropy(weights: list[float]) -> float:
    if len(weights) <= 1:
        return 0.0
    values = np.asarray(weights, dtype=np.float64)
    if np.any(values < 0.0) or float(values.sum()) <= 0.0:
        raise ValueError("fusion weights must be non-negative with positive sum")
    normalized = values / float(values.sum())
    return float(
        -np.sum(normalized * np.log(np.clip(normalized, 1e-12, 1.0))) / math.log(len(values))
    )


def observable_features(row: dict[str, Any]) -> np.ndarray:
    """Return only post-fusion, pre-commit features available at deployment."""

    required = {
        "fused_confidence",
        "budget",
        "spent_cost",
        "semantic_tool_called",
        "fusion_weights",
        "predicted_component_count",
        "prior_component_count",
        "raw_add_component_count",
        "raw_remove_component_count",
        "retained_add_component_count",
        "retained_remove_component_count",
        "vector_replay_iou",
        "topology_quality_before",
        "topology_quality_after",
        "vector_delta_topology_valid",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"writeback lacks observable features: {missing}")
    weights = row["fusion_weights"]
    if not isinstance(weights, list) or not weights:
        raise ValueError("fusion_weights must be a non-empty list")
    weight_values = [float(value) for value in weights]
    values = (
        float(row["fused_confidence"]),
        float(row["budget"]),
        float(row["spent_cost"]),
        float(bool_value(row["semantic_tool_called"])),
        float(len(weight_values)),
        max(weight_values),
        _entropy(weight_values),
        float(row["predicted_component_count"]),
        float(row["prior_component_count"]),
        float(row["raw_add_component_count"]),
        float(row["raw_remove_component_count"]),
        float(row["retained_add_component_count"]),
        float(row["retained_remove_component_count"]),
        float(row["vector_replay_iou"]),
        float(row["topology_quality_before"]),
        float(row["topology_quality_after"]),
        float(bool_value(row["vector_delta_topology_valid"])),
    )
    return np.asarray(values, dtype=np.float32)


def _target_operation(row: dict[str, Any]) -> str:
    return target_operation(str(row["target"])).value


def collect_delete_rows(
    records: Iterable[tuple[str, Path]],
    *,
    split: str,
    gain_epsilon: float,
) -> dict[str, Any]:
    """Stream changed DELETE rows without retaining full raw writeback records."""

    paths = dict(records)
    if tuple(sorted(paths)) != POLICIES:
        raise ValueError(f"records must contain exactly {POLICIES}")
    vectors = []
    confidence = []
    beneficial = []
    harmful = []
    target_keep = []
    aoi_ids = []
    for _policy, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        seen = set()
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                validate_row(row, expected_split=split)
                key = (str(row["task_id"]), float(row["budget"]))
                if key in seen:
                    raise ValueError(f"duplicate task-budget row in {path}:{line_number}")
                seen.add(key)
                if (
                    not bool_value(row["writeback_changed"])
                    or str(row["effective_operation"]) != "DELETE"
                ):
                    continue
                target = _target_operation(row)
                gain = float(row["raster_iou_gain"])
                vectors.append(observable_features(row))
                confidence.append(float(row["fused_confidence"]))
                beneficial.append(target == "DELETE" and gain > gain_epsilon)
                harmful.append(target != "DELETE" or gain < -gain_epsilon)
                target_keep.append(target == "KEEP")
                aoi_ids.append(str(row["aoi_id"]))
    if not vectors:
        raise ValueError("no changed DELETE proposals found")
    return {
        "features": np.stack(vectors),
        "confidence": np.asarray(confidence, dtype=np.float64),
        "beneficial": np.asarray(beneficial, dtype=bool),
        "harmful": np.asarray(harmful, dtype=bool),
        "target_keep": np.asarray(target_keep, dtype=bool),
        "aoi_ids": np.asarray(aoi_ids, dtype=object),
    }


def _macro_rate(numerator: np.ndarray, denominator: np.ndarray, groups: np.ndarray) -> float:
    names, inverse = np.unique(groups, return_inverse=True)
    totals = np.bincount(inverse, weights=denominator.astype(np.float64), minlength=len(names))
    numerators = np.bincount(inverse, weights=numerator.astype(np.float64), minlength=len(names))
    valid = totals > 0.0
    return float(np.mean(numerators[valid] / totals[valid])) if np.any(valid) else 0.0


def score_gate(scores: np.ndarray, support: dict[str, Any], threshold: float) -> dict[str, float]:
    accepted = scores >= threshold
    beneficial = support["beneficial"]
    harmful = support["harmful"]
    target_keep = support["target_keep"]
    aoi_ids = support["aoi_ids"]
    return {
        "threshold": float(threshold),
        "beneficial_delete_recall": _macro_rate(accepted & beneficial, beneficial, aoi_ids),
        "harmful_accept_rate": _macro_rate(accepted & harmful, harmful, aoi_ids),
        "target_keep_false_edit_rate": _macro_rate(accepted & target_keep, target_keep, aoi_ids),
        "accept_rate": _macro_rate(accepted, np.ones_like(accepted, dtype=bool), aoi_ids),
    }


def select_threshold(
    scores: np.ndarray,
    support: dict[str, Any],
    *,
    maximum_harmful_accept_rate: float,
    maximum_false_edit_rate: float,
    threshold_count: int,
) -> dict[str, float]:
    if threshold_count < 2:
        raise ValueError("threshold_count must be at least two")
    candidates = [
        score_gate(scores, support, float(threshold))
        for threshold in np.linspace(0.0, 1.0, threshold_count)
    ]
    feasible = [
        point
        for point in candidates
        if point["harmful_accept_rate"] <= maximum_harmful_accept_rate + 1e-12
        and point["target_keep_false_edit_rate"] <= maximum_false_edit_rate + 1e-12
    ]
    if not feasible:
        raise RuntimeError("no risk threshold meets the train-only safety constraints")
    return max(
        feasible,
        key=lambda point: (
            point["beneficial_delete_recall"],
            -point["harmful_accept_rate"],
            -point["target_keep_false_edit_rate"],
            # Keep the lowest equally safe threshold.  A higher threshold that
            # admits exactly the same proposals merely spends recall headroom
            # and makes the operating point less stable under score jitter.
            -point["threshold"],
        ),
    )


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        return None
    ranked = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[ranked]
    sorted_labels = labels[ranked]
    rank_sum = 0.0
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        rank_sum += average_rank * float(sorted_labels[start:end].sum())
        start = end
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def _fit_model(
    support: dict[str, Any], *, seed: int, max_iter: int, max_leaf_nodes: int
) -> HistGradientBoostingClassifier:
    labels = support["beneficial"]
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        raise ValueError("risk-gate train support lacks both beneficial and other DELETE proposals")
    weights = np.where(
        labels,
        len(labels) / (2.0 * positive_count),
        len(labels) / (2.0 * negative_count),
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=1e-3,
        random_state=seed,
    )
    model.fit(support["features"], labels.astype(np.int8), sample_weight=weights)
    return model


def screen(
    train_records: Iterable[tuple[str, Path]],
    validation_records: Iterable[tuple[str, Path]],
    *,
    seed: int,
    gain_epsilon: float,
    maximum_harmful_accept_rate: float,
    maximum_false_edit_rate: float,
    threshold_count: int,
    max_iter: int,
    max_leaf_nodes: int,
) -> dict[str, Any]:
    train = collect_delete_rows(train_records, split="train", gain_epsilon=gain_epsilon)
    validation = collect_delete_rows(validation_records, split="val", gain_epsilon=gain_epsilon)
    model = _fit_model(train, seed=seed, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes)
    risk_train = model.predict_proba(train["features"])[:, 1]
    risk_validation = model.predict_proba(validation["features"])[:, 1]
    methods = {
        "confidence": (train["confidence"], validation["confidence"]),
        "risk_gate": (risk_train, risk_validation),
    }
    result_methods = {}
    for name, (train_scores, validation_scores) in methods.items():
        operating_point = select_threshold(
            train_scores,
            train,
            maximum_harmful_accept_rate=maximum_harmful_accept_rate,
            maximum_false_edit_rate=maximum_false_edit_rate,
            threshold_count=threshold_count,
        )
        result_methods[name] = {
            "train_auroc_beneficial_vs_rest": _auroc(train_scores, train["beneficial"]),
            "validation_auroc_beneficial_vs_rest": _auroc(
                validation_scores, validation["beneficial"]
            ),
            "train_operating_point": operating_point,
            "validation_at_train_threshold": score_gate(
                validation_scores, validation, operating_point["threshold"]
            ),
        }
    return {
        "schema_version": "sn7-delete-operation-risk-gate-screen-v1",
        "seed": seed,
        "split": {"fit": "train", "evaluation": "val"},
        "test_assets_read": False,
        "feature_names": list(FEATURE_NAMES),
        "forbidden_features": [
            "target",
            "raster_iou",
            "raster_iou_gain",
            "map_quality_after",
            "false_edit",
            "missed_edit",
            "wrong_edit",
            "target_component_count",
            "component_count_absolute_error",
            "added_change_iou",
            "removed_change_iou",
            "added_polygon_iou",
            "removed_polygon_iou",
        ],
        "support": {
            "train_rows": int(len(train["beneficial"])),
            "validation_rows": int(len(validation["beneficial"])),
            "train_beneficial_delete_count": int(train["beneficial"].sum()),
            "validation_beneficial_delete_count": int(validation["beneficial"].sum()),
            "train_aoi_count": int(len(set(train["aoi_ids"]))),
            "validation_aoi_count": int(len(set(validation["aoi_ids"]))),
        },
        "constraints": {
            "gain_epsilon": gain_epsilon,
            "maximum_harmful_accept_rate": maximum_harmful_accept_rate,
            "maximum_false_edit_rate": maximum_false_edit_rate,
            "threshold_count": threshold_count,
        },
        "model": {
            "family": "HistGradientBoostingClassifier",
            "learning_rate": 0.05,
            "max_iter": max_iter,
            "max_leaf_nodes": max_leaf_nodes,
            "class_balance": "inverse_prevalence_sample_weight",
        },
        "methods": result_methods,
        "risk_minus_confidence_validation": {
            name: (
                result_methods["risk_gate"]["validation_at_train_threshold"][name]
                - result_methods["confidence"]["validation_at_train_threshold"][name]
            )
            for name in result_methods["risk_gate"]["validation_at_train_threshold"]
            if name != "threshold"
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-record", action="append", type=parse_record, required=True)
    parser.add_argument("--validation-record", action="append", type=parse_record, required=True)
    parser.add_argument("--gain-epsilon", type=float, default=1e-6)
    parser.add_argument("--maximum-harmful-accept-rate", type=float, default=0.01)
    parser.add_argument("--maximum-false-edit-rate", type=float, default=0.01)
    parser.add_argument("--threshold-count", type=int, default=101)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.gain_epsilon < 0.0 or args.threshold_count < 2:
        raise ValueError("invalid screen configuration")
    if not 0.0 <= args.maximum_harmful_accept_rate <= 1.0:
        raise ValueError("harmful-acceptance cap must be in [0, 1]")
    if not 0.0 <= args.maximum_false_edit_rate <= 1.0:
        raise ValueError("false-edit cap must be in [0, 1]")
    result = screen(
        args.train_record,
        args.validation_record,
        seed=args.seed,
        gain_epsilon=args.gain_epsilon,
        maximum_harmful_accept_rate=args.maximum_harmful_accept_rate,
        maximum_false_edit_rate=args.maximum_false_edit_rate,
        threshold_count=args.threshold_count,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
    )
    inputs = {}
    for scope, records in (("train", args.train_record), ("validation", args.validation_record)):
        inputs[scope] = {
            policy: {"path": str(path.resolve()), "sha256": sha256(path)}
            for policy, path in records
        }
    result["inputs"] = inputs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
