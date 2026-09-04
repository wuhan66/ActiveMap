#!/usr/bin/env python3
"""Factor selector failure into candidate ranking and terminal gating."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

EDIT_ORDER = ("KEEP", "ADD", "DELETE", "RESHAPE")
MASK_FEATURES_SCHEMA = "target-free-mask-features-v2"
MASK_FEATURE_DIM = 10


def normalized_entropy(probabilities: np.ndarray) -> float:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, None)
    values /= values.sum()
    return float(-np.sum(values * np.log(values)) / np.log(len(values)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_features(row: dict[str, Any], feature_set: str) -> np.ndarray:
    evidence = np.asarray(row["evidence_features"], dtype=np.float32)
    if feature_set in ("evidence-value", "policy-relative", "policy-relative-mask"):
        metadata = row.get("metadata", {})
        predictions = metadata.get("evidence_predictions")
        if not isinstance(predictions, dict):
            raise ValueError("evidence-value features require candidate predictions")
        if feature_set == "policy-relative-mask":
            contract = metadata.get("mask_feature_contract")
            if (
                not isinstance(contract, dict)
                or contract.get("schema_version") != MASK_FEATURES_SCHEMA
            ):
                raise ValueError("policy-relative-mask features require the v2 mask contract")
            if contract.get("target_free") is not True:
                raise ValueError("mask feature contract must declare target-free provenance")
        hypothesis = np.asarray(row["hypothesis_features"], dtype=np.float32)
        prior_probabilities = hypothesis[:4]
        prior_entropy = normalized_entropy(prior_probabilities)
        prior_confidence = float(hypothesis[13])
        initial_operation = str(row["edit_type"])
        if initial_operation not in EDIT_ORDER:
            raise ValueError(f"unsupported initial operation: {initial_operation}")
        enriched = []
        for index, evidence_id in enumerate(row["evidence_ids"]):
            prediction = predictions.get(evidence_id)
            if not isinstance(prediction, dict):
                raise ValueError(f"missing candidate prediction: {evidence_id}")
            probabilities = list(prediction["edit_probabilities"])
            geometry_delta = list(prediction["geometry_delta"])
            if len(probabilities) != 4 or len(geometry_delta) != 8:
                raise ValueError("candidate prediction has invalid feature dimensions")
            features = (
                list(evidence[index])
                + [
                    float(row["evidence_costs"][index]),
                    float(row["false_edit_risks"][index]),
                ]
                + probabilities
                + [float(prediction["confidence"])]
                + geometry_delta
            )
            if feature_set in ("policy-relative", "policy-relative-mask"):
                candidate_probabilities = np.asarray(probabilities, dtype=np.float32)
                probability_delta = candidate_probabilities - prior_probabilities
                ordered = np.sort(candidate_probabilities)
                gated_operation = str(
                    prediction.get(
                        "gated_edit",
                        EDIT_ORDER[int(np.argmax(candidate_probabilities))],
                    )
                )
                if gated_operation not in EDIT_ORDER:
                    raise ValueError(
                        f"unsupported candidate operation: {gated_operation}"
                    )
                gated_one_hot = [
                    float(operation == gated_operation) for operation in EDIT_ORDER
                ]
                candidate_entropy = normalized_entropy(candidate_probabilities)
                features += (
                    gated_one_hot
                    + probability_delta.tolist()
                    + np.abs(probability_delta).tolist()
                    + [
                        candidate_entropy,
                        candidate_entropy - prior_entropy,
                        float(ordered[-1] - ordered[-2]),
                        float(prediction["confidence"]) - prior_confidence,
                        float(gated_operation == initial_operation),
                        float(np.abs(probability_delta).sum()),
                    ]
                )
            if feature_set == "policy-relative-mask":
                mask_features = list(prediction.get("mask_features", []))
                if len(mask_features) != MASK_FEATURE_DIM:
                    raise ValueError("candidate prediction has invalid mask feature dimensions")
                features += [float(value) for value in mask_features]
            enriched.append(features)
        evidence = np.asarray(enriched, dtype=np.float32)
    elif feature_set != "basic":
        raise ValueError(f"unsupported feature set: {feature_set}")
    hypothesis = np.asarray(row["hypothesis_features"], dtype=np.float32)
    state = np.asarray(row["state_features"], dtype=np.float32)
    context = np.concatenate([hypothesis, state])
    return np.concatenate(
        [evidence, np.repeat(context[None, :], len(evidence), axis=0)], axis=1
    )


def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(
        row.get("split") == "test"
        or row.get("metadata", {}).get("test_assets_read") is True
        for row in rows
    ):
        raise ValueError("candidate learnability audit rejects test provenance")
    train = [row for row in rows if row.get("split") == "train"]
    val = [row for row in rows if row.get("split") == "val"]
    if not train or not val or len(train) + len(val) != len(rows):
        raise ValueError("manifest must contain only non-empty train and val splits")
    return train, val


def flatten(
    rows: list[dict[str, Any]], feature_set: str
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    deltas = []
    for row in rows:
        utility = np.asarray(row["oracle_utilities"], dtype=np.float32)
        features.append(candidate_features(row, feature_set))
        deltas.append(utility - float(row["stop_utility"]))
    return np.concatenate(features), np.concatenate(deltas)


def state_observations(
    model: Any,
    rows: list[dict[str, Any]],
    feature_set: str,
    model_type: str,
) -> dict[str, np.ndarray]:
    values: dict[str, list[Any]] = {
        "score": [],
        "chosen_delta": [],
        "oracle_delta": [],
        "target_acquire": [],
        "exact": [],
        "top3_positive": [],
        "top5_positive": [],
    }
    for row in rows:
        features = candidate_features(row, feature_set)
        scores = (
            model.predict_proba(features)[:, 1]
            if model_type == "classifier"
            else model.predict(features)
        )
        deltas = np.asarray(row["oracle_utilities"], dtype=np.float64) - float(
            row["stop_utility"]
        )
        order = np.argsort(scores)[::-1]
        chosen = int(order[0])
        oracle = int(np.argmax(deltas))
        values["score"].append(float(scores[chosen]))
        values["chosen_delta"].append(float(deltas[chosen]))
        values["oracle_delta"].append(float(deltas[oracle]))
        values["target_acquire"].append(bool(deltas[oracle] > 0.0))
        values["exact"].append(chosen == oracle)
        values["top3_positive"].append(bool(np.any(deltas[order[:3]] > 0.0)))
        values["top5_positive"].append(bool(np.any(deltas[order[:5]] > 0.0)))
    return {name: np.asarray(items) for name, items in values.items()}


def point(
    observations: dict[str, np.ndarray], threshold: float, *, oracle_rank: bool
) -> dict[str, float]:
    acquire = observations["score"] > threshold
    target = observations["target_acquire"].astype(bool)
    delta = (
        observations["oracle_delta"]
        if oracle_rank
        else observations["chosen_delta"]
    )
    calls = int(acquire.sum())
    return {
        "threshold": float(threshold),
        "acquire_rate": float(np.mean(acquire)),
        "false_call_rate": float(np.mean(acquire & ~target)),
        "harmful_call_fraction": float(np.sum(acquire & (delta < 0.0)) / max(calls, 1)),
        "acquire_recall": float(np.sum(acquire & target) / max(int(target.sum()), 1)),
        "exact_acquire_rate": float(
            np.mean(acquire & target & observations["exact"].astype(bool))
        ),
        "mean_chosen_utility": float(np.mean(np.where(acquire, delta, 0.0))),
    }


def frontier(
    observations: dict[str, np.ndarray],
    *,
    oracle_rank: bool,
    max_false_call_rate: float,
    max_harmful_call_fraction: float,
    min_acquire_recall: float,
) -> dict[str, Any]:
    scores = observations["score"].astype(np.float64)
    thresholds = np.unique(
        np.concatenate(
            [
                np.quantile(scores, np.linspace(0.0, 1.0, 201)),
                [np.nextafter(scores.min(), -np.inf), np.nextafter(scores.max(), np.inf)],
            ]
        )
    )
    points = [point(observations, value, oracle_rank=oracle_rank) for value in thresholds]
    feasible = [
        value
        for value in points
        if value["false_call_rate"] <= max_false_call_rate
        and value["harmful_call_fraction"] <= max_harmful_call_fraction
        and value["acquire_recall"] >= min_acquire_recall
    ]
    selected = (
        max(feasible, key=lambda value: (value["mean_chosen_utility"], -value["acquire_rate"]))
        if feasible
        else None
    )
    return {"feasible_point_count": len(feasible), "selected_feasible": selected}


def ranking_summary(observations: dict[str, np.ndarray]) -> dict[str, float]:
    target = observations["target_acquire"].astype(bool)
    learned_oracle_gate = np.where(target, observations["chosen_delta"], 0.0)
    return {
        "target_acquire_rate": float(np.mean(target)),
        "exact_best_recall": float(np.mean(observations["exact"][target])),
        "top1_positive_recall": float(np.mean(observations["chosen_delta"][target] > 0.0)),
        "top3_positive_recall": float(np.mean(observations["top3_positive"][target])),
        "top5_positive_recall": float(np.mean(observations["top5_positive"][target])),
        "oracle_gate_harmful_call_fraction": float(
            np.mean(observations["chosen_delta"][target] < 0.0)
        ),
        "oracle_gate_mean_chosen_utility": float(np.mean(learned_oracle_gate)),
    }


def main() -> None:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--positive-weight", type=float, default=32.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument(
        "--feature-set",
        choices=("basic", "evidence-value", "policy-relative", "policy-relative-mask"),
        default="basic",
    )
    parser.add_argument(
        "--model-type", choices=("regressor", "classifier"), default="regressor"
    )
    parser.add_argument("--max-false-call-rate", type=float, default=0.02)
    parser.add_argument("--max-harmful-call-fraction", type=float, default=0.20)
    parser.add_argument("--min-acquire-recall", type=float, default=0.10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.positive_weight <= 0 or args.max_iter <= 0:
        raise ValueError("positive weight and max iterations must be positive")

    train_rows, val_rows = load_rows(args.samples)
    train_x, train_y = flatten(train_rows, args.feature_set)
    positive = train_y > 0.0
    weights = np.where(positive, args.positive_weight, 1.0)
    model_class = (
        HistGradientBoostingClassifier
        if args.model_type == "classifier"
        else HistGradientBoostingRegressor
    )
    model = model_class(
        learning_rate=0.08,
        max_iter=args.max_iter,
        max_leaf_nodes=31,
        l2_regularization=1e-3,
        random_state=20260831,
    ).fit(
        train_x,
        positive.astype(np.int64) if args.model_type == "classifier" else train_y,
        sample_weight=weights,
    )
    observations = state_observations(
        model, val_rows, args.feature_set, args.model_type
    )
    constraints = {
        "max_false_call_rate": args.max_false_call_rate,
        "max_harmful_call_fraction": args.max_harmful_call_fraction,
        "min_acquire_recall": args.min_acquire_recall,
    }
    result = {
        "schema_version": "selector-candidate-learnability-audit-v1",
        "samples": str(args.samples.resolve()),
        "samples_sha256": sha256(args.samples),
        "test_assets_read": False,
        "train_states": len(train_rows),
        "validation_states": len(val_rows),
        "train_candidates": int(len(train_y)),
        "train_positive_candidate_rate": float(np.mean(positive)),
        "feature_set": args.feature_set,
        "model_type": args.model_type,
        "feature_dimension": int(train_x.shape[1]),
        "positive_weight": args.positive_weight,
        "ranking": ranking_summary(observations),
        "full_learned_frontier": frontier(
            observations, oracle_rank=False, **constraints
        ),
        "learned_gate_oracle_rank_frontier": frontier(
            observations, oracle_rank=True, **constraints
        ),
        "constraints": constraints,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
