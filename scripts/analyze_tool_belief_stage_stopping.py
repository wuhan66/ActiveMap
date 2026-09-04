#!/usr/bin/env python3
"""Measure stopping opportunity for a causal stage-aware Tool-Belief head."""

from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.analyze_tool_belief_stopping import EDIT_ORDER, _reward, _summary
from scripts.train_tool_belief_decision_head import (
    FEATURE_DIM,
    KEEP_INDEX,
    DecisionTrajectoryDataset,
    HierarchicalDecisionHead,
    _feature_names,
    _read_jsonl,
)


def _costs(path: Path, split: str) -> dict[str, list[float]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in _read_jsonl(path):
        if row.get("split", split) != split:
            raise ValueError(f"trajectory row is not from {split}")
        grouped.setdefault(str(row["sequence_id"]), []).append(row)
    result = {}
    for sequence_id, rows in grouped.items():
        rows.sort(key=lambda item: int(item["step"]))
        if [int(item["step"]) for item in rows] != [1, 2, 3]:
            raise ValueError(f"{sequence_id} does not contain steps 1, 2, 3")
        values = [0.0]
        for row in rows:
            values.append(float(row.get("spent_cost", 0.18 * int(row["step"]))))
        if any(
            right <= left
            for left, right in zip(values[:-1], values[1:], strict=True)
        ):
            raise ValueError(f"{sequence_id} costs must increase at every stage")
        result[sequence_id] = values
    return result


def _predict(
    checkpoint: Path,
    dataset: DecisionTrajectoryDataset,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload.get("protocol") != "hierarchical_decision_head_prefix_v2":
        raise ValueError("checkpoint is not a per-stage calibrated prefix head")
    if (
        payload.get("feature_dim") != FEATURE_DIM
        or payload.get("feature_names") != _feature_names()
    ):
        raise ValueError("checkpoint feature contract does not match runtime")
    thresholds = payload.get("decision_thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != {"0", "1", "2", "3"}:
        raise ValueError("checkpoint has no complete frozen per-stage thresholds")
    resolved_device = torch.device(device)
    model = HierarchicalDecisionHead(
        hidden_dim=int(payload["hidden_dim"]), dropout=float(payload["dropout"])
    ).to(resolved_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    prediction = np.empty(len(dataset), dtype=np.int64)
    confidence = np.empty(len(dataset), dtype=np.float64)
    with torch.inference_mode():
        for stage in range(4):
            selected = dataset.stages == stage
            update_logit, operation_logit = model(dataset.features[selected].to(resolved_device))
            update_probability = torch.sigmoid(update_logit)
            operation = torch.argmax(operation_logit, dim=1) + 1
            stage_prediction = torch.where(
                update_probability >= float(thresholds[str(stage)]),
                operation,
                torch.full_like(operation, KEEP_INDEX),
            )
            stage_confidence = torch.where(
                stage_prediction == KEEP_INDEX,
                1.0 - update_probability,
                update_probability,
            )
            prediction[selected.numpy()] = stage_prediction.cpu().numpy()
            confidence[selected.numpy()] = stage_confidence.cpu().numpy()
    return prediction, confidence


def analyze_stage_choices(
    sequence_ids: list[str],
    targets: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    costs: np.ndarray,
) -> dict[str, Any]:
    sequence_count = len(sequence_ids)
    expected = sequence_count * 4
    if any(len(values) != expected for values in (targets, predictions, confidences, costs)):
        raise ValueError("stage arrays must contain exactly four states per sequence")
    fixed = {}
    for stage in range(4):
        indices = np.arange(stage, expected, 4)
        fixed[f"stage_{stage}"] = _summary(
            f"fixed_stage_{stage}",
            targets[indices].tolist(),
            predictions[indices].tolist(),
            confidences[indices].tolist(),
            costs[indices].tolist(),
        )

    oracle_targets = []
    oracle_predictions = []
    oracle_confidences = []
    oracle_costs = []
    selected_counts: Counter[int] = Counter()
    rows = []
    for index, sequence_id in enumerate(sequence_ids):
        start = index * 4
        choices = []
        for stage in range(4):
            offset = start + stage
            utility = _reward(int(targets[offset]), int(predictions[offset])) - float(costs[offset])
            choices.append((utility, -float(costs[offset]), stage, offset))
        _, _, stage, selected = max(choices)
        oracle_targets.append(int(targets[selected]))
        oracle_predictions.append(int(predictions[selected]))
        oracle_confidences.append(float(confidences[selected]))
        oracle_costs.append(float(costs[selected]))
        selected_counts[stage] += 1
        rows.append(
            {
                "sequence_id": sequence_id,
                "selected_stage": stage,
                "target": EDIT_ORDER[int(targets[selected])],
                "prediction": EDIT_ORDER[int(predictions[selected])],
                "cost": float(costs[selected]),
            }
        )
    oracle = _summary(
        "validation_stage_stopping_oracle",
        oracle_targets,
        oracle_predictions,
        oracle_confidences,
        oracle_costs,
    )
    return {
        "fixed_stages": fixed,
        "oracle": oracle,
        "oracle_selected_stage_counts": {
            str(stage): selected_counts[stage] for stage in range(4)
        },
        "oracle_utility_gain_over_stage_0": (
            oracle["mean_joint_utility"] - fixed["stage_0"]["mean_joint_utility"]
        ),
        "oracle_rows": rows,
    }


def analyze_checkpoint(
    checkpoint: Path,
    details: Path,
    *,
    split: str,
    device: str = "cpu",
) -> dict[str, Any]:
    dataset = DecisionTrajectoryDataset(details, split, stages=(0, 1, 2, 3))
    predictions, confidences = _predict(checkpoint, dataset, device=device)
    cost_by_sequence = _costs(details, split)
    sequence_ids = list(cost_by_sequence)
    if [example["sequence_id"] for example in dataset.examples[::4]] != sequence_ids:
        raise ValueError("trajectory and cost sequence ordering differs")
    costs = np.asarray(
        [cost for sequence_id in sequence_ids for cost in cost_by_sequence[sequence_id]],
        dtype=np.float64,
    )
    result = analyze_stage_choices(
        sequence_ids,
        dataset.target.numpy(),
        predictions,
        confidences,
        costs,
    )
    return {
        "protocol": {
            "schema_version": "tool-belief-stage-stopping-opportunity-v1",
            "split": split,
            "sequence_count": len(sequence_ids),
            "oracle_is_upper_bound_not_deployable": True,
            "test_assets_read": False,
        },
        **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("details", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = analyze_checkpoint(
        args.checkpoint, args.details, split=args.split, device=args.device
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    printable = {key: value for key, value in report.items() if key != "oracle_rows"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
