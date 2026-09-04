#!/usr/bin/env python3
"""Independently evaluate a causal prefix-trained hierarchical decision head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.analyze_tool_belief_stopping import _summary
from scripts.train_tool_belief_decision_head import (
    FEATURE_DIM,
    KEEP_INDEX,
    DecisionTrajectoryDataset,
    HierarchicalDecisionHead,
    _feature_names,
)


def _predict(
    model: HierarchicalDecisionHead,
    features: torch.Tensor,
    *,
    threshold: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        update_logit, operation_logit = model(features.to(device))
        update_probability = torch.sigmoid(update_logit)
        operation = torch.argmax(operation_logit, dim=1) + 1
        prediction = torch.where(
            update_probability >= threshold,
            operation,
            torch.full_like(operation, KEEP_INDEX),
        )
        confidence = torch.where(
            prediction == KEEP_INDEX, 1.0 - update_probability, update_probability
        )
    return prediction.cpu().numpy(), confidence.cpu().numpy()


def _metrics(
    name: str,
    target: np.ndarray,
    prediction: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, Any]:
    return _summary(
        name,
        target.tolist(),
        prediction.tolist(),
        confidence.tolist(),
        [0.0] * len(target),
    )


def evaluate_stage_checkpoint(
    checkpoint: Path,
    details: Path,
    *,
    device: str = "cpu",
    minimum_final_macro_f1_gain: float = 0.01,
    max_false_edit_increase: float = 0.02,
    max_missed_edit_increase: float = 0.02,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload.get("protocol") != "hierarchical_decision_head_prefix_v2":
        raise ValueError("checkpoint is not a causal prefix decision head")
    if payload.get("feature_dim") != FEATURE_DIM:
        raise ValueError("checkpoint feature dimension does not match runtime")
    if payload.get("feature_names") != _feature_names():
        raise ValueError("checkpoint feature names do not match runtime")
    stages = tuple(int(stage) for stage in payload.get("prefix_stages", []))
    if stages != (0, 1, 2, 3):
        raise ValueError("prefix checkpoint must contain stages 0,1,2,3")
    thresholds = payload.get("decision_thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != {"0", "1", "2", "3"}:
        raise ValueError("checkpoint has no complete frozen per-stage thresholds")

    resolved_device = torch.device(device)
    dataset = DecisionTrajectoryDataset(details, "val", stages=stages)
    model = HierarchicalDecisionHead(
        hidden_dim=int(payload["hidden_dim"]), dropout=float(payload["dropout"])
    ).to(resolved_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    target = dataset.target.numpy()
    baseline = dataset.baseline.numpy()
    residual = dataset.residual.numpy()
    stage_values = dataset.stages.numpy()
    prediction = np.empty_like(target)
    confidence = np.empty(len(target), dtype=np.float64)
    for stage in stages:
        selected = stage_values == stage
        stage_prediction, stage_confidence = _predict(
            model,
            dataset.features[selected],
            threshold=float(thresholds[str(stage)]),
            device=resolved_device,
        )
        prediction[selected] = stage_prediction
        confidence[selected] = stage_confidence
    summaries: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for stage in stages:
        selected = stage_values == stage
        stage_target = target[selected]
        identity = _metrics(
            f"identity_stage_{stage}",
            stage_target,
            baseline[selected],
            np.max(dataset.features[selected, :4].numpy(), axis=1),
        )
        learned = _metrics(
            f"hierarchical_stage_{stage}",
            stage_target,
            prediction[selected],
            confidence[selected],
        )
        residual_metrics = _metrics(
            f"residual_stage_{stage}",
            stage_target,
            residual[selected],
            np.full(int(np.sum(selected)), 0.5),
        )
        summaries[f"stage_{stage}"] = {
            "identity": identity,
            "residual_argmax": residual_metrics,
            "hierarchical": learned,
            "hierarchical_minus_identity_macro_f1": (
                learned["macro_f1"] - identity["macro_f1"]
            ),
        }
        checks[f"stage_{stage}_false_edit_safety"] = (
            learned["false_edit_rate"] - identity["false_edit_rate"]
            <= max_false_edit_increase
        )
        checks[f"stage_{stage}_missed_edit_safety"] = (
            learned["missed_edit_rate"] - identity["missed_edit_rate"]
            <= max_missed_edit_increase
        )

    final = summaries["stage_3"]
    checks["final_macro_f1_gain"] = (
        final["hierarchical_minus_identity_macro_f1"]
        >= minimum_final_macro_f1_gain
    )
    causal_zero_checks = {
        f"stage_{stage}": bool(
            torch.count_nonzero(dataset.features[dataset.stages == stage, 9 + 5 * stage :])
            == 0
        )
        for stage in (0, 1, 2)
    }
    checks["causal_future_blocks_zero"] = all(causal_zero_checks.values())
    return {
        "protocol": {
            "schema_version": "tool-belief-stage-head-eval-v1",
            "split": "val",
            "sequence_count": len(dataset) // len(stages),
            "state_count": len(dataset),
            "checkpoint_epoch": int(payload["epoch"]),
            "frozen_decision_thresholds": {
                stage: float(value) for stage, value in thresholds.items()
            },
            "test_assets_read": False,
        },
        "stage_summaries": summaries,
        "causal_future_zero_checks": causal_zero_checks,
        "gates": {
            "thresholds": {
                "minimum_final_macro_f1_gain": minimum_final_macro_f1_gain,
                "max_false_edit_increase": max_false_edit_increase,
                "max_missed_edit_increase": max_missed_edit_increase,
            },
            "checks": checks,
            "passed": all(checks.values()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("val_details", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimum-final-macro-f1-gain", type=float, default=0.01)
    parser.add_argument("--max-false-edit-increase", type=float, default=0.02)
    parser.add_argument("--max-missed-edit-increase", type=float, default=0.02)
    args = parser.parse_args()
    report = evaluate_stage_checkpoint(
        args.checkpoint,
        args.val_details,
        device=args.device,
        minimum_final_macro_f1_gain=args.minimum_final_macro_f1_gain,
        max_false_edit_increase=args.max_false_edit_increase,
        max_missed_edit_increase=args.max_missed_edit_increase,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
