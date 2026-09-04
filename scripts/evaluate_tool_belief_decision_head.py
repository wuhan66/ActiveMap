#!/usr/bin/env python3
"""Reload and independently evaluate the frozen hierarchical decision head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.analyze_tool_belief_stopping import EDIT_ORDER, _summary
from scripts.train_tool_belief_decision_head import (
    FEATURE_DIM,
    KEEP_INDEX,
    DecisionTrajectoryDataset,
    HierarchicalDecisionHead,
    _feature_names,
)


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


def _predict(
    model: HierarchicalDecisionHead,
    features: torch.Tensor,
    *,
    device: torch.device,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return (
        prediction.cpu().numpy(),
        confidence.cpu().numpy(),
        update_probability.cpu().numpy(),
    )


def evaluate_checkpoint(
    checkpoint: Path,
    val_details: Path,
    *,
    device: str = "cpu",
    minimum_macro_f1_gain: float = 0.01,
    max_false_edit_increase: float = 0.02,
    max_missed_edit_increase: float = 0.02,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload.get("protocol") != "hierarchical_decision_head_v1":
        raise ValueError("checkpoint is not a hierarchical decision head")
    if payload.get("feature_dim") != FEATURE_DIM:
        raise ValueError("checkpoint feature dimension does not match runtime")
    if payload.get("feature_names") != _feature_names():
        raise ValueError("checkpoint feature names do not match runtime")
    threshold = payload.get("decision_threshold")
    if threshold is None:
        raise ValueError("checkpoint has no frozen decision threshold")
    resolved_device = torch.device(device)
    model = HierarchicalDecisionHead(
        hidden_dim=int(payload["hidden_dim"]), dropout=float(payload["dropout"])
    ).to(resolved_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    paired = DecisionTrajectoryDataset(val_details, "val", "paired")
    no_quality = DecisionTrajectoryDataset(val_details, "val", "no_quality")
    target = paired.target.numpy()
    baseline = paired.baseline.numpy()
    residual = paired.residual.numpy()
    baseline_confidence = np.max(paired.features[:, :4].numpy(), axis=1)
    residual_confidence = np.max(paired.features[:, -5:-1].numpy(), axis=1)
    prediction, confidence, update_probability = _predict(
        model, paired.features, device=resolved_device, threshold=float(threshold)
    )
    no_quality_prediction, no_quality_confidence, _ = _predict(
        model, no_quality.features, device=resolved_device, threshold=float(threshold)
    )
    no_anchor_features = paired.features.clone()
    no_anchor_features[:, 4:9] = 0.0
    no_anchor_prediction, no_anchor_confidence, _ = _predict(
        model, no_anchor_features, device=resolved_device, threshold=float(threshold)
    )
    no_current_features = paired.features.clone()
    no_current_features[:, :4] = 0.25
    no_current_features[:, 4:9] = 0.0
    no_current_prediction, no_current_confidence, _ = _predict(
        model, no_current_features, device=resolved_device, threshold=float(threshold)
    )
    summaries = {
        "identity": _metrics("identity", target, baseline, baseline_confidence),
        "residual_argmax": _metrics(
            "residual_argmax", target, residual, residual_confidence
        ),
        "hierarchical": _metrics("hierarchical", target, prediction, confidence),
        "no_quality": _metrics(
            "no_quality", target, no_quality_prediction, no_quality_confidence
        ),
        "no_anchor": _metrics(
            "no_anchor", target, no_anchor_prediction, no_anchor_confidence
        ),
        "no_current": _metrics(
            "no_current", target, no_current_prediction, no_current_confidence
        ),
    }
    identity = summaries["identity"]
    learned = summaries["hierarchical"]
    checks = {
        "macro_f1_gain": (
            learned["macro_f1"] - identity["macro_f1"] >= minimum_macro_f1_gain
        ),
        "false_edit_safety": (
            learned["false_edit_rate"] - identity["false_edit_rate"]
            <= max_false_edit_increase
        ),
        "missed_edit_safety": (
            learned["missed_edit_rate"] - identity["missed_edit_rate"]
            <= max_missed_edit_increase
        ),
    }
    report = {
        "protocol": {
            "schema_version": "tool-belief-decision-head-eval-v1",
            "split": "val",
            "sequence_count": len(paired),
            "checkpoint_epoch": int(payload["epoch"]),
            "frozen_decision_threshold": float(threshold),
            "test_assets_read": False,
        },
        "summaries": summaries,
        "ablation_deltas": {
            name: metrics["macro_f1"] - learned["macro_f1"]
            for name, metrics in summaries.items()
            if name in {"no_quality", "no_anchor", "no_current"}
        },
        "gates": {
            "thresholds": {
                "minimum_macro_f1_gain": minimum_macro_f1_gain,
                "max_false_edit_increase": max_false_edit_increase,
                "max_missed_edit_increase": max_missed_edit_increase,
            },
            "checks": checks,
            "passed": all(checks.values()),
        },
    }
    details = [
        {
            "sequence_id": example["sequence_id"],
            "target": EDIT_ORDER[int(target[index])],
            "identity": EDIT_ORDER[int(baseline[index])],
            "residual_argmax": EDIT_ORDER[int(residual[index])],
            "hierarchical": EDIT_ORDER[int(prediction[index])],
            "update_probability": float(update_probability[index]),
        }
        for index, example in enumerate(paired.examples)
    ]
    return report, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("val_details", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimum-macro-f1-gain", type=float, default=0.01)
    parser.add_argument("--max-false-edit-increase", type=float, default=0.02)
    parser.add_argument("--max-missed-edit-increase", type=float, default=0.02)
    args = parser.parse_args()
    report, details = evaluate_checkpoint(
        args.checkpoint,
        args.val_details,
        device=args.device,
        minimum_macro_f1_gain=args.minimum_macro_f1_gain,
        max_false_edit_increase=args.max_false_edit_increase,
        max_missed_edit_increase=args.max_missed_edit_increase,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "details.jsonl").open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
