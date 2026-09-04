#!/usr/bin/env python3
"""Reload and independently evaluate a frozen Tool-Belief stopping policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.analyze_tool_belief_stage_stopping import analyze_stage_choices
from scripts.train_tool_belief_stopping_policy import (
    POLICY_FEATURE_DIM,
    StoppingPolicy,
    _infer,
    _safety_passes,
    build_policy_data,
    rollout_policy,
)


def evaluate_policy(
    checkpoint: Path,
    stage_checkpoint: Path,
    details: Path,
    selector_states: Path,
    *,
    device: str = "cpu",
    minimum_utility_gain: float = 1e-6,
    safety_margin: float = 0.02,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload.get("protocol") != "tool_belief_stopping_policy_v2":
        raise ValueError("checkpoint is not a Tool-Belief stopping policy")
    if payload.get("feature_dim") != POLICY_FEATURE_DIM:
        raise ValueError("stopping-policy feature dimension does not match runtime")
    threshold = payload.get("decision_threshold")
    if threshold is None:
        raise ValueError("checkpoint has no frozen train-calibrated threshold")
    data = build_policy_data(
        stage_checkpoint, details, selector_states, split="val", device="cpu"
    )
    resolved_device = torch.device(device)
    model = StoppingPolicy(
        hidden_dim=int(payload["hidden_dim"]), dropout=float(payload["dropout"])
    ).to(resolved_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    probabilities = _infer(model, data, device=resolved_device)
    learned, selected_stages = rollout_policy(
        data, probabilities, float(threshold), name="learned_stopping"
    )
    fixed = {}
    for stage in range(4):
        forced = np.zeros_like(probabilities)
        if stage:
            forced[:, :stage] = 1.0
        fixed[f"stage_{stage}"], _ = rollout_policy(
            data, forced, 0.5, name=f"fixed_stage_{stage}"
        )
    oracle = analyze_stage_choices(
        data.sequence_ids,
        np.repeat(data.targets, 4),
        data.predictions.reshape(-1),
        data.confidences.reshape(-1),
        data.costs.reshape(-1),
    )
    utility_gain = learned["mean_joint_utility"] - fixed["stage_0"]["mean_joint_utility"]
    checks = {
        "positive_utility_gain": utility_gain >= minimum_utility_gain,
        "false_and_missed_edit_safety": _safety_passes(
            learned, fixed["stage_0"], safety_margin
        ),
        "not_above_oracle": (
            learned["mean_joint_utility"]
            <= oracle["oracle"]["mean_joint_utility"] + 1e-12
        ),
    }
    report = {
        "protocol": {
            "schema_version": "tool-belief-stopping-policy-eval-v1",
            "split": "val",
            "sequence_count": len(data.sequence_ids),
            "checkpoint_epoch": int(payload["epoch"]),
            "frozen_train_threshold": float(threshold),
            "test_assets_read": False,
        },
        "fixed_stages": fixed,
        "learned": learned,
        "validation_oracle": oracle["oracle"],
        "oracle_selected_stage_counts": oracle["oracle_selected_stage_counts"],
        "utility_gain_over_stage_0": utility_gain,
        "gates": {
            "thresholds": {
                "minimum_utility_gain": minimum_utility_gain,
                "safety_margin": safety_margin,
            },
            "checks": checks,
            "passed": all(checks.values()),
        },
    }
    rows = [
        {
            "sequence_id": sequence_id,
            "selected_stage": int(selected_stages[index]),
            "continue_probabilities": probabilities[index].tolist(),
            "target": int(data.targets[index]),
            "prediction": int(data.predictions[index, selected_stages[index]]),
            "stage_predictions": data.predictions[index].tolist(),
            "stage_confidences": data.confidences[index].tolist(),
            "cost": float(data.costs[index, selected_stages[index]]),
        }
        for index, sequence_id in enumerate(data.sequence_ids)
    ]
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("stage_checkpoint", type=Path)
    parser.add_argument("val_details", type=Path)
    parser.add_argument("selector_states", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimum-utility-gain", type=float, default=1e-6)
    parser.add_argument("--safety-margin", type=float, default=0.02)
    args = parser.parse_args()
    report, rows = evaluate_policy(
        args.checkpoint,
        args.stage_checkpoint,
        args.val_details,
        args.selector_states,
        device=args.device,
        minimum_utility_gain=args.minimum_utility_gain,
        safety_margin=args.safety_margin,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "details.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
