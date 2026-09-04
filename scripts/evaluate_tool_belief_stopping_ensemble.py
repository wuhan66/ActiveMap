#!/usr/bin/env python3
"""Evaluate a fixed ensemble of train-calibrated Tool-Belief stopping policies."""

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


def evaluate_ensemble(
    checkpoints: list[Path],
    stage_checkpoint: Path,
    details: Path,
    selector_states: Path,
    *,
    device: str = "cpu",
    minimum_utility_gain: float = 1e-6,
    safety_margin: float = 0.02,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(checkpoints) < 2:
        raise ValueError("stopping ensemble requires at least two fixed members")
    data = build_policy_data(
        stage_checkpoint, details, selector_states, split="val", device="cpu"
    )
    resolved_device = torch.device(device)
    member_margins = []
    member_protocol = []
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if payload.get("protocol") != "tool_belief_stopping_policy_v2":
            raise ValueError(f"{checkpoint} is not a stopping-policy v2 checkpoint")
        if payload.get("feature_dim") != POLICY_FEATURE_DIM:
            raise ValueError(f"{checkpoint} feature dimension does not match runtime")
        threshold = payload.get("decision_threshold")
        if threshold is None:
            raise ValueError(f"{checkpoint} has no frozen train threshold")
        model = StoppingPolicy(
            hidden_dim=int(payload["hidden_dim"]), dropout=float(payload["dropout"])
        ).to(resolved_device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        probability = _infer(model, data, device=resolved_device)
        member_margins.append(probability - float(threshold))
        member_protocol.append(
            {
                "checkpoint": str(checkpoint),
                "epoch": int(payload["epoch"]),
                "train_threshold": float(threshold),
            }
        )
    margins = np.stack(member_margins)
    mean_margin = np.mean(margins, axis=0)
    learned, selected_stages = rollout_policy(
        data, mean_margin, 0.0, name="fixed_margin_ensemble"
    )
    stage0, _ = rollout_policy(
        data, np.zeros_like(mean_margin), 1.1, name="stage_0_head"
    )
    oracle = analyze_stage_choices(
        data.sequence_ids,
        np.repeat(data.targets, 4),
        data.predictions.reshape(-1),
        data.confidences.reshape(-1),
        data.costs.reshape(-1),
    )
    utility_gain = learned["mean_joint_utility"] - stage0["mean_joint_utility"]
    checks = {
        "positive_utility_gain": utility_gain >= minimum_utility_gain,
        "false_and_missed_edit_safety": _safety_passes(
            learned, stage0, safety_margin
        ),
        "not_above_oracle": (
            learned["mean_joint_utility"]
            <= oracle["oracle"]["mean_joint_utility"] + 1e-12
        ),
    }
    report = {
        "protocol": {
            "schema_version": "tool-belief-stopping-ensemble-eval-v1",
            "split": "val",
            "sequence_count": len(data.sequence_ids),
            "member_count": len(checkpoints),
            "aggregation": "mean_train_calibrated_continue_margin",
            "members": member_protocol,
            "test_assets_read": False,
        },
        "stage_0": stage0,
        "learned_ensemble": learned,
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
            "mean_margins": mean_margin[index].tolist(),
            "member_margins": margins[:, index, :].tolist(),
            "stage_predictions": data.predictions[index].tolist(),
            "target": int(data.targets[index]),
            "prediction": int(data.predictions[index, selected_stages[index]]),
            "cost": float(data.costs[index, selected_stages[index]]),
        }
        for index, sequence_id in enumerate(data.sequence_ids)
    ]
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--val-details", type=Path, required=True)
    parser.add_argument("--selector-states", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report, rows = evaluate_ensemble(
        args.checkpoint,
        args.stage_checkpoint,
        args.val_details,
        args.selector_states,
        device=args.device,
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
