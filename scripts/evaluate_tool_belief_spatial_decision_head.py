#!/usr/bin/env python3
"""Independently evaluate a frozen causal spatial Tool-Belief decision head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.analyze_tool_belief_stopping import _summary
from scripts.train_tool_belief_decision_head import EDIT_ORDER, KEEP_INDEX
from scripts.train_tool_belief_spatial_decision_head import (
    SpatialDecisionDataset,
    SpatialHierarchicalDecisionHead,
)


def _predict(
    model: SpatialHierarchicalDecisionHead,
    loader: DataLoader,
    thresholds: dict[str, float],
    *,
    device: torch.device,
) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "target",
            "baseline",
            "stage",
            "update_probability",
            "update_operation",
            "future_mass",
        )
    }
    model.eval()
    with torch.inference_mode():
        for features, spatial, target, baseline, _residual, stage in loader:
            update_logit, operation_logit = model(features.to(device), spatial.to(device))
            future_mass = []
            for sample_index, stage_value in enumerate(stage.tolist()):
                future_mass.append(float(spatial[sample_index, stage_value:].abs().sum()))
            collected["target"].append(target.numpy())
            collected["baseline"].append(baseline.numpy())
            collected["stage"].append(stage.numpy())
            collected["update_probability"].append(
                torch.sigmoid(update_logit).cpu().numpy()
            )
            collected["update_operation"].append(
                (torch.argmax(operation_logit, dim=1) + 1).cpu().numpy()
            )
            collected["future_mass"].append(np.asarray(future_mass, dtype=np.float32))
    output = {name: np.concatenate(parts) for name, parts in collected.items()}
    output["prediction"] = np.asarray(
        [
            operation if probability >= thresholds[str(int(stage))] else KEEP_INDEX
            for probability, operation, stage in zip(
                output["update_probability"],
                output["update_operation"],
                output["stage"],
                strict=True,
            )
        ],
        dtype=np.int64,
    )
    return output


def _stage_metrics(values: dict[str, np.ndarray], name: str) -> dict[str, Any]:
    results = {}
    for stage in sorted(int(value) for value in np.unique(values["stage"])):
        selected = values["stage"] == stage
        prediction = values["prediction"][selected]
        probability = values["update_probability"][selected]
        confidence = np.where(prediction == KEEP_INDEX, 1.0 - probability, probability)
        results[str(stage)] = _summary(
            name,
            values["target"][selected].tolist(),
            prediction.tolist(),
            confidence.tolist(),
            [0.0] * int(selected.sum()),
        )
    return results


def _identity_metrics(values: dict[str, np.ndarray]) -> dict[str, Any]:
    results = {}
    for stage in sorted(int(value) for value in np.unique(values["stage"])):
        selected = values["stage"] == stage
        results[str(stage)] = _summary(
            "identity",
            values["target"][selected].tolist(),
            values["baseline"][selected].tolist(),
            [0.5] * int(selected.sum()),
            [0.0] * int(selected.sum()),
        )
    return results


def _model_from_checkpoint(
    checkpoint: dict[str, Any], device: torch.device
) -> tuple[SpatialHierarchicalDecisionHead, bool]:
    config = checkpoint["config"]
    use_spatial = bool(config.get("use_spatial", True))
    model = SpatialHierarchicalDecisionHead(
        hidden_dim=int(config["hidden_dim"]),
        spatial_base_channels=int(config["spatial_base_channels"]),
        dropout=float(config["dropout"]),
        use_spatial=use_spatial,
        fusion_mode=str(config.get("fusion_mode", "concat")),
        gate_bias=float(config.get("gate_bias", -2.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, use_spatial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("details", type=Path)
    parser.add_argument("records", type=Path)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--safety-margin", type=float, default=0.02)
    parser.add_argument("--minimum-final-gain", type=float, default=0.01)
    args = parser.parse_args()
    checkpoint: dict[str, Any] = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    config = checkpoint["config"]
    supported_protocols = {
        "spatial_hierarchical_decision_head_v1",
        "spatial_hierarchical_decision_head_v2_gated_residual",
    }
    if config.get("protocol") not in supported_protocols:
        raise ValueError("checkpoint is not a supported spatial decision head")
    thresholds = {
        str(stage): float(value)
        for stage, value in checkpoint["decision_thresholds"].items()
    }
    if set(thresholds) != {"0", "1", "2", "3"}:
        raise ValueError("checkpoint must contain thresholds for stages 0,1,2,3")
    data = SpatialDecisionDataset(
        args.details,
        args.records,
        args.artifact_dir,
        args.split,
        spatial_size=int(config["spatial_size"]),
        stages=(0, 1, 2, 3),
    )
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    model, checkpoint_uses_spatial = _model_from_checkpoint(checkpoint, device)
    spatial_values = _predict(model, loader, thresholds, device=device)
    spatial_metrics = _stage_metrics(spatial_values, "spatial")
    identity_metrics = _identity_metrics(spatial_values)
    model.use_spatial = False
    masked_values = _predict(model, loader, thresholds, device=device)
    masked_metrics = _stage_metrics(masked_values, "masked_spatial")
    masking_probability_change = float(
        np.max(
            np.abs(
                spatial_values["update_probability"]
                - masked_values["update_probability"]
            )
        )
    )
    masking_prediction_changes = int(
        np.count_nonzero(
            spatial_values["prediction"] != masked_values["prediction"]
        )
    )
    no_spatial_invariance = checkpoint_uses_spatial or (
        masking_probability_change == 0.0 and masking_prediction_changes == 0
    )
    stage_safety = {
        stage: (
            spatial_metrics[stage]["false_edit_rate"]
            <= identity_metrics[stage]["false_edit_rate"] + args.safety_margin + 1e-12
            and spatial_metrics[stage]["missed_edit_rate"]
            <= identity_metrics[stage]["missed_edit_rate"] + args.safety_margin + 1e-12
        )
        for stage in spatial_metrics
    }
    final_gain = spatial_metrics["3"]["macro_f1"] - identity_metrics["3"]["macro_f1"]
    report = {
        "protocol": f"{config['protocol']}_independent_eval",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_uses_spatial": checkpoint_uses_spatial,
        "split": args.split,
        "example_count": len(data),
        "thresholds": thresholds,
        "identity_by_stage": identity_metrics,
        "spatial_by_stage": spatial_metrics,
        "masked_spatial_by_stage": masked_metrics,
        "stage_safety": stage_safety,
        "final_macro_f1_gain_over_identity": final_gain,
        "final_macro_f1_gain_over_masked_spatial": (
            spatial_metrics["3"]["macro_f1"] - masked_metrics["3"]["macro_f1"]
        ),
        "maximum_update_probability_change_when_masked": masking_probability_change,
        "prediction_changes_when_masked": masking_prediction_changes,
        "no_spatial_masking_invariance": no_spatial_invariance,
        "maximum_causal_future_mass": float(spatial_values["future_mass"].max()),
        "test_assets_read": False,
        "passed": (
            all(stage_safety.values())
            and final_gain >= args.minimum_final_gain
            and float(spatial_values["future_mass"].max()) == 0.0
            and no_spatial_invariance
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for example, target, prediction, probability in zip(
            data.examples,
            spatial_values["target"],
            spatial_values["prediction"],
            spatial_values["update_probability"],
            strict=True,
        ):
            handle.write(
                json.dumps(
                    {
                        "sequence_id": example["sequence_id"],
                        "stage": example["stage"],
                        "target": EDIT_ORDER[int(target)],
                        "prediction": EDIT_ORDER[int(prediction)],
                        "update_probability": float(probability),
                    }
                )
                + "\n"
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
