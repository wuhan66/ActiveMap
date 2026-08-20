"""Validation-only threshold calibration for explicit ADD and REMOVE raster heads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from activemap.nn.updater import PriorConditionedUNet, UpdaterConfig
from activemap.training.selector import resolve_device
from activemap.training.updater_data import UpdaterDataset
from activemap.updater_records import load_updater_samples


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _channel_sweep(
    probabilities: np.ndarray,
    target: np.ndarray,
    domain: np.ndarray,
    valid: np.ndarray,
    thresholds: np.ndarray,
) -> list[dict[str, float]]:
    target = target.astype(bool) & domain & valid
    active_domain = domain & valid
    positive_samples = target.reshape(len(target), -1).any(axis=1)
    stable_samples = ~positive_samples
    sweep: list[dict[str, float]] = []
    for threshold in thresholds:
        prediction = (probabilities >= threshold) & active_domain
        intersection = (prediction & target).sum(axis=(1, 2, 3), dtype=np.float64)
        union = (prediction | target).sum(axis=(1, 2, 3), dtype=np.float64)
        positive_iou = np.divide(
            intersection[positive_samples],
            union[positive_samples],
            out=np.zeros(int(positive_samples.sum()), dtype=np.float64),
            where=union[positive_samples] > 0,
        )
        stable_prediction = prediction[stable_samples].sum(dtype=np.float64)
        stable_domain = active_domain[stable_samples].sum(dtype=np.float64)
        true_positive = float((prediction & target).sum())
        predicted_positive = float(prediction.sum())
        target_positive = float(target.sum())
        precision = _safe_divide(true_positive, predicted_positive)
        recall = _safe_divide(true_positive, target_positive)
        pixel_f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        sweep.append(
            {
                "threshold": float(threshold),
                "mean_positive_iou": (float(positive_iou.mean()) if len(positive_iou) else 0.0),
                "pixel_precision": precision,
                "pixel_recall": recall,
                "pixel_f1": pixel_f1,
                "stable_false_positive_fraction": _safe_divide(stable_prediction, stable_domain),
                "positive_sample_count": int(positive_samples.sum()),
                "stable_sample_count": int(stable_samples.sum()),
            }
        )
    return sweep


def select_temporal_change_thresholds(
    add_probabilities: np.ndarray,
    remove_probabilities: np.ndarray,
    target_mask: np.ndarray,
    prior_mask: np.ndarray,
    valid_mask: np.ndarray,
    *,
    max_stable_false_positive: float = 0.005,
    grid_steps: int = 37,
) -> dict[str, Any]:
    """Select independent validation thresholds under stable-map safety constraints."""

    arrays = (
        add_probabilities,
        remove_probabilities,
        target_mask,
        prior_mask,
        valid_mask,
    )
    if any(array.shape != target_mask.shape for array in arrays):
        raise ValueError("temporal calibration arrays must have identical shapes")
    if target_mask.ndim != 4 or target_mask.shape[1] != 1:
        raise ValueError("temporal calibration arrays must have shape [N,1,H,W]")
    if not 0.0 <= max_stable_false_positive <= 1.0:
        raise ValueError("max_stable_false_positive must be between zero and one")
    if grid_steps < 3:
        raise ValueError("grid_steps must be at least three")

    target = target_mask >= 0.5
    prior = prior_mask >= 0.5
    valid = valid_mask >= 0.5
    thresholds = np.linspace(0.05, 0.95, grid_steps)
    add_sweep = _channel_sweep(add_probabilities, target & ~prior, ~prior, valid, thresholds)
    remove_sweep = _channel_sweep(remove_probabilities, prior & ~target, prior, valid, thresholds)

    def select(sweep: list[dict[str, float]]) -> tuple[dict[str, float], bool]:
        feasible = [
            row
            for row in sweep
            if row["stable_false_positive_fraction"] <= max_stable_false_positive
        ]
        candidates = feasible or sweep
        return (
            max(
                candidates,
                key=lambda row: (
                    row["mean_positive_iou"],
                    row["pixel_f1"],
                    -row["stable_false_positive_fraction"],
                    row["threshold"],
                ),
            ),
            bool(feasible),
        )

    selected_add, add_constraint = select(add_sweep)
    selected_remove, remove_constraint = select(remove_sweep)
    add_iou = selected_add["mean_positive_iou"]
    remove_iou = selected_remove["mean_positive_iou"]
    harmonic = _safe_divide(2.0 * add_iou * remove_iou, add_iou + remove_iou)
    return {
        "selection_objective": (
            "maximize per-channel mean positive-sample IoU subject to stable-map "
            "false-positive constraints"
        ),
        "max_stable_false_positive": max_stable_false_positive,
        "constraint_satisfied": add_constraint and remove_constraint,
        "selected": {
            "add_threshold": selected_add["threshold"],
            "remove_threshold": selected_remove["threshold"],
            "add": selected_add,
            "remove": selected_remove,
            "harmonic_mean_positive_iou": harmonic,
        },
        "add_sweep": add_sweep,
        "remove_sweep": remove_sweep,
    }


def calibrate_updater_temporal_change(
    checkpoint_path: Path,
    samples_path: Path,
    output: Path,
    *,
    split: str = "val",
    device: str = "auto",
    batch_size: int = 8,
    num_workers: int = 0,
    max_stable_false_positive: float = 0.005,
    grid_steps: int = 37,
) -> dict[str, Any]:
    """Run one validation inference pass and freeze ADD/REMOVE mask thresholds."""

    target_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    config = UpdaterConfig(**checkpoint["model_config"])
    if not config.temporal_change_head:
        raise ValueError("temporal calibration requires temporal_change_head=true")
    model = PriorConditionedUNet(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(target_device).eval()
    samples = load_updater_samples(samples_path, split=split)
    loader = DataLoader(
        UpdaterDataset(samples, temporal_pair_input=config.temporal_pair_input),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    add_probabilities = []
    remove_probabilities = []
    targets = []
    priors = []
    valid_masks = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(target_device)
            prior = batch["prior_mask"].to(target_device)
            outputs = model(image, prior)
            temporal = torch.sigmoid(outputs["temporal_change_logits"])
            add_probabilities.append(temporal[:, 0:1].cpu().numpy())
            remove_probabilities.append(temporal[:, 1:2].cpu().numpy())
            targets.append(batch["target_mask"].numpy())
            priors.append(batch["prior_mask"].numpy())
            valid_masks.append(batch["valid_mask"].numpy())
    result = select_temporal_change_thresholds(
        np.concatenate(add_probabilities),
        np.concatenate(remove_probabilities),
        np.concatenate(targets),
        np.concatenate(priors),
        np.concatenate(valid_masks),
        max_stable_false_positive=max_stable_false_positive,
        grid_steps=grid_steps,
    )
    result.update(
        {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "samples": str(samples_path.resolve()),
            "split": split,
            "sample_count": len(samples),
            "device": str(target_device),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
