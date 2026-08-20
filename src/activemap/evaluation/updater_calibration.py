"""Validation-only operating-point selection for hierarchical edit heads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from activemap.nn.updater import PriorConditionedUNet, UpdaterConfig, operation_probabilities
from activemap.training.selector import resolve_device
from activemap.training.updater_data import UpdaterDataset
from activemap.updater_records import load_updater_samples


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def classification_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    matrix = np.zeros((4, 4), dtype=np.int64)
    np.add.at(matrix, (target, prediction), 1)
    per_edit = []
    for index in range(4):
        true_positive = int(matrix[index, index])
        precision = _safe_divide(true_positive, int(matrix[:, index].sum()))
        recall = _safe_divide(true_positive, int(matrix[index].sum()))
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if precision + recall > 0.0
            else 0.0
        )
        per_edit.append({"precision": precision, "recall": recall, "f1": f1})
    keep_total = int(matrix[0].sum())
    changed_total = int(matrix[1:].sum())
    return {
        "edit_accuracy": _safe_divide(int(np.trace(matrix)), int(matrix.sum())),
        "macro_f1": float(np.mean([item["f1"] for item in per_edit])),
        "false_edit_rate": _safe_divide(keep_total - int(matrix[0, 0]), keep_total),
        "missed_update_rate": _safe_divide(int(matrix[1:, 0].sum()), changed_total),
        "delete_precision": per_edit[2]["precision"],
        "delete_recall": per_edit[2]["recall"],
        "delete_f1": per_edit[2]["f1"],
        "confusion": matrix.tolist(),
    }


def _threshold_predictions(
    auxiliary: np.ndarray,
    has_prior: np.ndarray,
    presence: np.ndarray,
    change: np.ndarray,
    *,
    presence_threshold: float,
    change_threshold: float,
) -> np.ndarray:
    prediction = auxiliary.copy()
    prediction[has_prior & (presence < presence_threshold)] = 2
    remaining = has_prior & (presence >= presence_threshold)
    prediction[remaining & (change < change_threshold)] = 0
    prediction[remaining & (change >= change_threshold)] = 3
    return prediction


def select_operating_point(
    target: np.ndarray,
    auxiliary: np.ndarray,
    has_prior: np.ndarray,
    presence: np.ndarray,
    change: np.ndarray,
    *,
    max_false_edit: float,
    grid_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sweep = []
    grid = np.linspace(0.05, 0.95, grid_steps)
    for presence_threshold in grid:
        for change_threshold in grid:
            prediction = _threshold_predictions(
                auxiliary,
                has_prior,
                presence,
                change,
                presence_threshold=float(presence_threshold),
                change_threshold=float(change_threshold),
            )
            metrics = classification_metrics(target, prediction)
            sweep.append(
                {
                    "presence_threshold": float(presence_threshold),
                    "change_threshold": float(change_threshold),
                    **metrics,
                }
            )
    feasible = [item for item in sweep if item["false_edit_rate"] <= max_false_edit]
    candidates = feasible or sweep
    selected = max(
        candidates,
        key=lambda item: (
            item["macro_f1"],
            item["delete_f1"],
            -item["false_edit_rate"],
        ),
    )
    return selected, sweep


def calibrate_updater_hierarchy(
    checkpoint_path: Path,
    samples_path: Path,
    output: Path,
    *,
    split: str = "val",
    device: str = "auto",
    batch_size: int = 64,
    max_false_edit: float = 0.05,
    grid_steps: int = 33,
) -> dict[str, Any]:
    target_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    config = UpdaterConfig(**checkpoint["model_config"])
    if not config.hierarchical_edit:
        raise ValueError("hierarchy calibration requires a hierarchical updater checkpoint")
    model = PriorConditionedUNet(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(target_device).eval()
    samples = load_updater_samples(samples_path, split=split)
    loader = DataLoader(
        UpdaterDataset(samples, temporal_pair_input=config.temporal_pair_input),
        batch_size=batch_size,
        shuffle=False,
    )
    targets = []
    auxiliaries = []
    prior_flags = []
    presences = []
    changes = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(target_device)
            prior = batch["prior_mask"].to(target_device)
            outputs = model(image, prior)
            targets.append(batch["edit_target"].numpy())
            auxiliaries.append(
                torch.argmax(operation_probabilities(outputs, prior), dim=-1).cpu().numpy()
            )
            prior_flags.append((prior.flatten(1).amax(dim=1) > 0.5).cpu().numpy())
            presences.append(torch.sigmoid(outputs["presence_logits"]).cpu().numpy())
            changes.append(torch.sigmoid(outputs["change_logits"]).cpu().numpy())
    selected, sweep = select_operating_point(
        np.concatenate(targets),
        np.concatenate(auxiliaries),
        np.concatenate(prior_flags),
        np.concatenate(presences),
        np.concatenate(changes),
        max_false_edit=max_false_edit,
        grid_steps=grid_steps,
    )
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "samples": str(samples_path.resolve()),
        "split": split,
        "sample_count": len(samples),
        "selection_objective": "maximize macro_f1 subject to false_edit_rate constraint",
        "max_false_edit": max_false_edit,
        "constraint_satisfied": selected["false_edit_rate"] <= max_false_edit,
        "selected": selected,
        "sweep": sweep,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
