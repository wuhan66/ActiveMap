"""Deterministic validation for the independent operation selector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from activemap.nn.operation_selector import OperationSelector, OperationSelectorConfig
from activemap.training.operation_selector import (
    INDEX_TO_NAME,
    CachedOperationDataset,
    operation_metrics,
)
from activemap.training.selector import resolve_device


def gated_operation_predictions(
    probabilities: torch.Tensor, update_threshold: float
) -> torch.Tensor:
    if not 0.0 <= update_threshold <= 1.0:
        raise ValueError("update_threshold must be between zero and one")
    keep_probability = probabilities[:, 0]
    update_probability = 1.0 - keep_probability
    update_prediction = torch.argmax(probabilities[:, 1:], dim=1) + 1
    return torch.where(
        update_probability >= update_threshold,
        update_prediction,
        torch.zeros_like(update_prediction),
    )


def calibrate_operation_predictions(
    prediction_path: Path,
    output_path: Path,
    *,
    max_false_edit: float = 0.05,
    grid_steps: int = 101,
) -> dict[str, Any]:
    if not 0.0 <= max_false_edit <= 1.0:
        raise ValueError("max_false_edit must be between zero and one")
    if grid_steps < 3:
        raise ValueError("grid_steps must be at least three")
    records = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("operation calibration requires predictions")
    names = [INDEX_TO_NAME[index] for index in range(4)]
    name_to_index = {name: index for index, name in INDEX_TO_NAME.items()}
    probabilities = torch.tensor(
        [[record["probabilities"][name] for name in names] for record in records],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [name_to_index[record["target"]] for record in records], dtype=torch.long
    )
    sweep: list[dict[str, float]] = []
    for threshold in torch.linspace(0.0, 1.0, grid_steps).tolist():
        predictions = gated_operation_predictions(probabilities, threshold)
        pseudo_logits = nn.functional.one_hot(predictions, num_classes=4).float()
        metrics = operation_metrics(pseudo_logits, targets)
        sweep.append({"update_threshold": threshold, **metrics})
    feasible = [row for row in sweep if row["false_edit_rate"] <= max_false_edit]
    candidates = feasible or sweep
    selected = max(
        candidates,
        key=lambda row: (
            row["macro_f1"],
            -row["missed_update_rate"],
            -row["update_threshold"],
        ),
    )
    summary = {
        "selection_objective": "maximize macro F1 under a stable-map false-edit constraint",
        "max_false_edit": max_false_edit,
        "constraint_satisfied": bool(feasible),
        "sample_count": len(records),
        "prediction_path": str(prediction_path.resolve()),
        "selected": selected,
        "sweep": sweep,
        "test_evaluation": None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def export_updater_operation_baseline(
    feature_cache_path: Path,
    output_dir: Path,
    *,
    split: str = "val",
) -> dict[str, Any]:
    """Export the frozen updater's own edit logits in selector prediction format."""

    if split not in {"train", "val"}:
        raise ValueError("updater operation baseline only permits train or val")
    cache = torch.load(feature_cache_path, map_location="cpu", weights_only=False)
    if split not in cache["splits"]:
        raise ValueError(f"feature cache does not contain split={split}")
    payload = cache["splits"][split]
    base_channels = int(cache["updater_model_config"]["base_channels"])
    edit_start = base_channels * 12 + 9
    edit_stop = edit_start + 4
    context = payload["context"].float()
    if context.shape[1] < edit_stop:
        raise ValueError("feature cache context does not contain updater edit logits")
    logits = context[:, edit_start:edit_stop]
    probabilities = torch.softmax(logits, dim=1)
    targets = payload["targets"].long()
    confidence, predictions = probabilities.max(dim=1)
    metrics = operation_metrics(logits, targets)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for index, sample_id in enumerate(payload["sample_ids"]):
            record = {
                "sample_id": str(sample_id),
                "target": INDEX_TO_NAME[int(targets[index])],
                "prediction": INDEX_TO_NAME[int(predictions[index])],
                "confidence": float(confidence[index]),
                "probabilities": {
                    INDEX_TO_NAME[class_index]: float(probabilities[index, class_index])
                    for class_index in range(4)
                },
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    summary = {
        "method": "frozen_updater_edit_logits",
        "feature_cache": str(feature_cache_path.resolve()),
        "updater_checkpoint": cache["updater_checkpoint"],
        "split": split,
        "sample_count": int(targets.shape[0]),
        "metrics": metrics,
        "prediction_path": str(prediction_path.resolve()),
        "test_evaluation": None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def evaluate_operation_selector_checkpoint(
    checkpoint_path: Path,
    feature_cache_path: Path,
    output_dir: Path,
    *,
    split: str = "val",
    device: str = "auto",
    batch_size: int = 64,
    num_workers: int = 0,
    update_threshold: float | None = None,
) -> dict[str, Any]:
    if split not in {"train", "val"}:
        raise ValueError("operation selector cache evaluation only permits train or val")
    target_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    cache = torch.load(feature_cache_path, map_location="cpu", weights_only=False)
    if split not in cache["splits"]:
        raise ValueError(f"feature cache does not contain split={split}")
    model = OperationSelector(OperationSelectorConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(target_device).eval()
    dataset = CachedOperationDataset(cache["splits"][split])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    logits_parts = []
    target_parts = []
    sample_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            logits_parts.append(
                model(
                    batch["spatial"].to(target_device),
                    batch["context"].to(target_device),
                ).cpu()
            )
            target_parts.append(batch["target"].cpu())
            sample_ids.extend(str(value) for value in batch["sample_id"])
    logits = torch.cat(logits_parts)
    targets = torch.cat(target_parts)
    probabilities = torch.softmax(logits, dim=1)
    if update_threshold is None:
        confidence, predictions = probabilities.max(dim=1)
        metrics = operation_metrics(logits, targets)
    else:
        predictions = gated_operation_predictions(probabilities, update_threshold)
        confidence = probabilities.gather(1, predictions[:, None]).squeeze(1)
        pseudo_logits = nn.functional.one_hot(predictions, num_classes=4).float()
        metrics = operation_metrics(pseudo_logits, targets)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for index, sample_id in enumerate(sample_ids):
            record = {
                "sample_id": sample_id,
                "target": INDEX_TO_NAME[int(targets[index])],
                "prediction": INDEX_TO_NAME[int(predictions[index])],
                "confidence": float(confidence[index]),
                "probabilities": {
                    INDEX_TO_NAME[class_index]: float(probabilities[index, class_index])
                    for class_index in range(probabilities.shape[1])
                },
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "feature_cache": str(feature_cache_path.resolve()),
        "updater_checkpoint": cache["updater_checkpoint"],
        "split": split,
        "sample_count": len(dataset),
        "device": str(target_device),
        "metrics": metrics,
        "update_threshold": update_threshold,
        "prediction_path": str(prediction_path.resolve()),
        "test_evaluation": None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
