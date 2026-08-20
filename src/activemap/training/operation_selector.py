"""Feature caching and class-balanced training for typed operation selection."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from activemap.config import load_yaml
from activemap.models import EditOperation
from activemap.nn.operation_selector import (
    OperationSelector,
    OperationSelectorConfig,
    operation_selector_inputs,
)
from activemap.nn.updater import PriorConditionedUNet, UpdaterConfig
from activemap.training.monitoring import RunMonitor, save_checkpoint
from activemap.training.selector import resolve_device, set_global_seed
from activemap.training.updater_data import EDIT_TO_INDEX, UpdaterDataset
from activemap.updater_records import load_updater_samples

INDEX_TO_NAME = {index: operation.value for operation, index in EDIT_TO_INDEX.items()}


class CachedOperationDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(self, payload: dict[str, Any]) -> None:
        self.spatial = payload["spatial"]
        self.context = payload["context"]
        self.targets = payload["targets"]
        self.sample_ids = payload["sample_ids"]

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        return {
            "spatial": self.spatial[index].float(),
            "context": self.context[index].float(),
            "target": self.targets[index].long(),
            "sample_id": self.sample_ids[index],
        }


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _extract_split(
    updater: PriorConditionedUNet,
    samples_path: Path,
    split: str,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    input_size: int,
    spatial_size: int,
) -> dict[str, Any]:
    samples = load_updater_samples(samples_path, split=split)
    loader = DataLoader(
        UpdaterDataset(
            samples,
            input_size=input_size,
            temporal_pair_input=updater.config.temporal_pair_input,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    spatial_parts: list[Tensor] = []
    context_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    sample_ids: list[str] = []
    updater.eval()
    with torch.inference_mode():
        for raw_batch in tqdm(loader, desc=f"Cache {split}", unit="batch"):
            batch = _to_device(raw_batch, device)
            outputs = updater(batch["image"], batch["prior_mask"])
            spatial, context = operation_selector_inputs(
                outputs, batch["prior_mask"], spatial_size=spatial_size
            )
            spatial_parts.append(spatial.detach().cpu().half())
            context_parts.append(context.detach().cpu().float())
            target_parts.append(batch["edit_target"].detach().cpu())
            sample_ids.extend(str(value) for value in raw_batch["sample_id"])
    return {
        "spatial": torch.cat(spatial_parts),
        "context": torch.cat(context_parts),
        "targets": torch.cat(target_parts),
        "sample_ids": sample_ids,
    }


def build_operation_feature_cache(
    updater_checkpoint: Path,
    samples_path: Path,
    cache_path: Path,
    *,
    device: str = "auto",
    batch_size: int = 4,
    num_workers: int = 0,
    input_size: int = 512,
    spatial_size: int = 64,
) -> dict[str, Any]:
    """Cache frozen updater evidence for train and validation only."""

    target_device = resolve_device(device)
    checkpoint = torch.load(updater_checkpoint, map_location=target_device, weights_only=False)
    updater = PriorConditionedUNet(UpdaterConfig(**checkpoint["model_config"]))
    updater.load_state_dict(checkpoint["state_dict"])
    updater.to(target_device).eval()
    splits = {
        split: _extract_split(
            updater,
            samples_path,
            split,
            device=target_device,
            batch_size=batch_size,
            num_workers=num_workers,
            input_size=input_size,
            spatial_size=spatial_size,
        )
        for split in ("train", "val")
    }
    context_dim = int(splits["train"]["context"].shape[1])
    payload = {
        "schema_version": 1,
        "updater_checkpoint": str(updater_checkpoint.resolve()),
        "updater_epoch": int(checkpoint.get("epoch", -1)),
        "updater_model_config": checkpoint["model_config"],
        "samples": str(samples_path.resolve()),
        "input_size": input_size,
        "spatial_size": spatial_size,
        "context_dim": context_dim,
        "splits": splits,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(cache_path)
    return payload


def _class_counts(targets: Tensor, classes: int) -> Tensor:
    counts = torch.bincount(targets.long(), minlength=classes).float()
    if torch.any(counts == 0):
        missing = [INDEX_TO_NAME[index] for index in torch.where(counts == 0)[0].tolist()]
        raise ValueError(f"training split is missing operation classes: {missing}")
    return counts


def effective_class_weights(targets: Tensor, classes: int, beta: float) -> Tensor:
    if not 0.0 <= beta < 1.0:
        raise ValueError("class_weight_beta must be in [0, 1)")
    counts = _class_counts(targets, classes)
    if beta == 0.0:
        return torch.ones_like(counts)
    weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
    return weights / weights.mean()


def focal_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    *,
    class_weights: Tensor,
    gamma: float,
    label_smoothing: float,
) -> Tensor:
    if gamma < 0.0:
        raise ValueError("focal_gamma must be non-negative")
    cross_entropy = nn.functional.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    probabilities = torch.softmax(logits, dim=1)
    target_probability = probabilities.gather(1, targets[:, None]).squeeze(1)
    return (((1.0 - target_probability) ** gamma) * cross_entropy).mean()


def operation_metrics(logits: Tensor, targets: Tensor) -> dict[str, float]:
    predictions = torch.argmax(logits, dim=1)
    classes = logits.shape[1]
    confusion = torch.zeros(classes, classes, dtype=torch.long)
    for target, prediction in zip(targets.tolist(), predictions.tolist(), strict=True):
        confusion[target, prediction] += 1
    metrics: dict[str, float] = {
        "accuracy": float((predictions == targets).float().mean()),
    }
    f1_values: list[float] = []
    recall_values: list[float] = []
    for index in range(classes):
        true_positive = float(confusion[index, index])
        false_positive = float(confusion[:, index].sum() - confusion[index, index])
        false_negative = float(confusion[index, :].sum() - confusion[index, index])
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        name = INDEX_TO_NAME[index].lower()
        metrics[f"precision_{name}"] = precision
        metrics[f"recall_{name}"] = recall
        metrics[f"f1_{name}"] = f1
        f1_values.append(f1)
        recall_values.append(recall)
    keep_index = EDIT_TO_INDEX[EditOperation.KEEP]
    keep_targets = targets == keep_index
    predicted_keep = predictions == keep_index
    metrics["macro_f1"] = float(np.mean(f1_values))
    metrics["balanced_accuracy"] = float(np.mean(recall_values))
    metrics["false_edit_rate"] = float(
        ((~predicted_keep) & keep_targets).sum() / keep_targets.sum().clamp_min(1)
    )
    update_targets = ~keep_targets
    metrics["missed_update_rate"] = float(
        (predicted_keep & update_targets).sum() / update_targets.sum().clamp_min(1)
    )
    return metrics


def _run_epoch(
    model: OperationSelector,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    class_weights: Tensor,
    focal_gamma: float,
    label_smoothing: float,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
    description: str,
) -> dict[str, float]:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_samples = 0
    logits_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    progress = tqdm(loader, desc=description, unit="batch")
    for raw_batch in progress:
        batch = _to_device(raw_batch, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        logits = model(batch["spatial"], batch["context"])
        loss = focal_cross_entropy(
            logits,
            batch["target"],
            class_weights=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
        if optimizer is not None:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        count = int(batch["target"].shape[0])
        total_loss += float(loss.detach().cpu()) * count
        total_samples += count
        logits_parts.append(logits.detach().cpu())
        target_parts.append(batch["target"].detach().cpu())
        progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
    metrics = operation_metrics(torch.cat(logits_parts), torch.cat(target_parts))
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def _selection_score(metrics: dict[str, float], false_edit_limit: float) -> float:
    violation = max(metrics["false_edit_rate"] - false_edit_limit, 0.0)
    return metrics["macro_f1"] - 2.0 * violation


def train_operation_selector(
    config_path: Path, *, output_override: Path | None = None
) -> dict[str, Any]:
    config = load_yaml(config_path)
    seed = int(config.get("seed", 20260731))
    set_global_seed(seed)
    data = config["data"]
    training = config.get("training", {})
    model_payload = config.get("model", {})
    updater_checkpoint = Path(data["updater_checkpoint"])
    samples_path = Path(data["samples"])
    cache_path = Path(data["feature_cache"])
    output_dir = output_override or Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    device = resolve_device(str(training.get("device", "auto")))
    spatial_size = int(model_payload.get("spatial_size", 64))
    refresh_cache = bool(data.get("refresh_cache", False))
    if refresh_cache or not cache_path.is_file():
        build_operation_feature_cache(
            updater_checkpoint,
            samples_path,
            cache_path,
            device=str(device),
            batch_size=int(data.get("cache_batch_size", 4)),
            num_workers=int(data.get("cache_num_workers", 0)),
            input_size=int(data.get("input_size", 512)),
            spatial_size=spatial_size,
        )
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if str(Path(cache["updater_checkpoint"]).resolve()) != str(updater_checkpoint.resolve()):
        raise ValueError("feature cache was built from a different updater checkpoint")
    if int(cache["spatial_size"]) != spatial_size:
        raise ValueError("feature cache spatial size does not match the selector config")
    model_config = OperationSelectorConfig(
        context_dim=int(cache["context_dim"]),
        spatial_size=spatial_size,
        base_channels=int(model_payload.get("base_channels", 32)),
        hidden_dim=int(model_payload.get("hidden_dim", 256)),
        dropout=float(model_payload.get("dropout", 0.20)),
        use_spatial=bool(model_payload.get("use_spatial", True)),
        use_context=bool(model_payload.get("use_context", True)),
    )
    model = OperationSelector(model_config).to(device)
    train_dataset = CachedOperationDataset(cache["splits"]["train"])
    val_dataset = CachedOperationDataset(cache["splits"]["val"])
    train_targets = cache["splits"]["train"]["targets"].long()
    class_counts = _class_counts(train_targets, model_config.edit_classes)
    cache_summary = {
        "schema_version": cache["schema_version"],
        "updater_checkpoint": cache["updater_checkpoint"],
        "updater_epoch": cache["updater_epoch"],
        "samples": cache["samples"],
        "input_size": cache["input_size"],
        "spatial_size": cache["spatial_size"],
        "context_dim": cache["context_dim"],
        "split_counts": {
            split: int(payload["targets"].shape[0]) for split, payload in cache["splits"].items()
        },
        "contains_test": "test" in cache["splits"],
    }
    (output_dir / "feature_cache_summary.json").write_text(
        json.dumps(cache_summary, indent=2) + "\n", encoding="utf-8"
    )
    sampling_power = float(training.get("sampling_power", 0.5))
    sample_weights = class_counts[train_targets].pow(-sampling_power)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    batch_size = int(training.get("batch_size", 32))
    workers = int(training.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    class_weights = effective_class_weights(
        train_targets,
        model_config.edit_classes,
        float(training.get("class_weight_beta", 0.99)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training.get("scheduler_factor", 0.5)),
        patience=int(training.get("scheduler_patience", 4)),
        min_lr=float(training.get("min_learning_rate", 1e-6)),
    )
    epochs = int(training.get("max_epochs", 60))
    min_epochs = int(training.get("min_epochs", 10))
    patience = int(training.get("patience", 12))
    false_edit_limit = float(training.get("false_edit_limit", 0.05))
    focal_gamma = float(training.get("focal_gamma", 1.5))
    label_smoothing = float(training.get("label_smoothing", 0.02))
    grad_clip = float(training.get("grad_clip", 1.0))
    monitor = RunMonitor(output_dir, config.get("monitoring", {}))
    best_score = -float("inf")
    best_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    monitor.write_state("running", epoch=0, total_epochs=epochs)
    for epoch in range(1, epochs + 1):
        if monitor.wait_if_paused(epoch=epoch) or monitor.should_stop():
            monitor.write_state("stopped", epoch=epoch - 1, total_epochs=epochs)
            break
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            optimizer=optimizer,
            grad_clip=grad_clip,
            description=f"Epoch {epoch:03d}/{epochs:03d} train",
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model,
                val_loader,
                device=device,
                class_weights=class_weights,
                focal_gamma=focal_gamma,
                label_smoothing=label_smoothing,
                optimizer=None,
                grad_clip=grad_clip,
                description=f"Epoch {epoch:03d}/{epochs:03d} validation",
            )
        score = _selection_score(val_metrics, false_edit_limit)
        scheduler.step(score)
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "selection_score": score,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        monitor.record_epoch(record)
        checkpoint = {
            "state_dict": model.state_dict(),
            "model_config": model_config.as_dict(),
            "updater_checkpoint": str(updater_checkpoint.resolve()),
            "feature_cache": str(cache_path.resolve()),
            "seed": seed,
            "epoch": epoch,
            "val_metrics": val_metrics,
            "selection_score": score,
            "class_counts": class_counts.tolist(),
            "class_weights": class_weights.detach().cpu().tolist(),
        }
        if score > best_score:
            best_score = score
            stale_epochs = 0
            save_checkpoint(output_dir / "best.pt", checkpoint)
        else:
            stale_epochs += 1
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            save_checkpoint(output_dir / "best_val_loss.pt", checkpoint)
        save_checkpoint(
            output_dir / "last.pt",
            {
                **checkpoint,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_score": best_score,
                "best_loss": best_loss,
                "stale_epochs": stale_epochs,
                "history": history,
            },
        )
        monitor.write_state(
            "running",
            epoch=epoch,
            total_epochs=epochs,
            selection_score=score,
            latest_metrics=val_metrics,
        )
        if epoch >= min_epochs and stale_epochs >= patience:
            break
    completed_epoch = history[-1]["epoch"] if history else 0
    summary = {
        "config": str(config_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "updater_checkpoint": str(updater_checkpoint.resolve()),
        "feature_cache": str(cache_path.resolve()),
        "device": str(device),
        "parameter_count": model.parameter_count(),
        "class_counts": {
            INDEX_TO_NAME[index]: int(value) for index, value in enumerate(class_counts.tolist())
        },
        "class_weights": {
            INDEX_TO_NAME[index]: float(value)
            for index, value in enumerate(class_weights.detach().cpu().tolist())
        },
        "best_selection_score": best_score,
        "best_val_loss": best_loss,
        "epochs_completed": completed_epoch,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    monitor.write_state(
        "completed",
        epoch=completed_epoch,
        total_epochs=epochs,
        converged=completed_epoch < epochs,
    )
    monitor.close()
    return summary
