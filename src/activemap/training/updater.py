"""Training and validation loop for the prior-conditioned map updater."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from activemap.config import load_yaml
from activemap.models import EditOperation
from activemap.nn.updater import (
    PriorConditionedUNet,
    UpdaterConfig,
    operation_probabilities,
    updater_loss,
)
from activemap.training.monitoring import RunMonitor, save_checkpoint
from activemap.training.provenance import archive_updater_run, write_updater_provenance
from activemap.training.selector import resolve_device, set_global_seed
from activemap.training.updater_data import (
    EDIT_TO_INDEX,
    UpdaterAugmentationConfig,
    UpdaterDataset,
)
from activemap.training.visualization import render_updater_progress
from activemap.updater_records import load_updater_samples


def initialize_updater_weights(
    model: PriorConditionedUNet,
    checkpoint_path: Path,
    *,
    scope: str,
    device: torch.device,
) -> dict[str, Any]:
    """Initialize a stage from either all weights or the shared visual encoder."""

    checkpoint: Any = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict: dict[str, Tensor] = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    if not isinstance(state_dict, dict):
        raise ValueError(f"invalid updater checkpoint: {checkpoint_path}")
    if scope == "full":
        model.load_state_dict(state_dict, strict=True)
        loaded_keys = list(state_dict)
    elif scope == "encoder":
        encoder_prefixes = ("encoder1.", "encoder2.", "encoder3.", "bottleneck.")
        encoder_state = {
            key: value for key, value in state_dict.items() if key.startswith(encoder_prefixes)
        }
        if not encoder_state:
            raise ValueError(f"checkpoint has no encoder weights: {checkpoint_path}")
        result = model.load_state_dict(encoder_state, strict=False)
        if result.unexpected_keys:
            raise ValueError(f"unexpected encoder keys: {result.unexpected_keys}")
        loaded_keys = list(encoder_state)
    elif scope == "segmentation":
        segmentation_prefixes = (
            "encoder1.",
            "encoder2.",
            "encoder3.",
            "bottleneck.",
            "up3.",
            "decoder3.",
            "up2.",
            "decoder2.",
            "up1.",
            "decoder1.",
            "segmentation_head.",
        )
        segmentation_state = {
            key: value for key, value in state_dict.items() if key.startswith(segmentation_prefixes)
        }
        if not segmentation_state:
            raise ValueError(f"checkpoint has no segmentation weights: {checkpoint_path}")
        result = model.load_state_dict(segmentation_state, strict=False)
        if result.unexpected_keys:
            raise ValueError(f"unexpected segmentation keys: {result.unexpected_keys}")
        loaded_keys = list(segmentation_state)
    elif scope == "compatible":
        model_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        if not compatible_state:
            raise ValueError(f"checkpoint has no compatible weights: {checkpoint_path}")
        result = model.load_state_dict(compatible_state, strict=False)
        if result.unexpected_keys:
            raise ValueError(f"unexpected compatible keys: {result.unexpected_keys}")
        loaded_keys = list(compatible_state)
    else:
        raise ValueError(
            "training.init_scope must be 'full', 'encoder', 'segmentation', or 'compatible'"
        )
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "scope": scope,
        "loaded_tensor_count": len(loaded_keys),
    }


def set_updater_trainable_scope(
    model: PriorConditionedUNet,
    *,
    roi_only: bool = False,
    evidence_only: bool = False,
) -> int:
    """Freeze inherited parameters during a short residual-head warm-up."""

    if roi_only and evidence_only:
        raise ValueError("ROI-only and evidence-only warm-up are mutually exclusive")
    if roi_only and not model.config.prior_guided_roi:
        raise ValueError("ROI-only warm-up requires model.prior_guided_roi=true")
    if evidence_only and not (
        model.config.segmentation_evidence
        or model.config.temporal_change_to_edit_head
        or model.config.temporal_spatial_edit_head
    ):
        raise ValueError("evidence-only warm-up requires an enabled evidence head")
    evidence_prefixes = (
        "segmentation_evidence_head.",
        "temporal_change_evidence_head.",
        "temporal_spatial_edit_encoder.",
    )
    for name, parameter in model.named_parameters():
        selected = (
            name.startswith("roi_projection.") if roi_only else name.startswith(evidence_prefixes)
        )
        parameter.requires_grad = not (roi_only or evidence_only) or selected
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def updater_config_from_payload(model_payload: dict[str, Any]) -> UpdaterConfig:
    """Parse every persisted updater architecture option from YAML."""

    return UpdaterConfig(
        image_channels=int(model_payload.get("image_channels", 3)),
        prior_channels=int(model_payload.get("prior_channels", 1)),
        base_channels=int(model_payload.get("base_channels", 32)),
        geometry_dim=int(model_payload.get("geometry_dim", 8)),
        edit_classes=int(model_payload.get("edit_classes", 4)),
        dropout=float(model_payload.get("dropout", 0.10)),
        use_prior=bool(model_payload.get("use_prior", True)),
        hierarchical_edit=bool(model_payload.get("hierarchical_edit", False)),
        auxiliary_edit_head=bool(model_payload.get("auxiliary_edit_head", True)),
        geometry_head_mode=str(model_payload.get("geometry_head_mode", "shared")),
        prior_guided_roi=bool(model_payload.get("prior_guided_roi", False)),
        segmentation_evidence=bool(model_payload.get("segmentation_evidence", False)),
        vector_change_encoder=bool(model_payload.get("vector_change_encoder", False)),
        vector_change_to_edit_head=bool(model_payload.get("vector_change_to_edit_head", False)),
        temporal_change_head=bool(model_payload.get("temporal_change_head", False)),
        temporal_change_to_edit_head=bool(model_payload.get("temporal_change_to_edit_head", False)),
        temporal_spatial_edit_head=bool(model_payload.get("temporal_spatial_edit_head", False)),
        temporal_pair_input=bool(model_payload.get("temporal_pair_input", False)),
    )


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    # Pinned host buffers can be copied asynchronously on CUDA, while CPU and
    # accelerator backends retain their existing synchronous behavior.
    non_blocking = device.type == "cuda"
    return {
        key: value.to(device, non_blocking=non_blocking) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _batch_metrics(outputs: dict[str, Tensor], batch: dict[str, Any]) -> dict[str, float]:
    prediction = torch.sigmoid(outputs["segmentation_logits"]) >= 0.5
    target = batch["target_mask"] >= 0.5
    valid = batch["valid_mask"] >= 0.5
    intersection = ((prediction & target) & valid).sum(dim=(1, 2, 3)).float()
    union = ((prediction | target) & valid).sum(dim=(1, 2, 3)).float()
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    target_pixels = (target & valid).sum(dim=(1, 2, 3))
    foreground = target_pixels > 0
    foreground_iou = (
        iou[foreground].mean() if foreground.any() else torch.zeros((), device=iou.device)
    )
    empty = ~foreground
    predicted_pixels = (prediction & valid).sum(dim=(1, 2, 3)).float()
    valid_pixels = valid.sum(dim=(1, 2, 3)).float().clamp_min(1.0)
    empty_false_positive = (
        (predicted_pixels[empty] / valid_pixels[empty]).mean()
        if empty.any()
        else torch.zeros((), device=iou.device)
    )
    hierarchical_prediction = torch.argmax(
        operation_probabilities(outputs, batch["prior_mask"]), dim=-1
    )
    auxiliary_prediction = (
        torch.argmax(outputs["edit_logits"], dim=-1)
        if "edit_logits" in outputs
        else hierarchical_prediction
    )
    edit_target = batch["edit_target"]
    supervision_types = batch.get("supervision_type", ["real_temporal"] * int(edit_target.shape[0]))
    full_scene = torch.tensor(
        [value == "full_scene_temporal" for value in supervision_types],
        device=edit_target.device,
    )
    temporal_supervision = torch.tensor(
        [value in {"real_temporal", "full_scene_temporal"} for value in supervision_types],
        device=edit_target.device,
    )
    prior = batch["prior_mask"] >= 0.5
    predicted_added = prediction & ~prior & valid
    target_added = target & ~prior & valid
    predicted_removed = ~prediction & prior & valid
    target_removed = ~target & prior & valid

    def change_iou_totals(predicted_change: Tensor, target_change: Tensor) -> tuple[Tensor, Tensor]:
        target_present = target_change.sum(dim=(1, 2, 3)) > 0
        selected = temporal_supervision & target_present
        if not selected.any():
            zero = torch.zeros((), device=edit_target.device)
            return zero, zero
        change_intersection = (predicted_change & target_change).sum(dim=(1, 2, 3)).float()
        change_union = (predicted_change | target_change).sum(dim=(1, 2, 3)).float()
        change_iou = torch.where(
            change_union > 0,
            change_intersection / change_union,
            torch.zeros_like(change_union),
        )
        return change_iou[selected].sum(), selected.sum().float()

    added_iou_sum, added_iou_count = change_iou_totals(predicted_added, target_added)
    removed_iou_sum, removed_iou_count = change_iou_totals(predicted_removed, target_removed)
    edit_prediction = torch.where(full_scene, auxiliary_prediction, hierarchical_prediction)
    keep = edit_target == 0
    false_edit = (
        (edit_prediction[keep] != 0).float().mean()
        if keep.any()
        else torch.zeros((), device=edit_target.device)
    )
    changed = ~keep
    missed_edit = (
        (edit_prediction[changed] == 0).float().mean()
        if changed.any()
        else torch.zeros((), device=edit_target.device)
    )
    delete = edit_target == EDIT_TO_INDEX[EditOperation.DELETE]
    delete_correct = ((edit_prediction == edit_target) & delete).sum()
    return {
        "iou": float(iou.mean().detach().cpu()),
        "foreground_iou": float(foreground_iou.detach().cpu()),
        "empty_scene_false_positive_fraction": float(empty_false_positive.detach().cpu()),
        "edit_accuracy": float((edit_prediction == edit_target).float().mean().detach().cpu()),
        "false_edit_rate": float(false_edit.detach().cpu()),
        "missed_edit_rate": float(missed_edit.detach().cpu()),
        "delete_correct": float(delete_correct.detach().cpu()),
        "delete_total": float(delete.sum().detach().cpu()),
        "added_change_iou_sum": float(added_iou_sum.detach().cpu()),
        "added_change_iou_count": float(added_iou_count.detach().cpu()),
        "removed_change_iou_sum": float(removed_iou_sum.detach().cpu()),
        "removed_change_iou_count": float(removed_iou_count.detach().cpu()),
    }


def _quality_score(val_metrics: dict[str, float], training: dict[str, Any]) -> tuple[float, str]:
    mode = str(training.get("quality_score_mode", "auto"))
    if mode not in {"auto", "raster", "temporal_change", "edit"}:
        raise ValueError(
            "training.quality_score_mode must be auto, raster, temporal_change, or edit"
        )
    has_temporal = (
        val_metrics.get("added_change_iou_count", 0.0) > 0
        or val_metrics.get("removed_change_iou_count", 0.0) > 0
    )
    resolved_mode = "temporal_change" if mode == "auto" and has_temporal else mode
    if resolved_mode == "auto":
        resolved_mode = "raster"
    if resolved_mode == "edit":
        spatial_quality = 0.0
    elif resolved_mode == "temporal_change":
        temporal_values = [
            val_metrics[name]
            for name, count_name in (
                ("added_change_iou", "added_change_iou_count"),
                ("removed_change_iou", "removed_change_iou_count"),
            )
            if val_metrics.get(count_name, 0.0) > 0
        ]
        if not temporal_values:
            raise ValueError(
                "quality_score_mode=temporal_change requires validation samples with changes"
            )
        spatial_quality = float(np.mean(temporal_values))
    else:
        spatial_quality = val_metrics["iou"]
    score = (
        spatial_quality
        + val_metrics["edit_accuracy"]
        + float(training.get("quality_delete_recall_weight", 0.0)) * val_metrics["delete_recall"]
        - float(training.get("quality_false_edit_weight", 0.0))
        * val_metrics.get("false_edit_rate", 0.0)
        - float(training.get("quality_missed_edit_weight", 0.0))
        * val_metrics.get("missed_edit_rate", 0.0)
    )
    return score, resolved_mode


def _safety_eligible(
    val_metrics: dict[str, float], training: dict[str, Any]
) -> tuple[bool, dict[str, float]]:
    """Apply optional hard validation gates before selecting best_safety.pt."""

    configured = {
        "false_edit_rate": training.get("safety_false_edit_limit"),
        "missed_edit_rate": training.get("safety_missed_edit_limit"),
    }
    limits: dict[str, float] = {}
    for metric, raw_limit in configured.items():
        if raw_limit is None:
            continue
        limit = float(raw_limit)
        if not 0.0 <= limit <= 1.0:
            raise ValueError(f"training safety limit for {metric} must be between zero and one")
        limits[metric] = limit
    return all(val_metrics[metric] <= limit for metric, limit in limits.items()), limits


def _edit_class_weights(samples: list[Any], *, power: float = 0.5) -> Tensor:
    """Compute normalized inverse-frequency weights without zero-class explosions."""

    counts = torch.zeros(len(EDIT_TO_INDEX), dtype=torch.float32)
    for sample in samples:
        counts[EDIT_TO_INDEX[sample.edit_type]] += 1.0
    present = counts > 0
    if not present.any():
        raise ValueError("cannot compute edit class weights from an empty training split")
    weights = torch.zeros_like(counts)
    weights[present] = torch.pow(counts[present].sum() / counts[present], power)
    weights[present] /= weights[present].mean()
    weights[~present] = weights[present].max()
    return weights


def _training_sample_weights(
    samples: list[Any],
    *,
    dataset_balance_power: float,
    edit_sampling_weights: dict[str, float],
    source_group_balance_power: float = 0.0,
) -> list[float] | None:
    dataset_counts = Counter(sample.dataset_name for sample in samples)
    source_group_keys = [
        (
            sample.dataset_name,
            sample.aoi_id,
            sample.source_metadata.get("annotation_index", sample.object_id or sample.sample_id),
        )
        for sample in samples
    ]
    source_group_counts = Counter(source_group_keys)
    normalized_edit_weights = {
        key.upper(): float(value) for key, value in edit_sampling_weights.items()
    }
    unknown = set(normalized_edit_weights) - {operation.value for operation in EditOperation}
    if unknown:
        raise ValueError(f"unknown edit sampling operations: {sorted(unknown)}")
    if any(value <= 0.0 for value in normalized_edit_weights.values()):
        raise ValueError("data.edit_sampling_weights values must be positive")
    if (
        dataset_balance_power == 0.0
        and source_group_balance_power == 0.0
        and all(
            normalized_edit_weights.get(operation.value, 1.0) == 1.0 for operation in EditOperation
        )
    ):
        return None
    return [
        float(dataset_counts[sample.dataset_name] ** (-dataset_balance_power))
        * float(source_group_counts[group_key] ** (-source_group_balance_power))
        * normalized_edit_weights.get(sample.edit_type.value, 1.0)
        for sample, group_key in zip(samples, source_group_keys, strict=True)
    ]


def _filter_samples_by_edit(samples: list[Any], allowed_edits: list[str] | None) -> list[Any]:
    if allowed_edits is None:
        return samples
    if not isinstance(allowed_edits, list) or not allowed_edits:
        raise ValueError("data.allowed_edits must be a non-empty list")
    normalized = {str(value).upper() for value in allowed_edits}
    known = {operation.value for operation in EditOperation}
    unknown = normalized - known
    if unknown:
        raise ValueError(f"unknown data.allowed_edits operations: {sorted(unknown)}")
    filtered = [sample for sample in samples if sample.edit_type.value in normalized]
    if not filtered:
        raise ValueError("data.allowed_edits removed every updater sample")
    return filtered


def run_updater_epoch(
    model: PriorConditionedUNet,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    loss_settings: dict[str, Any],
    grad_clip: float,
    scaler: Any | None = None,
    use_amp: bool = False,
    epoch: int = 1,
    total_epochs: int = 1,
    phase: str = "train",
    show_progress: bool = True,
    gradient_accumulation_steps: int = 1,
    frozen_backbone: bool = False,
) -> dict[str, float]:
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if optimizer is None or frozen_backbone:
        model.eval()
    else:
        model.train()
    collected: dict[str, list[float]] = {
        "loss": [],
        "iou": [],
        "edit_accuracy": [],
        "false_edit_rate": [],
        "missed_edit_rate": [],
    }
    batches = tqdm(
        loader,
        desc=f"Epoch {epoch:03d}/{total_epochs:03d} {phase}",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
        mininterval=1.0,
        disable=not show_progress,
    )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, raw_batch in enumerate(batches):
        batch = _move(raw_batch, device)
        amp_enabled = use_amp and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            outputs = model(batch["image"], batch["prior_mask"])
            full_scene_mask = torch.tensor(
                [value == "full_scene_temporal" for value in batch["supervision_type"]],
                device=device,
            )
            temporal_supervision_mask = torch.tensor(
                [
                    value in {"real_temporal", "full_scene_temporal"}
                    for value in batch["supervision_type"]
                ],
                device=device,
            )
            loss, components = updater_loss(
                outputs,
                target_mask=batch["target_mask"],
                valid_mask=batch["valid_mask"],
                edit_target=batch["edit_target"],
                geometry_target=batch["geometry_target"],
                segmentation_weight=loss_settings["segmentation"],
                segmentation_bce_weight=loss_settings["segmentation_bce"],
                segmentation_dice_weight=loss_settings["segmentation_dice"],
                segmentation_focal_weight=loss_settings["segmentation_focal"],
                segmentation_cldice_weight=loss_settings.get("segmentation_cldice", 0.0),
                cldice_iterations=loss_settings.get("cldice_iterations", 5),
                focal_gamma=loss_settings["focal_gamma"],
                focal_alpha=loss_settings["focal_alpha"],
                edit_weight=loss_settings["edit"],
                edit_class_weights=loss_settings["edit_class_weights"],
                edit_label_smoothing=loss_settings["edit_label_smoothing"],
                geometry_weight=loss_settings["geometry"],
                geometry_beta=loss_settings["geometry_beta"],
                false_edit_weight=loss_settings["false_edit"],
                missed_edit_weight=loss_settings["missed_edit"],
                confidence_weight=loss_settings["confidence"],
                confidence_target_mode=loss_settings["confidence_target_mode"],
                presence_weight=loss_settings["presence"],
                change_weight=loss_settings["change"],
                prior_mask=batch["prior_mask"],
                full_scene_mask=full_scene_mask,
                temporal_supervision_mask=temporal_supervision_mask,
                temporal_change_weight=loss_settings["temporal_change"],
                temporal_change_bce_weight=loss_settings["temporal_change_bce"],
                temporal_change_dice_weight=loss_settings["temporal_change_dice"],
                temporal_change_focal_weight=loss_settings["temporal_change_focal"],
                temporal_change_focal_alpha=loss_settings["temporal_change_focal_alpha"],
            )
        if optimizer is not None:
            backward_loss = loss / gradient_accumulation_steps
            if scaler is not None and scaler.is_enabled():
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()
            should_step = (
                batch_index + 1
            ) % gradient_accumulation_steps == 0 or batch_index + 1 == len(loader)
            if should_step:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        metrics = _batch_metrics(outputs, batch)
        collected["loss"].append(float(loss.detach().cpu()))
        for name, component_value in components.items():
            collected.setdefault(f"loss_{name}", []).append(float(component_value.cpu()))
        for name, metric_value in metrics.items():
            collected.setdefault(name, []).append(metric_value)
        batches.set_postfix(
            loss=f"{collected['loss'][-1]:.4f}",
            iou=f"{metrics['iou']:.3f}",
            edit_acc=f"{metrics['edit_accuracy']:.3f}",
            refresh=False,
        )
    aggregate_names = {
        "delete_correct",
        "delete_total",
        "added_change_iou_sum",
        "added_change_iou_count",
        "removed_change_iou_sum",
        "removed_change_iou_count",
    }
    summary = {
        name: float(np.mean(values))
        for name, values in collected.items()
        if name not in aggregate_names
    }
    delete_total = sum(collected.get("delete_total", []))
    summary["delete_recall"] = (
        sum(collected.get("delete_correct", [])) / delete_total if delete_total > 0 else 0.0
    )
    for prefix in ("added", "removed"):
        count = sum(collected.get(f"{prefix}_change_iou_count", []))
        total = sum(collected.get(f"{prefix}_change_iou_sum", []))
        summary[f"{prefix}_change_iou"] = total / count if count > 0 else 0.0
        summary[f"{prefix}_change_iou_count"] = count
    return summary


def train_updater(config_path: Path, *, output_override: Path | None = None) -> dict[str, Any]:
    config = load_yaml(config_path)
    seed = int(config.get("seed", 20260710))
    set_global_seed(seed)
    samples_path = Path(config["data"]["samples"])
    if not samples_path.is_absolute():
        samples_path = (config_path.parent / samples_path).resolve()
    output_dir = output_override or Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_payload = config.get("model", {})
    model_config = updater_config_from_payload(model_payload)
    model = PriorConditionedUNet(model_config)
    training = config.get("training", {})
    device = resolve_device(str(training.get("device", "auto")))
    model.to(device)

    data_settings = config.get("data", {})
    all_samples = _filter_samples_by_edit(
        load_updater_samples(samples_path), data_settings.get("allowed_edits")
    )
    train_samples = [sample for sample in all_samples if sample.split == "train"]
    val_samples = [sample for sample in all_samples if sample.split == "val"]
    if not train_samples or not val_samples:
        raise ValueError("updater training requires non-empty train and validation splits")
    write_updater_provenance(
        output_dir,
        config=config,
        config_path=config_path,
        samples_path=samples_path,
        samples=all_samples,
    )
    batch_size = int(training.get("batch_size", 16))
    workers = int(training.get("num_workers", 0))
    if workers < 0:
        raise ValueError("training.num_workers must be non-negative")
    pin_memory = bool(training.get("pin_memory", False)) and device.type == "cuda"
    persistent_workers = bool(training.get("persistent_workers", False))
    prefetch_factor = int(training.get("prefetch_factor", 2))
    if persistent_workers and workers == 0:
        raise ValueError("training.persistent_workers requires training.num_workers > 0")
    if prefetch_factor < 1:
        raise ValueError("training.prefetch_factor must be positive")
    loader_kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    input_size_value = data_settings.get("input_size")
    input_size = int(input_size_value) if input_size_value is not None else None
    balance_power = float(data_settings.get("dataset_balance_power", 0.0))
    if not 0.0 <= balance_power <= 1.0:
        raise ValueError("data.dataset_balance_power must be in [0, 1]")
    source_group_balance_power = float(data_settings.get("source_group_balance_power", 0.0))
    if not 0.0 <= source_group_balance_power <= 1.0:
        raise ValueError("data.source_group_balance_power must be in [0, 1]")
    edit_sampling_payload = data_settings.get("edit_sampling_weights", {})
    if not isinstance(edit_sampling_payload, dict):
        raise ValueError("data.edit_sampling_weights must be a mapping")
    augmentation = UpdaterAugmentationConfig.from_dict(training.get("augmentation"))
    weights = _training_sample_weights(
        train_samples,
        dataset_balance_power=balance_power,
        edit_sampling_weights=edit_sampling_payload,
        source_group_balance_power=source_group_balance_power,
    )
    sampler = None
    if weights is not None:
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(train_samples),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    train_loader = DataLoader(
        UpdaterDataset(
            train_samples,
            augmentation=augmentation,
            input_size=input_size,
            temporal_pair_input=model_config.temporal_pair_input,
        ),
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        **loader_kwargs,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        UpdaterDataset(
            val_samples,
            input_size=input_size,
            temporal_pair_input=model_config.temporal_pair_input,
        ),
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    loss_payload = config.get("loss", {})
    class_weight_payload = loss_payload.get("edit_class_weights", "auto")
    if class_weight_payload == "auto":
        class_weights = _edit_class_weights(
            train_samples, power=float(loss_payload.get("edit_class_weight_power", 0.5))
        )
    elif class_weight_payload is None:
        class_weights = None
    else:
        class_weights = torch.tensor(class_weight_payload, dtype=torch.float32)
        if class_weights.numel() != model_config.edit_classes:
            raise ValueError("loss.edit_class_weights must match model.edit_classes")
    if class_weights is not None:
        class_weights = class_weights.to(device)
    loss_settings: dict[str, Any] = {
        "segmentation": float(loss_payload.get("segmentation", 1.0)),
        "segmentation_bce": float(loss_payload.get("segmentation_bce", 1.0)),
        "segmentation_dice": float(loss_payload.get("segmentation_dice", 1.0)),
        "segmentation_focal": float(loss_payload.get("segmentation_focal", 0.25)),
        "segmentation_cldice": float(loss_payload.get("segmentation_cldice", 0.0)),
        "cldice_iterations": int(loss_payload.get("cldice_iterations", 5)),
        "focal_gamma": float(loss_payload.get("focal_gamma", 2.0)),
        "focal_alpha": float(loss_payload.get("focal_alpha", 0.75)),
        "edit": float(loss_payload.get("edit", 1.0)),
        "edit_class_weights": class_weights,
        "edit_label_smoothing": float(loss_payload.get("edit_label_smoothing", 0.05)),
        "geometry": float(loss_payload.get("geometry", 0.5)),
        "geometry_beta": float(loss_payload.get("geometry_beta", 0.1)),
        "false_edit": float(loss_payload.get("false_edit", 0.5)),
        "missed_edit": float(loss_payload.get("missed_edit", 0.25)),
        "confidence": float(loss_payload.get("confidence", 0.2)),
        "confidence_target_mode": str(loss_payload.get("confidence_target_mode", "mean")),
        "presence": float(loss_payload.get("presence", 0.0)),
        "change": float(loss_payload.get("change", 0.0)),
        "temporal_change": float(loss_payload.get("temporal_change", 0.0)),
        "temporal_change_bce": float(loss_payload.get("temporal_change_bce", 1.0)),
        "temporal_change_dice": float(loss_payload.get("temporal_change_dice", 1.0)),
        "temporal_change_focal": float(loss_payload.get("temporal_change_focal", 0.5)),
        "temporal_change_focal_alpha": float(loss_payload.get("temporal_change_focal_alpha", 0.9)),
    }
    max_epochs = int(training.get("max_epochs", training.get("epochs", 30)))
    min_epochs = int(training.get("min_epochs", 0))
    roi_warmup_epochs = int(training.get("roi_warmup_epochs", 0))
    evidence_warmup_epochs = int(training.get("evidence_warmup_epochs", 0))
    if roi_warmup_epochs < 0:
        raise ValueError("training.roi_warmup_epochs must be non-negative")
    if evidence_warmup_epochs < 0:
        raise ValueError("training.evidence_warmup_epochs must be non-negative")
    if roi_warmup_epochs > 0 and evidence_warmup_epochs > 0:
        raise ValueError("ROI and evidence warm-up cannot both be enabled")
    if roi_warmup_epochs > 0 and not model_config.prior_guided_roi:
        raise ValueError("training.roi_warmup_epochs requires model.prior_guided_roi=true")
    if evidence_warmup_epochs > 0 and not (
        model_config.segmentation_evidence
        or model_config.temporal_change_to_edit_head
        or model_config.temporal_spatial_edit_head
    ):
        raise ValueError("training.evidence_warmup_epochs requires an enabled evidence head")
    warmup_epochs = max(roi_warmup_epochs, evidence_warmup_epochs)
    joint_learning_rate_value = training.get("joint_learning_rate")
    joint_learning_rate = (
        float(joint_learning_rate_value) if joint_learning_rate_value is not None else None
    )
    early_stopping = training.get("early_stopping", {})
    patience = int(early_stopping.get("patience", training.get("patience", 7)))
    min_delta = float(early_stopping.get("min_delta", 1e-4))
    grad_clip = float(training.get("grad_clip", 1.0))
    gradient_accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps < 1:
        raise ValueError("training.gradient_accumulation_steps must be positive")
    use_amp = bool(training.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler_payload = training.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_payload.get("factor", 0.5)),
        patience=int(scheduler_payload.get("patience", 3)),
        threshold=float(scheduler_payload.get("threshold", min_delta)),
        threshold_mode="abs",
        cooldown=int(scheduler_payload.get("cooldown", 0)),
        min_lr=float(scheduler_payload.get("min_lr", 1e-6)),
    )
    checkpoint_every = int(training.get("checkpoint_every", 5))
    best_val_loss = float("inf")
    best_quality_score = -float("inf")
    best_tradeoff_score = -float("inf")
    best_safety_score = -float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    monitor = RunMonitor(output_dir, config.get("monitoring", {}))
    resume = bool(training.get("resume", False))
    start_epoch = 1
    last_checkpoint = output_dir / "last.pt"
    initialization: dict[str, Any] | None = None
    if resume and last_checkpoint.is_file():
        checkpoint = torch.load(last_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        best_quality_score = float(checkpoint.get("best_quality_score", -float("inf")))
        best_tradeoff_score = float(
            checkpoint.get(
                "best_tradeoff_score",
                checkpoint.get("best_safety_score", -float("inf")),
            )
        )
        best_safety_score = float(
            checkpoint.get("best_safety_score", checkpoint.get("best_score", -float("inf")))
        )
        _, active_safety_limits = _safety_eligible(
            {"false_edit_rate": 0.0, "missed_edit_rate": 0.0}, training
        )
        if checkpoint.get("safety_limits", {}) != active_safety_limits:
            best_safety_score = -float("inf")
        stale = int(checkpoint["stale"])
        history = list(checkpoint.get("history", []))
        initialization = checkpoint.get("initialization")
    elif training.get("init_checkpoint"):
        init_checkpoint = Path(str(training["init_checkpoint"]))
        if not init_checkpoint.is_absolute():
            init_checkpoint = (config_path.parent / init_checkpoint).resolve()
        if not init_checkpoint.is_file():
            raise FileNotFoundError(f"initialization checkpoint not found: {init_checkpoint}")
        initialization = initialize_updater_weights(
            model,
            init_checkpoint,
            scope=str(training.get("init_scope", "full")),
            device=device,
        )
    elif not resume and monitor.history_path.exists():
        monitor.history_path.unlink()
    monitor.write_state(
        "running",
        epoch=start_epoch - 1,
        total_epochs=max_epochs,
        min_epochs=min_epochs,
        resumed=resume and start_epoch > 1,
        initialization=initialization,
    )
    monitoring = config.get("monitoring", {})
    visualize_every = int(monitoring.get("visualize_every", 1))
    visualization_samples = int(monitoring.get("visualization_samples", 6))
    show_progress = bool(monitoring.get("progress_bar", True))
    stopped = False

    for epoch in range(start_epoch, max_epochs + 1):
        if monitor.wait_if_paused(epoch=epoch) or monitor.should_stop():
            stopped = True
            monitor.write_state("stopped", epoch=epoch - 1, total_epochs=max_epochs)
            break
        roi_only = roi_warmup_epochs > 0 and epoch <= roi_warmup_epochs
        evidence_only = evidence_warmup_epochs > 0 and epoch <= evidence_warmup_epochs
        trainable_parameter_count = set_updater_trainable_scope(
            model, roi_only=roi_only, evidence_only=evidence_only
        )
        if epoch == warmup_epochs + 1 and joint_learning_rate is not None:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = joint_learning_rate
        if roi_only:
            trainable_scope = "roi_only"
        elif evidence_only:
            trainable_scope = "evidence_only"
        else:
            trainable_scope = "all"
        monitor.write_state(
            "running",
            epoch=epoch,
            total_epochs=max_epochs,
            phase="train",
            trainable_scope=trainable_scope,
            trainable_parameter_count=trainable_parameter_count,
        )
        train_metrics = run_updater_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            loss_settings=loss_settings,
            grad_clip=grad_clip,
            scaler=scaler,
            use_amp=use_amp,
            epoch=epoch,
            total_epochs=max_epochs,
            phase="train",
            show_progress=show_progress,
            gradient_accumulation_steps=gradient_accumulation_steps,
            frozen_backbone=roi_only or evidence_only,
        )
        with torch.no_grad():
            monitor.write_state("running", epoch=epoch, total_epochs=max_epochs, phase="validation")
            val_metrics = run_updater_epoch(
                model,
                val_loader,
                device=device,
                optimizer=None,
                loss_settings=loss_settings,
                grad_clip=grad_clip,
                use_amp=use_amp,
                epoch=epoch,
                total_epochs=max_epochs,
                phase="validation",
                show_progress=show_progress,
            )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_metrics["loss"])
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": epoch,
            "trainable_scope": trainable_scope,
            "trainable_parameter_count": trainable_parameter_count,
            "learning_rate": learning_rate,
            "next_learning_rate": next_learning_rate,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        monitor.record_epoch(record)
        print(
            f"epoch={epoch}/{max_epochs} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_iou={val_metrics['iou']:.6f} "
            f"val_added_iou={val_metrics['added_change_iou']:.6f} "
            f"val_removed_iou={val_metrics['removed_change_iou']:.6f} "
            f"val_edit_accuracy={val_metrics['edit_accuracy']:.6f} "
            f"lr={learning_rate:.8g} next_lr={next_learning_rate:.8g}",
            flush=True,
        )
        quality_score, quality_score_mode = _quality_score(val_metrics, training)
        tradeoff_score = quality_score - val_metrics["false_edit_rate"]
        safety_eligible, safety_limits = _safety_eligible(val_metrics, training)
        safety_score = tradeoff_score if safety_eligible else -float("inf")
        improved_loss = val_metrics["loss"] < best_val_loss - min_delta
        checkpoint = {
            "state_dict": model.state_dict(),
            "model_config": model_config.as_dict(),
            "seed": seed,
            "epoch": epoch,
            "val_metrics": val_metrics,
            "learning_rate": learning_rate,
            "quality_score": quality_score,
            "quality_score_mode": quality_score_mode,
            "tradeoff_score": tradeoff_score,
            "safety_eligible": safety_eligible,
            "safety_limits": safety_limits,
            "edit_class_weights": (
                class_weights.detach().cpu().tolist() if class_weights is not None else None
            ),
        }
        if improved_loss:
            best_val_loss = val_metrics["loss"]
            stale = 0
            save_checkpoint(output_dir / "best.pt", checkpoint)
            save_checkpoint(output_dir / "best_val_loss.pt", checkpoint)
        else:
            stale += 1
        if quality_score > best_quality_score:
            best_quality_score = quality_score
            save_checkpoint(output_dir / "best_quality.pt", checkpoint)
        if tradeoff_score > best_tradeoff_score:
            best_tradeoff_score = tradeoff_score
            save_checkpoint(output_dir / "best_tradeoff.pt", checkpoint)
        if safety_score > best_safety_score:
            best_safety_score = safety_score
            save_checkpoint(output_dir / "best_safety.pt", checkpoint)
        latest_checkpoint = {
            **checkpoint,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_quality_score": best_quality_score,
            "best_tradeoff_score": best_tradeoff_score,
            "best_safety_score": best_safety_score,
            "initialization": initialization,
            "stale": stale,
            "history": history,
        }
        save_checkpoint(last_checkpoint, latest_checkpoint)
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            save_checkpoint(output_dir / "checkpoints" / f"epoch_{epoch:04d}.pt", checkpoint)
        if visualize_every > 0 and (epoch == start_epoch or epoch % visualize_every == 0):
            monitor.write_state(
                "running", epoch=epoch, total_epochs=max_epochs, phase="visualization"
            )
            render_updater_progress(
                model,
                val_samples,
                device=device,
                output_path=output_dir / "visualizations" / f"epoch_{epoch:04d}.png",
                count=visualization_samples,
            )
        monitor.write_state(
            "running",
            epoch=epoch,
            total_epochs=max_epochs,
            phase="idle",
            stale_epochs=stale,
            best_val_loss=best_val_loss,
            learning_rate=next_learning_rate,
            latest_metrics=val_metrics,
        )
        if monitor.should_stop():
            stopped = True
            monitor.write_state("stopped", epoch=epoch, total_epochs=max_epochs)
            break
        if epoch >= min_epochs and stale >= patience:
            break

    summary = {
        "config": str(config_path.resolve()),
        "samples": str(samples_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "parameter_count": model.parameter_count(),
        "edit_class_weights": (
            class_weights.detach().cpu().tolist() if class_weights is not None else None
        ),
        "max_epochs": max_epochs,
        "min_epochs": min_epochs,
        "roi_warmup_epochs": roi_warmup_epochs,
        "evidence_warmup_epochs": evidence_warmup_epochs,
        "joint_learning_rate": joint_learning_rate,
        "early_stopping_monitor": "val/loss",
        "early_stopping_patience": patience,
        "early_stopping_min_delta": min_delta,
        "best_val_loss": best_val_loss,
        "best_quality_score": best_quality_score,
        "best_tradeoff_score": best_tradeoff_score,
        "best_safety_score": best_safety_score,
        "safety_limits": _safety_eligible(
            {"false_edit_rate": 0.0, "missed_edit_rate": 0.0}, training
        )[1],
        "amp": use_amp,
        "epochs_completed": len(history),
        "initialization": initialization,
        "stopped_by_user": stopped,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not stopped:
        converged = bool(history and history[-1]["epoch"] >= min_epochs and stale >= patience)
        monitor.write_state(
            "completed",
            epoch=history[-1]["epoch"] if history else start_epoch - 1,
            total_epochs=max_epochs,
            converged=converged,
        )
    monitor.close()
    archive_value = config.get("archive_dir")
    if archive_value:
        archive_dir = Path(str(archive_value))
        if not archive_dir.is_absolute():
            archive_dir = (config_path.parent / archive_dir).resolve()
        archive_updater_run(output_dir, archive_dir)
    return summary
