#!/usr/bin/env python3
"""Train prior-conditioned current/add/remove heads on MUNO21."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from activemap.integrations.prior_conditioned_sam_road import (
    PriorConditionedChangeHead,
    change_targets,
    prior_conditioned_change_loss,
)
from activemap.integrations.sam_road import SAMRoadPredictor
from activemap.integrations.sam_road_training import (
    MUNO21RoadDataset,
    ThresholdMetrics,
    load_road_records,
    split_support,
)
from activemap.models import EditOperation
from activemap.updater_records import load_updater_samples

EDIT_ORDER = list(EditOperation)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _provenance_file(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"missing provenance file: {path}")
    return {"path": str(path), "sha256": _sha256(path)}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def _image_embeddings(base_model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    inputs = (image - base_model.pixel_mean) / base_model.pixel_std
    with torch.no_grad():
        return base_model.image_encoder(inputs)


def _operation_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    keep = EDIT_ORDER.index(EditOperation.KEEP)
    keep_count = max(int(np.sum(target == keep)), 1)
    update_count = max(int(np.sum(target != keep)), 1)
    matrix = np.zeros((len(EDIT_ORDER), len(EDIT_ORDER)), dtype=np.int64)
    np.add.at(matrix, (target, prediction), 1)
    true_positive = np.diag(matrix).astype(np.float64)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    per_class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return {
        "accuracy": float(np.mean(target == prediction)),
        "macro_f1": float(per_class_f1.mean()),
        "false_edit_rate": float(
            np.sum((target == keep) & (prediction != keep)) / keep_count
        ),
        "missed_edit_rate": float(
            np.sum((target != keep) & (prediction == keep)) / update_count
        ),
        "confusion_matrix": matrix.tolist(),
    }


def _operation_prediction(probability: np.ndarray, commit_threshold: float) -> np.ndarray:
    keep = EDIT_ORDER.index(EditOperation.KEEP)
    nonkeep_probability = probability[:, 1:]
    prediction = nonkeep_probability.argmax(axis=1) + 1
    best_nonkeep = nonkeep_probability.max(axis=1)
    commit = (best_nonkeep >= commit_threshold) & (
        best_nonkeep > probability[:, keep]
    )
    prediction[~commit] = keep
    return prediction


def _select_operation_point(
    target: np.ndarray,
    probability: np.ndarray,
    commit_thresholds: tuple[float, ...],
    *,
    max_false_edit: float,
) -> dict[str, Any]:
    grid = []
    for commit_threshold in commit_thresholds:
        metrics = _operation_metrics(
            target,
            _operation_prediction(probability, commit_threshold),
        )
        grid.append({"commit_threshold": commit_threshold, "metrics": metrics})
    selected = max(
        grid,
        key=lambda row: (
            row["metrics"]["false_edit_rate"] <= max_false_edit,
            row["metrics"]["macro_f1"],
            -row["metrics"]["false_edit_rate"],
            -row["commit_threshold"],
        ),
    )
    return {"selected": selected, "grid": grid, "max_false_edit": max_false_edit}


def _positive_weights(
    dataset: MUNO21RoadDataset,
    maximum: float,
) -> tuple[tuple[float, ...], dict[str, float]]:
    positive = np.zeros(3, dtype=np.int64)
    total = 0
    for index in tqdm(range(len(dataset)), desc="measure change labels", unit="image"):
        row = dataset[index]
        current, add, remove = change_targets(row["mask"], row["prior"])
        valid = row["valid"] > 0.5
        for channel, value in enumerate((current, add, remove)):
            positive[channel] += int(value[valid].sum().item())
        total += int(valid.sum().item())
    weights = tuple(
        max(1.0, min(maximum, (total - count) / max(count, 1))) for count in positive
    )
    fractions = {
        name: float(count / max(total, 1))
        for name, count in zip(("current", "add", "remove"), positive, strict=True)
    }
    return weights, fractions


def _operation_weights(records: list[Any]) -> tuple[float, ...]:
    counts = Counter(EDIT_ORDER.index(EditOperation(record.edit_type)) for record in records)
    weights = np.asarray(
        [1.0 / np.sqrt(max(counts[index], 1)) for index in range(len(EDIT_ORDER))],
        dtype=np.float64,
    )
    weights /= weights.mean()
    return tuple(float(value) for value in weights)


def _balanced_sampling_weights(records: list[Any]) -> list[float]:
    labels = [EDIT_ORDER.index(EditOperation(record.edit_type)) for record in records]
    counts = Counter(labels)
    return [1.0 / counts[label] for label in labels]


def _run_epoch(
    loader: DataLoader,
    base_model: nn.Module,
    head: nn.Module,
    *,
    device: torch.device,
    positive_weights: tuple[float, float, float],
    operation_weights: tuple[float, ...],
    positive_only_change_dice: bool,
    optimizer: torch.optim.Optimizer | None,
    thresholds: tuple[float, ...],
    commit_thresholds: tuple[float, ...],
    max_false_edit: float,
    description: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    training = optimizer is not None
    base_model.eval()
    head.train(training)
    sums: Counter[str] = Counter()
    channel_metrics = [ThresholdMetrics(thresholds) for _ in range(3)]
    operation_target = []
    operation_probability = []
    previews = []
    count = 0
    iterator = tqdm(loader, desc=description, unit="batch", leave=False)
    for batch in iterator:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        prior = batch["prior"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        operation_target_tensor = torch.tensor(
            [EDIT_ORDER.index(EditOperation(value)) for value in batch["edit_type"]],
            device=device,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            embeddings = _image_embeddings(base_model, image)
            output = head(embeddings, prior[:, None])
            logits = output["change_logits"]
            loss, components = prior_conditioned_change_loss(
                logits,
                output["operation_logits"],
                operation_target_tensor,
                target,
                prior,
                valid,
                positive_weights=positive_weights,
                operation_weights=torch.as_tensor(
                    operation_weights,
                    device=device,
                    dtype=logits.dtype,
                ),
                positive_only_change_dice=positive_only_change_dice,
            )
            if training:
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                optimizer.step()
            else:
                gradient_norm = torch.zeros((), device=device)
        batch_size = image.shape[0]
        for name, value in components.items():
            sums[name] += float(value.item()) * batch_size
        sums["gradient_norm"] += float(gradient_norm) * batch_size
        probability = torch.sigmoid(logits).detach().cpu().numpy()
        current, add, remove = change_targets(target, prior)
        targets = torch.stack([current, add, remove], dim=1).cpu().numpy()
        valid_np = valid.cpu().numpy().astype(bool)
        for channel in range(3):
            channel_metrics[channel].update(
                np.where(valid_np, probability[:, channel], 0.0),
                np.where(valid_np, targets[:, channel], 0.0),
            )
        operation_target.extend(
            EDIT_ORDER.index(EditOperation(value)) for value in batch["edit_type"]
        )
        operation_probability.extend(
            torch.softmax(output["operation_logits"], dim=1).detach().cpu().tolist()
        )
        if len(previews) < 8:
            for item_index in range(min(batch_size, 8 - len(previews))):
                previews.append(
                    {
                        "image": image[item_index].detach().cpu().numpy(),
                        "prior": prior[item_index].detach().cpu().numpy(),
                        "target": target[item_index].detach().cpu().numpy(),
                        "probability": probability[item_index],
                        "operation_probability": operation_probability[-batch_size + item_index],
                        "edit_type": batch["edit_type"][item_index],
                        "aoi_id": batch["aoi_id"][item_index],
                    }
                )
        count += batch_size
        iterator.set_postfix(loss=f"{loss.item():.4f}")

    operation = _select_operation_point(
        np.asarray(operation_target, dtype=np.int64),
        np.asarray(operation_probability, dtype=np.float64),
        commit_thresholds,
        max_false_edit=max_false_edit,
    )
    summary = {
        "images": count,
        "losses": {name: value / count for name, value in sorted(sums.items())},
        "channels": {
            name: {"best": metric.best(), "threshold_sweep": metric.summary()}
            for name, metric in zip(
                ("current", "add", "remove"), channel_metrics, strict=True
            )
        },
        "operation": operation,
    }
    return summary, previews


def _evaluate_fixed_operation(
    summary: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, Any]:
    commit_threshold = float(selected["commit_threshold"])
    matches = [
        row
        for row in summary["operation"]["grid"]
        if abs(float(row["commit_threshold"]) - commit_threshold) < 1e-8
    ]
    if len(matches) != 1:
        raise ValueError("selected operation point is unavailable in evaluation grid")
    return matches[0]


def _fixed_channel_metrics(
    summary: dict[str, Any],
    selected_thresholds: dict[str, float],
) -> dict[str, Any]:
    fixed = {}
    for name, threshold in selected_thresholds.items():
        matches = [
            row
            for row in summary["channels"][name]["threshold_sweep"]
            if abs(float(row["threshold"]) - threshold) < 1e-8
        ]
        if len(matches) != 1:
            raise ValueError(f"selected {name} threshold is unavailable")
        fixed[name] = matches[0]
    return fixed


def _write_qc(
    path: Path,
    previews: list[dict[str, Any]],
    operation_point: dict[str, Any],
    channel_thresholds: dict[str, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    current_threshold = channel_thresholds["current"]
    add_threshold = channel_thresholds["add"]
    remove_threshold = channel_thresholds["remove"]
    commit_threshold = float(operation_point["commit_threshold"])
    count = min(8, len(previews))
    figure, axes = plt.subplots(count, 6, figsize=(18, 3.2 * count), squeeze=False)
    titles = ("RGB", "Prior", "Target", "Current", "Target change", "Predicted change")
    for row_index, preview in enumerate(previews[:count]):
        target = preview["target"] >= 0.5
        prior = preview["prior"] >= 0.5
        target_change = np.zeros((*target.shape, 3), dtype=np.float32)
        target_change[..., 0] = np.logical_and(prior, ~target)
        target_change[..., 1] = np.logical_and(target, ~prior)
        prediction_change = np.zeros_like(target_change)
        prediction_change[..., 0] = preview["probability"][2] >= remove_threshold
        prediction_change[..., 1] = preview["probability"][1] >= add_threshold
        values = (
            np.clip(preview["image"].transpose(1, 2, 0), 0, 255).astype(np.uint8),
            prior,
            target,
            preview["probability"][0] >= current_threshold,
            target_change,
            prediction_change,
        )
        for column, (title, value) in enumerate(zip(titles, values, strict=True)):
            axes[row_index, column].imshow(value, cmap="gray" if value.ndim == 2 else None)
            axes[row_index, column].axis("off")
            if row_index == 0:
                axes[row_index, column].set_title(title)
        axes[row_index, 0].set_ylabel(
            f"{preview['aoi_id']}\n{preview['edit_type']}\n"
            f"op={np.round(preview['operation_probability'], 2)}",
            fontsize=8,
        )
    figure.suptitle(
        "Prior-conditioned MUNO21 change head "
        f"(red=remove, green=add, commit>={commit_threshold:.2f})",
        y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.995))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("updater_manifest", type=Path)
    parser.add_argument("sam_road_repo", type=Path)
    parser.add_argument("sam_road_config", type=Path)
    parser.add_argument("source_checkpoint", type=Path)
    parser.add_argument("sam_checkpoint", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--calibration-aoi", default="atlanta")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--decoder-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-positive-weight", type=float, default=30.0)
    parser.add_argument("--max-false-edit", type=float, default=0.10)
    parser.add_argument("--positive-only-change-dice", action="store_true")
    parser.add_argument(
        "--change-parameterization",
        choices=("independent", "current_difference"),
        default="independent",
    )
    parser.add_argument(
        "--operation-head",
        choices=("global_stats", "spatial_pyramid"),
        default="global_stats",
    )
    parser.add_argument(
        "--probability-thresholds", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80"
    )
    parser.add_argument(
        "--commit-thresholds", default="0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--operation-only-from", type=Path)
    parser.add_argument("--minimum-calibration-macro-f1", type=float)
    parser.add_argument("--code-receipt", type=Path)
    parser.add_argument("--data-inventory", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.resume is not None and args.operation_only_from is not None:
        parser.error("--resume and --operation-only-from are mutually exclusive")
    if args.output_root.exists() and args.resume is None:
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    probability_thresholds = tuple(
        float(value) for value in args.probability_thresholds.split(",")
    )
    commit_thresholds = tuple(
        float(value) for value in args.commit_thresholds.split(",")
    )
    _seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    samples = load_updater_samples(args.updater_manifest)
    prior_paths = {row.sample_id: Path(row.prior_mask_path) for row in samples}
    valid_paths = {
        row.sample_id: Path(row.valid_mask_path)
        for row in samples
        if row.valid_mask_path is not None
    }
    train_annotations = args.dataset_root / "annotations" / "train.json"
    target_annotations = args.dataset_root / "annotations" / "val.json"
    train_records = load_road_records(
        train_annotations,
        args.dataset_root / "images" / "train",
        exclude_aois={args.calibration_aoi},
    )
    calibration_records = load_road_records(
        train_annotations,
        args.dataset_root / "images" / "train",
        include_aois={args.calibration_aoi},
    )
    target_records = load_road_records(
        target_annotations, args.dataset_root / "images" / "val"
    )
    if args.smoke:
        train_records = train_records[:16]
        calibration_records = calibration_records[:8]
        target_records = target_records[:8]
        args.max_epochs = min(args.max_epochs, 1)
    args.output_root.mkdir(parents=True, exist_ok=args.resume is not None)
    (args.output_root / "checkpoints").mkdir(exist_ok=True)

    commit = subprocess.check_output(
        ["git", "-C", str(args.sam_road_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    predictor = SAMRoadPredictor(
        args.sam_road_repo,
        args.sam_road_config,
        args.source_checkpoint,
        args.sam_checkpoint,
        device=str(device),
        upstream_commit=commit,
    )
    base_model = predictor.model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad = False
    head = PriorConditionedChangeHead.build(
        base_model.map_decoder,
        parameterization=args.change_parameterization,
        operation_head=args.operation_head,
    ).to(device)
    operation_source_load = None
    if args.operation_only_from is not None:
        operation_source = torch.load(args.operation_only_from, map_location="cpu")
        source_state = operation_source["head_state_dict"]
        if args.operation_head == "spatial_pyramid":
            source_state = {
                name: value
                for name, value in source_state.items()
                if not name.startswith("operation_classifier.")
            }
            incompatible = head.load_state_dict(source_state, strict=False)
            operation_source_load = {
                "classifier_reinitialized": True,
                "missing_keys": incompatible.missing_keys,
                "unexpected_keys": incompatible.unexpected_keys,
            }
        else:
            head.load_state_dict(source_state)
            operation_source_load = {
                "classifier_reinitialized": False,
                "missing_keys": [],
                "unexpected_keys": [],
            }
        for parameter in head.parameters():
            parameter.requires_grad = False
        for parameter in head.operation_classifier.parameters():
            parameter.requires_grad = True

    def dataset(records: Any, augment: bool) -> MUNO21RoadDataset:
        return MUNO21RoadDataset(
            records,
            patch_size=predictor.patch_size,
            augment=augment,
            prior_paths=prior_paths,
            valid_paths=valid_paths,
        )

    train_dataset = dataset(train_records, True)
    plain_train_dataset = dataset(train_records, False)
    calibration_dataset = dataset(calibration_records, False)
    target_dataset = dataset(target_records, False)
    positive_weights, positive_fractions = _positive_weights(
        plain_train_dataset, args.max_positive_weight
    )
    operation_weights = (
        tuple(1.0 for _ in EDIT_ORDER)
        if args.operation_only_from is not None
        else _operation_weights(train_records)
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _worker_seed,
    }
    generator = torch.Generator().manual_seed(args.seed)
    if args.operation_only_from is not None:
        sampler = WeightedRandomSampler(
            _balanced_sampling_weights(train_records),
            num_samples=len(train_records),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(train_dataset, sampler=sampler, **loader_options)
    else:
        train_loader = DataLoader(
            train_dataset, shuffle=True, generator=generator, **loader_options
        )
    calibration_loader = DataLoader(calibration_dataset, shuffle=False, **loader_options)
    target_loader = DataLoader(target_dataset, shuffle=False, **loader_options)

    config = {
        "schema_version": "muno21-prior-conditioned-change-v2",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "protocol": {
            "training": split_support(train_records),
            "calibration": split_support(calibration_records),
            "target_validation": split_support(target_records),
            "frozen_test_access": False,
        },
        "model": {
            "source_checkpoint_sha256": _sha256(args.source_checkpoint),
            "sam_checkpoint_sha256": _sha256(args.sam_checkpoint),
            "operation_source_checkpoint": _provenance_file(
                args.operation_only_from
            ),
            "operation_source_load": operation_source_load,
            "upstream_commit": commit,
            "trainable_parameters": sum(
                p.numel() for p in head.parameters() if p.requires_grad
            ),
            "encoder_frozen": True,
            "operation_only": args.operation_only_from is not None,
        },
        "labels": {
            "positive_weights": positive_weights,
            "positive_fractions": positive_fractions,
            "operation_weights": operation_weights,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "device": str(device),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "provenance": {
            "code_receipt": _provenance_file(args.code_receipt),
            "data_inventory": _provenance_file(args.data_inventory),
        },
    }
    (args.output_root / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    parameter_groups = (
        [{"params": head.operation_classifier.parameters(), "lr": args.learning_rate}]
        if args.operation_only_from is not None
        else [
            {
                "params": [
                    *head.embedding_projection.parameters(),
                    *head.feature_extractor.parameters(),
                    *head.change_projection.parameters(),
                    *head.operation_classifier.parameters(),
                ],
                "lr": args.learning_rate,
            },
            {"params": head.road_decoder.parameters(), "lr": args.decoder_learning_rate},
        ]
    )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    start_epoch = 1
    best_epoch = 0
    best_score = -1.0
    best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    best_operation = None
    stale_epochs = 0
    if args.resume is not None:
        resume = torch.load(args.resume, map_location="cpu")
        head.load_state_dict(resume["head_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        best_epoch = int(resume["best_epoch"])
        best_score = float(resume["best_score"])
        best_state = resume["best_state_dict"]
        best_operation = resume["best_operation"]
        stale_epochs = int(resume["stale_epochs"])

    history_path = args.output_root / "history.jsonl"
    with history_path.open("a" if args.resume else "x", encoding="utf-8", buffering=1) as history:
        for epoch in range(start_epoch, args.max_epochs + 1):
            train_summary, _ = _run_epoch(
                train_loader,
                base_model,
                head,
                device=device,
                positive_weights=positive_weights,
                operation_weights=operation_weights,
                positive_only_change_dice=args.positive_only_change_dice,
                optimizer=optimizer,
                thresholds=probability_thresholds,
                commit_thresholds=commit_thresholds,
                max_false_edit=args.max_false_edit,
                description=f"epoch {epoch:03d}/{args.max_epochs:03d} train",
            )
            calibration_summary, _ = _run_epoch(
                calibration_loader,
                base_model,
                head,
                device=device,
                positive_weights=positive_weights,
                operation_weights=operation_weights,
                positive_only_change_dice=args.positive_only_change_dice,
                optimizer=None,
                thresholds=probability_thresholds,
                commit_thresholds=commit_thresholds,
                max_false_edit=args.max_false_edit,
                description=f"epoch {epoch:03d} calibration",
            )
            selected = calibration_summary["operation"]["selected"]
            score = float(selected["metrics"]["macro_f1"])
            safe = selected["metrics"]["false_edit_rate"] <= args.max_false_edit
            selection_score = score if safe else score - selected["metrics"]["false_edit_rate"]
            scheduler.step(selection_score)
            improved = selection_score > best_score + 1e-6
            if improved:
                best_epoch = epoch
                best_score = selection_score
                best_operation = selected
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in head.state_dict().items()
                }
                stale_epochs = 0
                torch.save(
                    {
                        "schema_version": "muno21-prior-conditioned-head-v2",
                        "epoch": epoch,
                        "operation_point": selected,
                        "head_config": {
                            "parameterization": args.change_parameterization,
                            "operation_head": args.operation_head,
                        },
                        "loss_config": {
                            "positive_only_change_dice": args.positive_only_change_dice,
                        },
                        "head_state_dict": best_state,
                        "source_checkpoint_sha256": config["model"][
                            "source_checkpoint_sha256"
                        ],
                        "frozen_test_access": False,
                    },
                    args.output_root / "checkpoints" / "best_head.pt",
                )
            else:
                stale_epochs += 1
            row = {
                "epoch": epoch,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "train": train_summary,
                "calibration": calibration_summary,
                "selection_score": selection_score,
                "improved": improved,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "best_operation": best_operation,
                "stale_epochs": stale_epochs,
            }
            history.write(json.dumps(row) + "\n")
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": train_summary["losses"]["total"],
                        "calibration_loss": calibration_summary["losses"]["total"],
                        "operation_macro_f1": selected["metrics"]["macro_f1"],
                        "false_edit_rate": selected["metrics"]["false_edit_rate"],
                        "missed_edit_rate": selected["metrics"]["missed_edit_rate"],
                        "commit_threshold": selected["commit_threshold"],
                        "best_epoch": best_epoch,
                        "stale_epochs": stale_epochs,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            torch.save(
                {
                    **row,
                    "head_state_dict": {
                        name: value.detach().cpu() for name, value in head.state_dict().items()
                    },
                    "best_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                args.output_root / "checkpoints" / "last.pt",
            )
            if stale_epochs >= args.patience:
                print(f"early stopping after {stale_epochs} stale epochs", flush=True)
                break

    if best_operation is None:
        raise RuntimeError("training produced no selected checkpoint")
    head.load_state_dict(best_state)
    calibration_final, _ = _run_epoch(
        calibration_loader,
        base_model,
        head,
        device=device,
        positive_weights=positive_weights,
        operation_weights=operation_weights,
        positive_only_change_dice=args.positive_only_change_dice,
        optimizer=None,
        thresholds=probability_thresholds,
        commit_thresholds=commit_thresholds,
        max_false_edit=args.max_false_edit,
        description="frozen calibration",
    )
    calibration_fixed = _evaluate_fixed_operation(calibration_final, best_operation)
    calibration_macro_f1 = float(calibration_fixed["metrics"]["macro_f1"])
    minimum_calibration = args.minimum_calibration_macro_f1
    if minimum_calibration is not None and calibration_macro_f1 < minimum_calibration:
        summary = {
            "schema_version": "muno21-prior-conditioned-change-result-v2",
            "best_epoch": best_epoch,
            "best_score": best_score,
            "selected_operation_point": best_operation,
            "calibration": calibration_final,
            "calibration_at_selected_operation_point": calibration_fixed,
            "target_validation": None,
            "target_gate": {
                "passed": False,
                "minimum_macro_f1": minimum_calibration,
                "observed_macro_f1": calibration_macro_f1,
            },
            "checkpoint": str(args.output_root / "checkpoints" / "best_head.pt"),
            "frozen_test_access": False,
        }
        (args.output_root / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "best_epoch": best_epoch,
                    "calibration_macro_f1": calibration_macro_f1,
                    "target_gate": "failed",
                    "summary": str(args.output_root / "summary.json"),
                    "frozen_test_access": False,
                },
                sort_keys=True,
            )
        )
        return
    target_final, target_previews = _run_epoch(
        target_loader,
        base_model,
        head,
        device=device,
        positive_weights=positive_weights,
        operation_weights=operation_weights,
        positive_only_change_dice=args.positive_only_change_dice,
        optimizer=None,
        thresholds=probability_thresholds,
        commit_thresholds=commit_thresholds,
        max_false_edit=args.max_false_edit,
        description="frozen target-val",
    )
    target_fixed = _evaluate_fixed_operation(target_final, best_operation)
    selected_channel_thresholds = {
        name: float(channel["best"]["threshold"])
        for name, channel in calibration_final["channels"].items()
    }
    target_fixed_channels = _fixed_channel_metrics(
        target_final, selected_channel_thresholds
    )
    _write_qc(
        args.output_root / "qc_target_val.png",
        target_previews,
        best_operation,
        selected_channel_thresholds,
    )
    summary = {
        "schema_version": "muno21-prior-conditioned-change-result-v2",
        "best_epoch": best_epoch,
        "best_score": best_score,
        "selected_operation_point": best_operation,
        "calibration": calibration_final,
        "calibration_at_selected_operation_point": calibration_fixed,
        "target_validation": target_final,
        "target_gate": {
            "passed": True,
            "minimum_macro_f1": minimum_calibration,
            "observed_macro_f1": calibration_macro_f1,
        },
        "target_at_selected_operation_point": target_fixed,
        "selected_channel_thresholds": selected_channel_thresholds,
        "target_channels_at_calibration_thresholds": target_fixed_channels,
        "checkpoint": str(args.output_root / "checkpoints" / "best_head.pt"),
        "frozen_test_access": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_score": best_score,
                "target_at_selected_operation_point": target_fixed,
                "summary": str(args.output_root / "summary.json"),
                "frozen_test_access": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
