#!/usr/bin/env python3
"""Adapt the official SAM-Road road decoder to test-free MUNO21 supervision."""

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
from torch.utils.data import DataLoader
from tqdm import tqdm

from activemap.integrations.sam_road import SAMRoadPredictor
from activemap.integrations.sam_road_training import (
    MUNO21RoadDataset,
    ThresholdMetrics,
    load_road_records,
    road_bce_dice_loss,
    split_support,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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


def _road_logits(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Run a frozen image encoder and differentiable road decoder."""

    inputs = (image - model.pixel_mean) / model.pixel_std
    with torch.no_grad():
        embeddings = model.image_encoder(inputs)
    logits = model.map_decoder(embeddings)
    return logits[:, 1]


def _trainable_decoder(model: nn.Module) -> int:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if not hasattr(model, "map_decoder"):
        raise ValueError("road-head adaptation requires SAM-Road's naive map_decoder")
    for parameter in model.map_decoder.parameters():
        parameter.requires_grad = True
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _foreground_weight(dataset: MUNO21RoadDataset, maximum: float) -> tuple[float, float]:
    foreground = 0
    pixels = 0
    iterator = tqdm(range(len(dataset)), desc="measure foreground", unit="image")
    for index in iterator:
        mask = dataset[index]["mask"]
        foreground += int(mask.sum().item())
        pixels += int(mask.numel())
    fraction = foreground / max(pixels, 1)
    weight = min(maximum, (pixels - foreground) / max(foreground, 1))
    return max(weight, 1.0), fraction


def _evaluate(
    loader: DataLoader,
    model: nn.Module,
    *,
    device: torch.device,
    thresholds: tuple[float, ...],
    positive_weight: float,
    description: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    metrics = ThresholdMetrics(thresholds)
    sums: Counter[str] = Counter()
    count = 0
    previews: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, unit="batch", leave=False):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            logits = _road_logits(model, image)
            loss, components = road_bce_dice_loss(
                logits, target, positive_weight=positive_weight
            )
            batch_size = image.shape[0]
            sums["loss"] += float(loss.item()) * batch_size
            for name, value in components.items():
                sums[name] += float(value.item()) * batch_size
            probability = torch.sigmoid(logits).cpu().numpy()
            target_np = target.cpu().numpy()
            metrics.update(probability, target_np)
            if len(previews) < 8:
                for item_index in range(min(batch_size, 8 - len(previews))):
                    previews.append(
                        {
                            "image": image[item_index].cpu().numpy(),
                            "target": target_np[item_index],
                            "probability": probability[item_index],
                            "aoi_id": batch["aoi_id"][item_index],
                            "edit_type": batch["edit_type"][item_index],
                        }
                    )
            count += batch_size
    summary = {
        "loss": sums["loss"] / count,
        "loss_bce": sums["bce"] / count,
        "loss_dice": sums["dice"] / count,
        "threshold_sweep": metrics.summary(),
        "best": metrics.best(),
        "images": count,
    }
    return summary, previews


def _train_epoch(
    loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    positive_weight: float,
    epoch: int,
    max_epochs: int,
) -> dict[str, float]:
    model.eval()
    model.map_decoder.train()
    sums: Counter[str] = Counter()
    count = 0
    iterator = tqdm(loader, desc=f"epoch {epoch:03d}/{max_epochs:03d} train", unit="batch")
    for batch in iterator:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = _road_logits(model, image)
        loss, components = road_bce_dice_loss(
            logits, target, positive_weight=positive_weight
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.map_decoder.parameters(), 5.0)
        optimizer.step()
        batch_size = image.shape[0]
        sums["loss"] += float(loss.item()) * batch_size
        sums["gradient_norm"] += float(gradient_norm) * batch_size
        for name, value in components.items():
            sums[name] += float(value.item()) * batch_size
        count += batch_size
        iterator.set_postfix(loss=f"{loss.item():.4f}")
    return {
        "loss": sums["loss"] / count,
        "loss_bce": sums["bce"] / count,
        "loss_dice": sums["dice"] / count,
        "gradient_norm": sums["gradient_norm"] / count,
    }


def _write_qc(
    path: Path,
    source: list[dict[str, Any]],
    adapted: list[dict[str, Any]],
    *,
    source_threshold: float,
    adapted_threshold: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    count = min(len(source), len(adapted), 8)
    figure, axes = plt.subplots(count, 4, figsize=(14, 3.4 * count), squeeze=False)
    for index in range(count):
        image = np.clip(adapted[index]["image"].transpose(1, 2, 0), 0, 255).astype(np.uint8)
        rows = (
            (image, "RGB"),
            (adapted[index]["target"], "Target road"),
            (source[index]["probability"] >= source_threshold, "Frozen SpaceNet"),
            (adapted[index]["probability"] >= adapted_threshold, "MUNO21 adapted"),
        )
        for column, (value, title) in enumerate(rows):
            axes[index, column].imshow(value, cmap=None if column == 0 else "gray")
            axes[index, column].axis("off")
            axes[index, column].set_title(title if index else f"{title}")
        axes[index, 0].set_ylabel(
            f"{adapted[index]['aoi_id']}\n{adapted[index]['edit_type']}", fontsize=9
        )
    figure.suptitle("SAM-Road MUNO21 road-head adaptation", fontsize=14)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _snapshot_decoder(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.map_decoder.state_dict().items()
    }


def _load_decoder(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.map_decoder.load_state_dict(state, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
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
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-positive-weight", type=float, default=12.0)
    parser.add_argument(
        "--thresholds",
        default="0.02,0.05,0.10,0.20,0.30,0.341,0.40,0.50,0.60,0.70,0.80",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-target-evaluation", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists() and args.resume is None:
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    if args.max_epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("max-epochs, patience and batch-size must be positive")
    thresholds = tuple(float(value) for value in args.thresholds.split(","))
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("threshold grid contains duplicates")
    _seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

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
        train_records = train_records[:8]
        calibration_records = calibration_records[:4]
        target_records = target_records[:4]
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
    model = predictor.model
    trainable_parameters = _trainable_decoder(model)
    source_decoder = _snapshot_decoder(model)

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset = MUNO21RoadDataset(train_records, patch_size=predictor.patch_size, augment=True)
    plain_train_dataset = MUNO21RoadDataset(
        train_records, patch_size=predictor.patch_size, augment=False
    )
    calibration_dataset = MUNO21RoadDataset(
        calibration_records, patch_size=predictor.patch_size, augment=False
    )
    target_dataset = MUNO21RoadDataset(
        target_records, patch_size=predictor.patch_size, augment=False
    )
    positive_weight, foreground_fraction = _foreground_weight(
        plain_train_dataset, args.max_positive_weight
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _worker_seed,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, drop_last=False, **loader_options
    )
    calibration_loader = DataLoader(calibration_dataset, shuffle=False, **loader_options)
    target_loader = DataLoader(target_dataset, shuffle=False, **loader_options)

    config = {
        "schema_version": "muno21-sam-road-head-v1",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "protocol": {
            "training": split_support(train_records),
            "calibration": split_support(calibration_records),
            "target_validation": split_support(target_records),
            "target_evaluated_after_freeze": not args.skip_target_evaluation,
            "frozen_test_access": False,
        },
        "model": {
            "upstream_commit": commit,
            "patch_size": predictor.patch_size,
            "trainable_scope": "map_decoder_only",
            "trainable_parameters": trainable_parameters,
            "source_checkpoint_sha256": _sha256(args.source_checkpoint),
            "sam_checkpoint_sha256": _sha256(args.sam_checkpoint),
        },
        "data": {
            "train_annotations_sha256": _sha256(train_annotations),
            "target_annotations_sha256": _sha256(target_annotations),
            "foreground_fraction": foreground_fraction,
            "positive_weight": positive_weight,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    (args.output_root / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    source_calibration, _ = _evaluate(
        calibration_loader,
        model,
        device=device,
        thresholds=thresholds,
        positive_weight=positive_weight,
        description="source calibration",
    )
    optimizer = torch.optim.AdamW(
        model.map_decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    start_epoch = 1
    best_epoch = 0
    best_f1 = -1.0
    best_threshold = float(source_calibration["best"]["threshold"])
    best_decoder = _snapshot_decoder(model)
    stale_epochs = 0
    if args.resume is not None:
        resume = torch.load(args.resume, map_location="cpu")
        _load_decoder(model, resume["map_decoder_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        best_epoch = int(resume["best_epoch"])
        best_f1 = float(resume["best_f1"])
        best_threshold = float(resume["best_threshold"])
        best_decoder = resume["best_decoder_state_dict"]
        stale_epochs = int(resume["stale_epochs"])

    history_path = args.output_root / "history.jsonl"
    history_mode = "a" if args.resume is not None else "x"
    with history_path.open(history_mode, encoding="utf-8", buffering=1) as history:
        for epoch in range(start_epoch, args.max_epochs + 1):
            train_metrics = _train_epoch(
                train_loader,
                model,
                optimizer,
                device=device,
                positive_weight=positive_weight,
                epoch=epoch,
                max_epochs=args.max_epochs,
            )
            calibration_metrics, _ = _evaluate(
                calibration_loader,
                model,
                device=device,
                thresholds=thresholds,
                positive_weight=positive_weight,
                description=f"epoch {epoch:03d} calibration",
            )
            epoch_f1 = float(calibration_metrics["best"]["f1"])
            scheduler.step(epoch_f1)
            improved = epoch_f1 > best_f1 + 1e-6
            if improved:
                best_epoch = epoch
                best_f1 = epoch_f1
                best_threshold = float(calibration_metrics["best"]["threshold"])
                best_decoder = _snapshot_decoder(model)
                stale_epochs = 0
                torch.save(
                    {
                        "schema_version": "muno21-sam-road-decoder-v1",
                        "epoch": epoch,
                        "threshold": best_threshold,
                        "map_decoder_state_dict": best_decoder,
                        "source_checkpoint_sha256": config["model"]["source_checkpoint_sha256"],
                    },
                    args.output_root / "checkpoints" / "best_decoder.pt",
                )
            else:
                stale_epochs += 1
            row = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "calibration": calibration_metrics,
                "improved": improved,
                "best_epoch": best_epoch,
                "best_f1": best_f1,
                "best_threshold": best_threshold,
                "stale_epochs": stale_epochs,
            }
            history.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            torch.save(
                {
                    **row,
                    "map_decoder_state_dict": _snapshot_decoder(model),
                    "best_decoder_state_dict": best_decoder,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                args.output_root / "checkpoints" / "last.pt",
            )
            if stale_epochs >= args.patience:
                print(f"early stopping after {stale_epochs} stale epochs", flush=True)
                break

    _load_decoder(model, source_decoder)
    source_target = None
    source_previews: list[dict[str, Any]] = []
    if not args.skip_target_evaluation:
        source_target, source_previews = _evaluate(
            target_loader,
            model,
            device=device,
            thresholds=thresholds,
            positive_weight=positive_weight,
            description="frozen source target-val",
        )
    _load_decoder(model, best_decoder)
    adapted_calibration, _ = _evaluate(
        calibration_loader,
        model,
        device=device,
        thresholds=thresholds,
        positive_weight=positive_weight,
        description="adapted calibration",
    )
    adapted_target = None
    adapted_previews: list[dict[str, Any]] = []
    if not args.skip_target_evaluation:
        adapted_target, adapted_previews = _evaluate(
            target_loader,
            model,
            device=device,
            thresholds=(best_threshold,),
            positive_weight=positive_weight,
            description="adapted target-val",
        )
        _write_qc(
            args.output_root / "qc_target_val.png",
            source_previews,
            adapted_previews,
            source_threshold=float(source_calibration["best"]["threshold"]),
            adapted_threshold=best_threshold,
        )
    full_checkpoint = None
    if not args.smoke:
        full_checkpoint = args.output_root / "checkpoints" / "best_full.ckpt"
        torch.save(
            {
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "activemap_adapter": {
                    "schema_version": "muno21-sam-road-full-v1",
                    "best_epoch": best_epoch,
                    "road_threshold": best_threshold,
                    "source_checkpoint_sha256": config["model"][
                        "source_checkpoint_sha256"
                    ],
                    "frozen_test_access": False,
                },
            },
            full_checkpoint,
        )
    source_target_at_source_threshold = None
    if source_target is not None:
        source_threshold = float(source_calibration["best"]["threshold"])
        source_target_at_source_threshold = next(
            row
            for row in source_target["threshold_sweep"]
            if float(row["threshold"]) == source_threshold
        )
    summary = {
        "schema_version": "muno21-sam-road-head-result-v1",
        "best_epoch": best_epoch,
        "selected_threshold": best_threshold,
        "source_calibration": source_calibration,
        "adapted_calibration": adapted_calibration,
        "source_target_at_source_threshold": source_target_at_source_threshold,
        "adapted_target": adapted_target,
        "target_f1_delta": (
            float(adapted_target["best"]["f1"])
            - float(source_target_at_source_threshold["f1"])
            if adapted_target is not None and source_target_at_source_threshold is not None
            else None
        ),
        "target_iou_delta": (
            float(adapted_target["best"]["iou"])
            - float(source_target_at_source_threshold["iou"])
            if adapted_target is not None and source_target_at_source_threshold is not None
            else None
        ),
        "deployable_checkpoint": str(full_checkpoint) if full_checkpoint else None,
        "frozen_test_access": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
