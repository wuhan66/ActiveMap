#!/usr/bin/env python3
"""Train official Open-CD BAN under the frozen SN7 editable-map protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from scripts.train_sn7_changemamba import (
    SN7ChangeDataset,
    _read_records,
    _validate,
)

BAN_MEAN = (122.7709, 116.7460, 104.0937)
BAN_STD = (68.5005, 66.6322, 70.3232)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OPENCD_COMMIT = "790a1972b538baaaffd4c3022a4d9afaa3f861e5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ban_normalize(value: Any) -> Any:
    """Convert the shared ImageNet-normalized RGB tensor to BAN's RGB scale."""
    import torch

    image_mean = torch.as_tensor(
        IMAGENET_MEAN, dtype=value.dtype, device=value.device
    )[:, None, None]
    image_std = torch.as_tensor(
        IMAGENET_STD, dtype=value.dtype, device=value.device
    )[:, None, None]
    ban_mean = torch.as_tensor(
        BAN_MEAN, dtype=value.dtype, device=value.device
    )[:, None, None]
    ban_std = torch.as_tensor(
        BAN_STD, dtype=value.dtype, device=value.device
    )[:, None, None]
    rgb_255 = (value * image_std + image_mean) * 255.0
    return (rgb_255 - ban_mean) / ban_std


class SN7BANDataset(SN7ChangeDataset):
    """Shared SN7 samples with BAN's official CLIP normalization."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        item["old_map"] = _ban_normalize(item["old_map"])
        item["new_rgb"] = _ban_normalize(item["new_rgb"])
        return item


def _set_nested(config: Any, keys: tuple[str, ...], value: Any) -> None:
    cursor = config
    for key in keys[:-1]:
        cursor = cursor[key]
    cursor[keys[-1]] = value


class BANPairModel:
    """Thin torch-compatible adapter around Open-CD's official BAN model."""

    def __new__(
        cls,
        *,
        opencd_repo: Path,
        config_path: Path,
        clip_checkpoint: Path | None,
        side_checkpoint: Path | None,
        initialize: bool,
    ) -> Any:
        import torch
        from mmengine.config import Config
        from mmengine.model import revert_sync_batchnorm
        from mmengine.registry import init_default_scope

        sys.path.insert(0, str(opencd_repo))
        import opencd.models  # noqa: F401
        import opencd.models.data_preprocessor  # noqa: F401
        from opencd.registry import MODELS

        init_default_scope("opencd")
        config = Config.fromfile(str(config_path))
        model_config = config.model
        model_config.test_cfg = dict(mode="whole")
        if initialize:
            if clip_checkpoint is None or side_checkpoint is None:
                raise ValueError("training requires both official pretrained checkpoints")
            model_config.pretrained = str(clip_checkpoint)
            _set_nested(
                model_config,
                ("decode_head", "ban_cfg", "side_enc_cfg", "init_cfg", "checkpoint"),
                str(side_checkpoint),
            )
        else:
            model_config.pretrained = None
            _set_nested(
                model_config,
                ("decode_head", "ban_cfg", "side_enc_cfg", "init_cfg"),
                None,
            )
        model = MODELS.build(model_config)
        if initialize:
            model.init_weights()
        model = revert_sync_batchnorm(model)

        class Adapter(torch.nn.Module):
            def __init__(self, wrapped: Any) -> None:
                super().__init__()
                self.wrapped = wrapped

            def forward(self, old_map: Any, new_rgb: Any) -> Any:
                logits = self.wrapped._forward(torch.cat((old_map, new_rgb), dim=1))
                if logits.shape[-2:] != old_map.shape[-2:]:
                    logits = torch.nn.functional.interpolate(
                        logits,
                        size=old_map.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                return logits

        return Adapter(model)


def _optimizer_groups(
    named_parameters: Iterable[tuple[str, Any]],
    *,
    learning_rate: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Reproduce BAN's encoder/decoder LR multipliers without MMEngine Runner."""
    buckets: dict[tuple[float, float], list[Any]] = {}
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        lr_multiplier = 0.1 if "image_encoder" in name else 1.0
        if "mask_decoder" in name:
            lr_multiplier = 10.0
        decay = 0.0 if "norm" in name.lower() or parameter.ndim == 1 else weight_decay
        key = (learning_rate * lr_multiplier, decay)
        buckets.setdefault(key, []).append(parameter)
    return [
        {"params": parameters, "lr": lr, "weight_decay": decay}
        for (lr, decay), parameters in sorted(buckets.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("opencd_repo", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--side-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-class-weight", type=float, default=5.0)
    parser.add_argument("--max-translation-pixels", type=int, default=0)
    parser.add_argument("--lovasz-weight", type=float, default=0.75)
    parser.add_argument("--gradient-clip", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--limit-per-edit", type=int)
    parser.add_argument("--overfit", action="store_true")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from mmseg.models.losses import LovaszLoss
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.positive_class_weight <= 0 or args.lovasz_weight < 0:
        raise ValueError("loss weights must be non-negative and class weight positive")
    for required in (
        args.manifest,
        args.opencd_repo,
        args.config,
        args.clip_checkpoint,
        args.side_checkpoint,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    args.output_dir.mkdir(parents=True)
    _seed_everything(args.seed)
    train_records = _read_records(
        args.manifest, "train", args.train_limit, args.limit_per_edit
    )
    val_records = (
        train_records
        if args.overfit
        else _read_records(args.manifest, "val", args.val_limit)
    )
    train_loader = DataLoader(
        SN7BANDataset(
            train_records,
            augment=not args.overfit,
            image_size=args.image_size,
            max_translation_pixels=args.max_translation_pixels,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        SN7BANDataset(val_records, augment=False, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = BANPairModel(
        opencd_repo=args.opencd_repo,
        config_path=args.config,
        clip_checkpoint=args.clip_checkpoint,
        side_checkpoint=args.side_checkpoint,
        initialize=True,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        _optimizer_groups(
            model.named_parameters(),
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))
    class_weights = torch.tensor(
        (1.0, args.positive_class_weight),
        dtype=torch.float32,
        device=args.device,
    )
    lovasz = LovaszLoss(
        loss_type="multi_class",
        per_image=False,
        reduction="none",
    )
    history_path = args.output_dir / "history.jsonl"
    best_score = -math.inf
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch:03d} train",
            unit="batch",
            mininterval=5.0,
            miniters=50,
        )
        for batch in progress:
            old_map = batch["old_map"].to(args.device, non_blocking=True)
            new_rgb = batch["new_rgb"].to(args.device, non_blocking=True)
            labels = batch["change"].to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                logits = model(old_map, new_rgb)
                ce = F.cross_entropy(
                    logits, labels, weight=class_weights, ignore_index=255
                )
                loss = ce + args.lovasz_weight * lovasz(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            progress.set_postfix(
                loss=f"{np.mean(losses[-50:]):.4f}",
                refresh=False,
            )

        metrics = _validate(model, val_loader, torch.device(args.device))
        if not args.overfit:
            scheduler.step(metrics["committed_map_iou"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "val": metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        payload = {
            "schema_version": "sn7-opencd-ban-v1",
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "arguments": vars(args),
            "validation": metrics,
        }
        torch.save(payload, args.output_dir / "latest.pt")
        if metrics["committed_map_iou"] > best_score:
            best_score = metrics["committed_map_iou"]
            best_epoch = epoch
            stale_epochs = 0
            payload.pop("optimizer")
            payload.pop("scheduler")
            torch.save(payload, args.output_dir / "best.pt")
        else:
            stale_epochs += 1
        if (
            not args.overfit
            and epoch >= args.min_epochs
            and stale_epochs >= args.patience
        ):
            break

    summary = {
        "schema_version": "sn7-opencd-ban-training-v1",
        "source_commit": OPENCD_COMMIT,
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "seed": args.seed,
        "train_count": len(train_records),
        "validation_count": len(val_records),
        "best_epoch": best_epoch,
        "best_committed_map_iou": best_score,
        "input_contract": "old editable map RGB + new RGB -> binary change",
        "writeback": "prior XOR predicted change",
        "selection_metric": "validation committed_map_iou",
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "positive_class_weight": args.positive_class_weight,
        "max_translation_pixels": args.max_translation_pixels,
        "lovasz_weight": args.lovasz_weight,
        "clip_checkpoint_sha256": _sha256(args.clip_checkpoint),
        "side_checkpoint_sha256": _sha256(args.side_checkpoint),
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
