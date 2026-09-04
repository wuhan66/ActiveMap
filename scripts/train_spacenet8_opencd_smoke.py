#!/usr/bin/env python3
"""Train an OpenCD two-image model on aligned SpaceNet8 flood labels."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_sn7_opencd_generic import OpenCDPairModel
from scripts.train_spacenet8_flood_smoke import FloodDataset, read_records


def evaluate(model: Any, loader: Any, device: Any) -> dict[str, float]:
    import torch

    model.eval()
    tp = fp = fn = 0
    mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            old_image = (image[:, :3] - mean) / std
            new_image = (image[:, 3:] - mean) / std
            prediction = model(old_image, new_image).argmax(dim=1).bool()
            target = batch["target"].to(device).bool()
            valid = batch["valid"].to(device).bool()
            prediction &= valid
            target &= valid
            tp += int((prediction & target).sum())
            fp += int((prediction & ~target & valid).sum())
            fn += int((~prediction & target & valid).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "iou": tp / max(tp + fp + fn, 1),
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "precision": precision,
        "recall": recall,
        "false_positive_pixels": float(fp),
        "false_negative_pixels": float(fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("opencd_repo", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--model-name", default="ChangeFormer-MiT-B0")
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--mode", choices=("pre_post", "post_only", "pre_only"), default="pre_post")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--positive-weight", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from mmseg.models.losses import LovaszLoss
    from torch.utils.data import DataLoader

    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train = read_records(args.manifest, "train")
    val = read_records(args.manifest, "val")
    train_loader = DataLoader(
        FloodDataset(train, args.mode, True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        FloodDataset(val, args.mode, False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    model = OpenCDPairModel(
        opencd_repo=args.opencd_repo,
        config_path=args.config,
        pretrained_checkpoint=args.pretrained_checkpoint,
        initialize=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    class_weights = torch.tensor((1.0, args.positive_weight), device=device)
    lovasz = LovaszLoss(loss_type="multi_class", per_image=False, reduction="none")
    mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
    history = args.output_root / "history.jsonl"
    best_score, best_epoch, stale = -math.inf, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True).long()
            valid = batch["valid"].to(device, non_blocking=True).bool()
            target[~valid] = 255
            old_image = (image[:, :3] - mean) / std
            new_image = (image[:, 3:] - mean) / std
            optimizer.zero_grad(set_to_none=True)
            logits = model(old_image, new_image)
            ce = F.cross_entropy(logits, target, weight=class_weights, ignore_index=255)
            loss = ce + 0.75 * lovasz(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, val_loader, device)
        scheduler.step(metrics["f1"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val": metrics,
        }
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if metrics["f1"] > best_score:
            best_score, best_epoch, stale = metrics["f1"], epoch, 0
            torch.save({"model": model.state_dict(), "row": row}, args.output_root / "best.pt")
        else:
            stale += 1
        if epoch >= args.min_epochs and stale >= args.patience:
            break
    summary = {
        "schema_version": "activemap-spacenet8-opencd-smoke-v1",
        "model": args.model_name,
        "mode": args.mode,
        "seed": args.seed,
        "train_count": len(train),
        "val_count": len(val),
        "best_epoch": best_epoch,
        "best_val_f1": best_score,
        "positive_weight": args.positive_weight,
        "learning_rate": args.learning_rate,
        "test_assets_read": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
