#!/usr/bin/env python3
"""Train a compact paired-image flood smoke model on SpaceNet8."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from activemap.nn.updater import PriorConditionedUNet, UpdaterConfig


class FloodDataset(Dataset):
    def __init__(self, records: list[dict[str, object]], mode: str, augment: bool) -> None:
        self.records = records
        self.mode = mode
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        payload = np.load(str(record["array_path"]))
        pre = torch.from_numpy(payload["pre"].astype(np.float32))
        post = torch.from_numpy(payload["post"].astype(np.float32))
        target = torch.from_numpy(payload["target"].astype(np.float32))
        valid = torch.from_numpy(payload["valid"].astype(np.float32))
        if self.mode == "post_only":
            pre = torch.zeros_like(pre)
        elif self.mode == "pre_only":
            post = torch.zeros_like(post)
        image = torch.cat((pre, post), dim=0)
        if self.augment:
            if torch.rand(()) < 0.5:
                image, target, valid = [
                    torch.flip(value, (-1,)) for value in (image, target, valid)
                ]
            if torch.rand(()) < 0.5:
                image, target, valid = [
                    torch.flip(value, (-2,)) for value in (image, target, valid)
                ]
        return {"image": image, "target": target, "valid": valid}


def read_records(path: Path, split: str) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"no {split} rows")
    return selected


def dice_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits) * valid
    target = target * valid
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    intersection = union = true_positive = false_positive = false_negative = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["target"].to(device).bool()
            valid = batch["valid"].to(device).bool()
            prior = torch.zeros((image.shape[0], 1, *image.shape[-2:]), device=device)
            prediction = torch.sigmoid(model(image, prior)["segmentation_logits"][:, 0]) >= 0.5
            prediction &= valid
            target &= valid
            true_positive += int((prediction & target).sum())
            false_positive += int((prediction & ~target & valid).sum())
            false_negative += int((~prediction & target & valid).sum())
            intersection += int((prediction & target).sum())
            union += int((prediction | target).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "iou": intersection / max(union, 1),
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "precision": precision,
        "recall": recall,
        "false_positive_pixels": float(false_positive),
        "false_negative_pixels": float(false_negative),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--mode", choices=("pre_post", "post_only", "pre_only"), default="pre_post")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--positive-weight", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
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
        num_workers=2,
    )
    val_loader = DataLoader(
        FloodDataset(val, args.mode, False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )
    config = UpdaterConfig(
        image_channels=6, prior_channels=1, base_channels=16, geometry_dim=1, edit_classes=2
    )
    model = PriorConditionedUNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    positive_weight = torch.tensor(args.positive_weight, device=device)
    history = args.output_root / "history.jsonl"
    best_score, best_epoch, stale = -math.inf, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            image = batch["image"].to(device)
            target = batch["target"].to(device)
            valid = batch["valid"].to(device)
            prior = torch.zeros((image.shape[0], 1, *image.shape[-2:]), device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image, prior)["segmentation_logits"][:, 0]
            bce = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=positive_weight, reduction="none"
            )
            bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
            loss = bce + dice_loss(logits, target, valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": metrics}
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if metrics["f1"] > best_score:
            best_score, best_epoch, stale = metrics["f1"], epoch, 0
            torch.save(
                {"model": model.state_dict(), "config": config.as_dict(), "row": row},
                args.output_root / "best.pt",
            )
        else:
            stale += 1
        if epoch >= 8 and stale >= args.patience:
            break
    summary = {
        "schema_version": "activemap-spacenet8-flood-smoke-training-v1",
        "mode": args.mode,
        "seed": args.seed,
        "train_count": len(train),
        "val_count": len(val),
        "best_epoch": best_epoch,
        "best_val_f1": best_score,
        "test_assets_read": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
