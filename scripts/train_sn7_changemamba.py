"""Train official ChangeMamba on the frozen SN7 image-map update protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Record:
    sample_id: str
    aoi_id: str
    split: str
    image: Path
    prior: Path
    target: Path
    valid: Path | None
    edit_type: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_records(
    manifest: Path,
    split: str,
    limit: int | None,
    limit_per_edit: int | None = None,
) -> list[Record]:
    rows: list[Record] = []
    edit_counts = {"KEEP": 0, "ADD": 0, "DELETE": 0, "RESHAPE": 0}
    root = manifest.parent
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["split"] != split:
                continue
            if (
                limit_per_edit is not None
                and edit_counts[item["edit_type"]] >= limit_per_edit
            ):
                continue
            rows.append(
                Record(
                    sample_id=item["sample_id"],
                    aoi_id=item["aoi_id"],
                    split=split,
                    image=_resolve(root, item["image_path"]),
                    prior=_resolve(root, item["prior_mask_path"]),
                    target=_resolve(root, item["target_mask_path"]),
                    valid=_resolve(root, item.get("valid_mask_path")),
                    edit_type=item["edit_type"],
                )
            )
            edit_counts[item["edit_type"]] += 1
            if limit_per_edit is not None and all(
                count >= limit_per_edit for count in edit_counts.values()
            ):
                break
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no {split} records in {manifest}")
    return rows


def _channels_first(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"expected three image dimensions, got {array.shape}")
    if array.shape[0] in {1, 3, 4}:
        output = array
    elif array.shape[-1] in {1, 3, 4}:
        output = np.moveaxis(array, -1, 0)
    else:
        raise ValueError(f"cannot infer channels from {array.shape}")
    output = output[:3].astype(np.float32)
    if output.max(initial=0.0) > 1.0:
        output /= 255.0
    return np.clip(output, 0.0, 1.0)


def _translate_no_wrap(value: Any, shift_y: int, shift_x: int) -> Any:
    """Translate the final two dimensions and fill exposed pixels with zero."""
    import torch

    output = torch.roll(value, shifts=(shift_y, shift_x), dims=(-2, -1))
    if shift_y > 0:
        output[..., :shift_y, :] = 0
    elif shift_y < 0:
        output[..., shift_y:, :] = 0
    if shift_x > 0:
        output[..., :, :shift_x] = 0
    elif shift_x < 0:
        output[..., :, shift_x:] = 0
    return output


class SN7ChangeDataset:
    def __init__(
        self,
        records: list[Record],
        *,
        augment: bool,
        image_size: int,
        input_mode: str = "image_prior",
        max_translation_pixels: int = 0,
        prior_input_translation_pixels: int = 0,
        corruption_seed: int = 0,
    ) -> None:
        import torch

        if input_mode not in {"image_prior", "image_only", "prior_only"}:
            raise ValueError(f"unsupported input mode: {input_mode}")
        self.records = records
        self.augment = augment
        self.image_size = image_size
        self.input_mode = input_mode
        if max_translation_pixels < 0:
            raise ValueError("max_translation_pixels must be non-negative")
        if prior_input_translation_pixels < 0:
            raise ValueError(
                "prior_input_translation_pixels must be non-negative"
            )
        self.max_translation_pixels = max_translation_pixels
        self.prior_input_translation_pixels = prior_input_translation_pixels
        self.corruption_seed = corruption_seed
        self.mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
        self.std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        record = self.records[index]
        image = torch.from_numpy(_channels_first(np.load(record.image)))
        prior = torch.from_numpy(
            np.asarray(np.load(record.prior), dtype=np.float32).squeeze()
        )
        target = torch.from_numpy(
            np.asarray(np.load(record.target), dtype=np.float32).squeeze()
        )
        valid = (
            torch.from_numpy(
                np.asarray(np.load(record.valid), dtype=np.float32).squeeze()
            )
            if record.valid is not None
            else torch.ones_like(target)
        )
        if prior.ndim != 2 or target.shape != prior.shape or valid.shape != prior.shape:
            raise ValueError(f"invalid mask shape for {record.sample_id}")
        if image.shape[-2:] != prior.shape:
            raise ValueError(f"image-mask mismatch for {record.sample_id}")

        if self.augment:
            if torch.rand(()) < 0.5:
                image, prior, target, valid = [
                    torch.flip(item, dims=(-1,))
                    for item in (image, prior, target, valid)
                ]
            if torch.rand(()) < 0.5:
                image, prior, target, valid = [
                    torch.flip(item, dims=(-2,))
                    for item in (image, prior, target, valid)
                ]
            rotations = int(torch.randint(0, 4, ()).item())
            if rotations:
                image, prior, target, valid = [
                    torch.rot90(item, rotations, dims=(-2, -1))
                    for item in (image, prior, target, valid)
                ]
            if self.max_translation_pixels:
                limit = self.max_translation_pixels
                shift_y = int(torch.randint(-limit, limit + 1, ()).item())
                shift_x = int(torch.randint(-limit, limit + 1, ()).item())
                image, prior, target, valid = [
                    _translate_no_wrap(item, shift_y, shift_x)
                    for item in (image, prior, target, valid)
                ]

        if image.shape[-1] != self.image_size:
            size = (self.image_size, self.image_size)
            image = F.interpolate(
                image[None], size=size, mode="bilinear", align_corners=False
            )[0]
            prior, target, valid = [
                F.interpolate(item[None, None], size=size, mode="nearest")[0, 0]
                for item in (prior, target, valid)
            ]

        prior_binary = prior >= 0.5
        target_binary = target >= 0.5
        valid_binary = valid >= 0.5
        change = torch.logical_xor(prior_binary, target_binary).long()
        change[~valid_binary] = 255
        prior_input = prior_binary
        prior_input_shift = (0, 0)
        if self.prior_input_translation_pixels:
            rng = random.Random(f"{self.corruption_seed}:{record.sample_id}")
            limit = self.prior_input_translation_pixels
            prior_input_shift = (
                rng.randint(-limit, limit),
                rng.randint(-limit, limit),
            )
            prior_input = _translate_no_wrap(
                prior_input,
                prior_input_shift[0],
                prior_input_shift[1],
            )
        prior_rgb = prior_input.float()[None].repeat(3, 1, 1)
        blank_rgb = torch.zeros_like(image)
        if self.input_mode == "image_prior":
            old_input, new_input = prior_rgb, image
        elif self.input_mode == "image_only":
            old_input, new_input = blank_rgb, image
        else:
            old_input, new_input = prior_rgb, blank_rgb
        return {
            "old_map": (old_input - self.mean) / self.std,
            "new_rgb": (new_input - self.mean) / self.std,
            "change": change,
            "prior": prior_binary,
            "target": target_binary,
            "valid": valid_binary,
            "sample_id": record.sample_id,
            "aoi_id": record.aoi_id,
            "edit_type": record.edit_type,
            "prior_input_shift": torch.tensor(prior_input_shift),
        }


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _predicted_edit(prior: Any, prediction: Any, valid: Any) -> str:
    changed = prediction & valid
    added = bool((changed & ~prior).any().item())
    removed = bool((changed & prior).any().item())
    if added and removed:
        return "RESHAPE"
    if added:
        return "ADD"
    if removed:
        return "DELETE"
    return "KEEP"


def _validate(model: Any, loader: Any, device: Any) -> dict[str, Any]:
    import torch

    model.eval()
    sample_prior_iou: list[float] = []
    sample_map_iou: list[float] = []
    sample_change_iou: list[float] = []
    edited_map_iou: list[float] = []
    operation_correct = 0
    keep_false_change: list[float] = []
    with torch.no_grad():
        for batch in loader:
            old_map = batch["old_map"].to(device, non_blocking=True)
            new_rgb = batch["new_rgb"].to(device, non_blocking=True)
            prediction = model(old_map, new_rgb).argmax(dim=1).bool().cpu()
            prior = batch["prior"].bool()
            target = batch["target"].bool()
            valid = batch["valid"].bool()
            committed = torch.logical_xor(prior, prediction)
            truth_change = torch.logical_xor(prior, target)
            for index in range(prediction.shape[0]):
                mask = valid[index]

                def iou(
                    left: Any, right: Any, valid_mask: Any = mask
                ) -> float:
                    intersection = ((left & right) & valid_mask).sum().item()
                    union = ((left | right) & valid_mask).sum().item()
                    return float(intersection / union) if union else 1.0

                prior_iou = iou(prior[index], target[index])
                map_iou = iou(committed[index], target[index])
                change_iou = iou(prediction[index], truth_change[index])
                sample_prior_iou.append(prior_iou)
                sample_map_iou.append(map_iou)
                sample_change_iou.append(change_iou)
                if batch["edit_type"][index] != "KEEP":
                    edited_map_iou.append(map_iou)
                else:
                    keep_false_change.append(
                        float((prediction[index] & mask).sum().item() / mask.sum().item())
                    )
                operation_correct += int(
                    _predicted_edit(prior[index], prediction[index], mask)
                    == batch["edit_type"][index]
                )
    count = len(sample_map_iou)
    return {
        "sample_count": float(count),
        "prior_map_iou": float(np.mean(sample_prior_iou)),
        "committed_map_iou": float(np.mean(sample_map_iou)),
        "map_iou_delta": float(
            np.mean(sample_map_iou) - np.mean(sample_prior_iou)
        ),
        "edited_map_iou": (
            float(np.mean(edited_map_iou)) if edited_map_iou else None
        ),
        "change_iou": float(np.mean(sample_change_iou)),
        "operation_accuracy": float(operation_correct / count),
        "keep_false_change_fraction": (
            float(np.mean(keep_false_change)) if keep_false_change else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("change_mamba_repo", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--positive-class-weight", type=float, default=1.0)
    parser.add_argument("--max-translation-pixels", type=int, default=0)
    parser.add_argument(
        "--input-mode",
        choices=("image_prior", "image_only", "prior_only"),
        default="image_prior",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--limit-per-edit", type=int)
    parser.add_argument("--overfit", action="store_true")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    sys.path.insert(0, str(args.change_mamba_repo))
    from changedetection.configs.config import _C
    from changedetection.models.ChangeMambaBCD import ChangeMambaBCD
    from changedetection.script.script_utils import get_vssm_kwargs
    from changedetection.utils_func.lovasz_loss import lovasz_softmax

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.positive_class_weight <= 0.0:
        raise ValueError("--positive-class-weight must be positive")
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
        SN7ChangeDataset(
            train_records,
            augment=not args.overfit,
            image_size=args.image_size,
            input_mode=args.input_mode,
            max_translation_pixels=args.max_translation_pixels,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        SN7ChangeDataset(
            val_records,
            augment=False,
            image_size=args.image_size,
            input_mode=args.input_mode,
            max_translation_pixels=0,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    config = _C.clone()
    config.defrost()
    config.merge_from_file(str(args.config))
    config.freeze()
    model = ChangeMambaBCD(
        pretrained=str(args.encoder_checkpoint) if args.encoder_checkpoint else None,
        **get_vssm_kwargs(config),
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
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
                    logits,
                    labels,
                    weight=class_weights,
                    ignore_index=255,
                )
                lovasz = lovasz_softmax(
                    F.softmax(logits, dim=1), labels, ignore=255
                )
                loss = ce + 0.75 * lovasz
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val": metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        torch.save(
            {
                "schema_version": "sn7-changemamba-v1",
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "arguments": vars(args),
                "validation": metrics,
            },
            args.output_dir / "latest.pt",
        )
        if metrics["committed_map_iou"] > best_score:
            best_score = metrics["committed_map_iou"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "schema_version": "sn7-changemamba-v1",
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "arguments": vars(args),
                    "validation": metrics,
                },
                args.output_dir / "best.pt",
            )
        else:
            stale_epochs += 1
        if (
            not args.overfit
            and epoch >= args.min_epochs
            and stale_epochs >= args.patience
        ):
            break

    summary = {
        "schema_version": "sn7-changemamba-training-v1",
        "source_commit": "9ce9cec13f9ea14bc0ad91f071577ec9b3a97983",
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "seed": args.seed,
        "train_count": len(train_records),
        "validation_count": len(val_records),
        "best_epoch": best_epoch,
        "best_committed_map_iou": best_score,
        "input_mode": args.input_mode,
        "input_contract": {
            "image_prior": "old editable map RGB + new RGB -> binary change",
            "image_only": "blank old-map branch + new RGB -> binary change",
            "prior_only": "old editable map RGB + blank new-image branch -> binary change",
        }[args.input_mode],
        "writeback": "prior XOR predicted change",
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "positive_class_weight": args.positive_class_weight,
        "max_translation_pixels": args.max_translation_pixels,
        "encoder_checkpoint_sha256": (
            _sha256(args.encoder_checkpoint)
            if args.encoder_checkpoint is not None
            else None
        ),
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
