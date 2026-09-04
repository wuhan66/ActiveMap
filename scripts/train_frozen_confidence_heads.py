from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from activemap.nn.updater import (
    PriorConditionedUNet,
    UpdaterConfig,
    combine_confidence_targets,
    operation_probabilities,
)
from activemap.training.selector import resolve_device
from activemap.training.updater_data import UpdaterDataset
from activemap.updater_records import load_updater_samples


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_features(
    model: PriorConditionedUNet,
    dataset: UpdaterDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    feature_batches: list[Tensor] = []
    segmentation_batches: list[Tensor] = []
    edit_batches: list[Tensor] = []
    captured: list[Tensor] = []

    def capture_shared(
        _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor
    ) -> None:
        captured.append(output.detach())

    handle = model.shared_head.register_forward_hook(capture_shared)
    model.eval()
    try:
        with torch.no_grad():
            for raw_batch in loader:
                batch = {
                    key: value.to(device, non_blocking=True)
                    if isinstance(value, Tensor)
                    else value
                    for key, value in raw_batch.items()
                }
                captured.clear()
                image = cast(Tensor, batch["image"])
                prior = cast(Tensor, batch["prior_mask"])
                target = cast(Tensor, batch["target_mask"])
                valid = cast(Tensor, batch["valid_mask"])
                edit_target = cast(Tensor, batch["edit_target"])
                outputs = model(image, prior)
                if len(captured) != 1:
                    raise RuntimeError("shared-head hook did not capture exactly one batch")
                mask_probabilities = torch.sigmoid(outputs["segmentation_logits"]) * valid
                valid_target = target * valid
                dimensions = tuple(range(1, target.ndim))
                intersection = torch.sum(mask_probabilities * valid_target, dim=dimensions)
                union = torch.sum(
                    mask_probabilities
                    + valid_target
                    - mask_probabilities * valid_target,
                    dim=dimensions,
                )
                segmentation_quality = (intersection + 1e-6) / (union + 1e-6)
                probabilities = operation_probabilities(outputs, prior)
                edit_quality = torch.gather(
                    probabilities, 1, edit_target[:, None]
                ).squeeze(1)
                feature_batches.append(captured[0].cpu())
                segmentation_batches.append(segmentation_quality.cpu())
                edit_batches.append(edit_quality.cpu())
    finally:
        handle.remove()
    return {
        "features": torch.cat(feature_batches),
        "segmentation_quality": torch.cat(segmentation_batches),
        "edit_quality": torch.cat(edit_batches),
    }


def _train_head(
    train_features: Tensor,
    train_target: Tensor,
    val_features: Tensor,
    val_target: Tensor,
    *,
    device: torch.device,
    seed: int,
    batch_size: int,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[dict[str, Tensor], list[dict[str, float]]]:
    _set_seed(seed)
    head = nn.Linear(train_features.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    train_features = train_features.to(device)
    train_target = train_target.to(device)
    val_features = val_features.to(device)
    val_target = val_target.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        head.train()
        permutation = torch.randperm(
            train_features.shape[0], generator=generator, device=device
        )
        train_loss_sum = 0.0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            logits = head(train_features[indices]).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, train_target[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(indices)
        head.eval()
        with torch.no_grad():
            val_logits = head(val_features).squeeze(1)
            val_loss = float(F.binary_cross_entropy_with_logits(val_logits, val_target))
        train_loss = train_loss_sum / train_features.shape[0]
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
                "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("confidence head training produced no checkpoint")
    return best_state, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train paired confidence heads over one frozen updater backbone."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--head-batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--presence-threshold", type=float, required=True)
    parser.add_argument("--change-threshold", type=float, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from activemap.evaluation.updater import evaluate_updater_checkpoint

    _set_seed(args.seed)
    device = resolve_device(args.device)
    source: dict[str, Any] = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    model = PriorConditionedUNet(UpdaterConfig(**source["model_config"]))
    model.load_state_dict(source["state_dict"])
    model.to(device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extracted = {}
    for split in ("train", "val"):
        samples = load_updater_samples(args.samples, split=split)
        extracted[split] = _extract_features(
            model,
            UpdaterDataset(samples),
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    torch.save(extracted, args.output_dir / "frozen_confidence_features.pt")

    report: dict[str, Any] = {
        "scope": "frozen_backbone_confidence_only",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "samples": str(args.samples.resolve()),
        "seed": args.seed,
        "modes": {},
    }
    for mode in ("mean", "product"):
        train_target = combine_confidence_targets(
            extracted["train"]["segmentation_quality"],
            extracted["train"]["edit_quality"],
            mode=mode,
        )
        val_target = combine_confidence_targets(
            extracted["val"]["segmentation_quality"],
            extracted["val"]["edit_quality"],
            mode=mode,
        )
        head_state, history = _train_head(
            extracted["train"]["features"],
            train_target,
            extracted["val"]["features"],
            val_target,
            device=device,
            seed=args.seed,
            batch_size=args.head_batch_size,
            max_epochs=args.max_epochs,
            min_epochs=args.min_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        state_dict = copy.deepcopy(source["state_dict"])
        state_dict["confidence_head.weight"] = head_state["weight"].cpu()
        state_dict["confidence_head.bias"] = head_state["bias"].cpu()
        checkpoint_path = args.output_dir / f"confidence_{mode}.pt"
        checkpoint = {
            **{
                key: value
                for key, value in source.items()
                if key not in {"state_dict", "optimizer_state_dict", "scheduler_state_dict"}
            },
            "state_dict": state_dict,
            "frozen_confidence_calibration": {
                "mode": mode,
                "source_checkpoint": str(args.checkpoint.resolve()),
                "seed": args.seed,
                "history": history,
            },
        }
        torch.save(checkpoint, checkpoint_path)
        evaluation = evaluate_updater_checkpoint(
            checkpoint_path,
            args.samples,
            args.output_dir / f"evaluation_{mode}",
            split="val",
            device=str(device),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            bootstrap_iterations=args.bootstrap,
            presence_threshold=args.presence_threshold,
            change_threshold=args.change_threshold,
        )
        report["modes"][mode] = {
            "checkpoint": str(checkpoint_path.resolve()),
            "history": history,
            "evaluation": evaluation,
        }
    mean_eval = report["modes"]["mean"]["evaluation"]
    product_eval = report["modes"]["product"]["evaluation"]
    unchanged_metrics = (
        "macro_f1",
        "edit_accuracy",
        "false_edit_rate",
        "missed_update_rate",
        "mean_raster_iou",
        "mean_polygon_iou",
        "topology_valid_rate",
    )
    report["main_task_invariance"] = {
        name: float(product_eval[name]) - float(mean_eval[name])
        for name in unchanged_metrics
    }
    report["calibration_delta_product_minus_mean"] = {
        name: float(product_eval[name]) - float(mean_eval[name])
        for name in ("ece", "brier", "nll", "aurc")
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["calibration_delta_product_minus_mean"], indent=2))


if __name__ == "__main__":
    main()
