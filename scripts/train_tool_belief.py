#!/usr/bin/env python3
"""Train the grounded residual Tool-to-Belief adapter with validation control."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from activemap.agent.tool_belief_data import ToolBeliefExample
from activemap.agent.tool_belief_model import (
    BELIEF_FEATURE_DIM,
    ToolBeliefResidualNetwork,
    apply_tool_belief_residual,
    encode_belief,
)
from activemap.agent.tool_features import (
    TOOL_RESULT_FEATURE_DIM,
    TOOL_RESULT_FEATURE_NAMES,
    encode_tool_result,
)
from activemap.models import EditOperation

EDIT_ORDER = list(EditOperation)


@dataclass(frozen=True)
class LossWeights:
    teacher_kl: float = 1.0
    operation: float = 0.5
    calibration: float = 0.2
    geometry: float = 0.1
    noop_probability_drift: float = 1.0
    noop_confidence_drift: float = 0.2
    noop_geometry_drift: float = 0.1


DEFAULT_LOSS_WEIGHTS = LossWeights()


class ToolBeliefDataset(Dataset):
    def __init__(self, path: Path) -> None:
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        rows.append(ToolBeliefExample.model_validate_json(line))
                    except Exception as exc:
                        raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc
        if not rows:
            raise ValueError(f"empty Tool-to-Belief dataset: {path}")
        splits = {row.split for row in rows}
        if len(splits) != 1:
            raise ValueError(f"dataset mixes splits: {sorted(splits)}")
        self.split = next(iter(splits))
        self.inputs = torch.tensor(
            [encode_belief(row.prior_belief) + encode_tool_result(row.tool_result) for row in rows],
            dtype=torch.float32,
        )
        self.target_probabilities = torch.tensor(
            [row.target_belief.edit_probabilities for row in rows], dtype=torch.float32
        )
        self.target_confidence = torch.tensor(
            [row.target_belief.confidence for row in rows],
            dtype=torch.float32,
        )
        self.target_geometry = torch.tensor(
            [row.target_belief.geometry_delta for row in rows], dtype=torch.float32
        )
        self.gt = torch.tensor([EDIT_ORDER.index(row.gt_edit) for row in rows], dtype=torch.long)
        self.semantic = torch.tensor(
            [row.metadata.get("target_kind") == "teacher_belief_update" for row in rows],
            dtype=torch.bool,
        )
        self.geometry_mask = self.semantic & torch.isin(
            self.gt,
            torch.tensor(
                [EDIT_ORDER.index(EditOperation.ADD), EDIT_ORDER.index(EditOperation.RESHAPE)]
            ),
        )
        self.record_count = len(rows)
        self.episode_count = len({row.episode_id for row in rows})
        self.class_counts = Counter(
            row.gt_edit.value
            for row in rows
            if row.metadata.get("target_kind") == "teacher_belief_update"
        )

    def __len__(self) -> int:
        return self.record_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.inputs[index],
            self.target_probabilities[index],
            self.target_confidence[index],
            self.target_geometry[index],
            self.gt[index],
            self.semantic[index],
            self.geometry_mask[index],
        )


def _macro_f1(target: np.ndarray, prediction: np.ndarray) -> float:
    scores = []
    for index in range(len(EDIT_ORDER)):
        tp = int(np.sum((target == index) & (prediction == index)))
        fp = int(np.sum((target != index) & (prediction == index)))
        fn = int(np.sum((target == index) & (prediction != index)))
        denominator = 2 * tp + fp + fn
        scores.append((2 * tp / denominator) if denominator else 0.0)
    return float(np.mean(scores))


def _classification_metrics(
    target: np.ndarray, prediction: np.ndarray, *, prefix: str = ""
) -> dict[str, float]:
    metrics = {}
    for index, operation in enumerate(EDIT_ORDER):
        tp = int(np.sum((target == index) & (prediction == index)))
        fp = int(np.sum((target != index) & (prediction == index)))
        fn = int(np.sum((target == index) & (prediction != index)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        name = operation.value.lower()
        metrics[f"{prefix}{name}_precision"] = precision
        metrics[f"{prefix}{name}_recall"] = recall
        metrics[f"{prefix}{name}_f1"] = f1
        for predicted_index, predicted_operation in enumerate(EDIT_ORDER):
            metrics[
                f"{prefix}{name}_as_{predicted_operation.value.lower()}_rate"
            ] = float(
                np.sum((target == index) & (prediction == predicted_index))
                / max(np.sum(target == index), 1)
            )
    return metrics


def _expected_calibration_error(
    confidence: np.ndarray, correctness: np.ndarray, *, bins: int = 10
) -> float:
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if np.any(selected):
            error += float(np.mean(selected)) * abs(
                float(np.mean(correctness[selected])) - float(np.mean(confidence[selected]))
            )
    return error


def _rates(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    keep = EDIT_ORDER.index(EditOperation.KEEP)
    keep_count = max(int(np.sum(target == keep)), 1)
    update_count = max(int(np.sum(target != keep)), 1)
    false_edit = float(np.sum((target == keep) & (prediction != keep)) / keep_count)
    missed_edit = float(np.sum((target != keep) & (prediction == keep)) / update_count)
    return false_edit, missed_edit


def _losses(
    batch: tuple[torch.Tensor, ...],
    model: nn.Module,
    class_weights: torch.Tensor,
    *,
    max_logit_delta: float,
    geometry_scale: float,
    loss_weights: LossWeights = DEFAULT_LOSS_WEIGHTS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
    (
        inputs,
        target_probabilities,
        target_confidence,
        target_geometry,
        gt,
        semantic,
        geometry_mask,
    ) = batch
    prediction = apply_tool_belief_residual(
        inputs[:, :BELIEF_FEATURE_DIM],
        model(inputs),
        max_logit_delta=max_logit_delta,
        geometry_scale=geometry_scale,
    )
    probabilities, confidence, geometry = prediction
    teacher = F.kl_div(
        probabilities.clamp_min(1e-7).log(), target_probabilities, reduction="batchmean"
    )
    if bool(semantic.any()):
        operation = F.nll_loss(
            probabilities[semantic].clamp_min(1e-7).log(),
            gt[semantic],
            weight=class_weights,
        )
    else:
        operation = probabilities.sum() * 0.0
    calibration = F.binary_cross_entropy(confidence, target_confidence)
    if bool(geometry_mask.any()):
        geometry_loss = F.smooth_l1_loss(
            geometry[geometry_mask], target_geometry[geometry_mask]
        )
    else:
        geometry_loss = geometry.sum() * 0.0
    noop = ~semantic
    noop_probability_drift = (
        torch.mean(torch.abs(probabilities[noop] - inputs[noop, :4]))
        if bool(noop.any())
        else probabilities.sum() * 0.0
    )
    noop_confidence_drift = (
        torch.mean(torch.abs(confidence[noop] - inputs[noop, 4]))
        if bool(noop.any())
        else confidence.sum() * 0.0
    )
    noop_geometry_drift = (
        F.smooth_l1_loss(geometry[noop], inputs[noop, 6:14])
        if bool(noop.any())
        else geometry.sum() * 0.0
    )
    components = {
        "teacher_kl": teacher,
        "operation": operation,
        "calibration": calibration,
        "geometry": geometry_loss,
        "noop_probability_drift": noop_probability_drift,
        "noop_confidence_drift": noop_confidence_drift,
        "noop_geometry_drift": noop_geometry_drift,
    }
    total = (
        loss_weights.teacher_kl * teacher
        + loss_weights.operation * operation
        + loss_weights.calibration * calibration
        + loss_weights.geometry * geometry_loss
        + loss_weights.noop_probability_drift * noop_probability_drift
        + loss_weights.noop_confidence_drift * noop_confidence_drift
        + loss_weights.noop_geometry_drift * noop_geometry_drift
    )
    return total, components, prediction


def _class_weights(dataset: ToolBeliefDataset, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(dataset.gt[dataset.semantic], minlength=len(EDIT_ORDER)).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    weights /= weights.mean()
    return weights.to(device)


def run_epoch(
    loader: DataLoader,
    model: nn.Module,
    class_weights: torch.Tensor,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_logit_delta: float,
    geometry_scale: float,
    loss_weights: LossWeights = DEFAULT_LOSS_WEIGHTS,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums: Counter[str] = Counter()
    predictions = []
    targets = []
    baseline_predictions = []
    semantic_confidences = []
    baseline_semantic_confidences = []
    sample_count = 0
    diagnostic_sums: Counter[str] = Counter()
    diagnostic_counts: Counter[str] = Counter()
    iterator = tqdm(loader, leave=False, unit="batch", desc="train" if training else "val")
    for raw_batch in iterator:
        batch = tuple(item.to(device) for item in raw_batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            total, components, prediction = _losses(
                batch,
                model,
                class_weights,
                max_logit_delta=max_logit_delta,
                geometry_scale=geometry_scale,
                loss_weights=loss_weights,
            )
            if training:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        count = int(batch[0].shape[0])
        sample_count += count
        sums["loss"] += float(total.detach()) * count
        for name, value in components.items():
            sums[name] += float(value.detach()) * count
        probabilities = prediction[0].detach()
        confidence = prediction[1].detach()
        geometry = prediction[2].detach()
        semantic = batch[5]
        noop = ~semantic
        diagnostic_sums["teacher_probability_l1"] += float(
            torch.abs(probabilities - batch[1]).mean(dim=1).sum()
        )
        diagnostic_sums["confidence_mae"] += float(torch.abs(confidence - batch[2]).sum())
        diagnostic_sums["baseline_teacher_probability_l1"] += float(
            torch.abs(batch[0][:, :4] - batch[1]).mean(dim=1).sum()
        )
        diagnostic_sums["baseline_confidence_mae"] += float(
            torch.abs(batch[0][:, 4] - batch[2]).sum()
        )
        diagnostic_counts["all"] += count
        if bool(noop.any()):
            noop_count = int(noop.sum())
            diagnostic_sums["noop_probability_drift"] += float(
                torch.abs(probabilities[noop] - batch[0][noop, :4]).mean(dim=1).sum()
            )
            diagnostic_sums["noop_confidence_drift"] += float(
                torch.abs(confidence[noop] - batch[0][noop, 4]).sum()
            )
            diagnostic_sums["noop_geometry_drift"] += float(
                torch.abs(geometry[noop] - batch[0][noop, 6:14]).mean(dim=1).sum()
            )
            diagnostic_counts["noop"] += noop_count
        geometry_mask = batch[6]
        if bool(geometry_mask.any()):
            geometry_count = int(geometry_mask.sum())
            diagnostic_sums["geometry_mae"] += float(
                torch.abs(geometry[geometry_mask] - batch[3][geometry_mask])
                .mean(dim=1)
                .sum()
            )
            diagnostic_sums["baseline_geometry_mae"] += float(
                torch.abs(batch[0][geometry_mask, 6:14] - batch[3][geometry_mask])
                .mean(dim=1)
                .sum()
            )
            diagnostic_counts["geometry"] += geometry_count
        if bool(semantic.any()):
            semantic_prediction = torch.argmax(probabilities[semantic], dim=1)
            predictions.append(semantic_prediction.cpu().numpy())
            targets.append(batch[4][semantic].cpu().numpy())
            baseline_predictions.append(
                torch.argmax(batch[0][semantic, :4], dim=1).cpu().numpy()
            )
            semantic_confidences.append(confidence[semantic].cpu().numpy())
            baseline_semantic_confidences.append(batch[0][semantic, 4].cpu().numpy())
        iterator.set_postfix(loss=f"{float(total.detach()):.4f}")

    target = np.concatenate(targets)
    predicted = np.concatenate(predictions)
    baseline = np.concatenate(baseline_predictions)
    confidence_values = np.concatenate(semantic_confidences)
    baseline_confidence_values = np.concatenate(baseline_semantic_confidences)
    false_edit, missed_edit = _rates(target, predicted)
    baseline_false_edit, baseline_missed_edit = _rates(target, baseline)
    metrics = {name: value / sample_count for name, value in sums.items()}
    all_count = max(diagnostic_counts["all"], 1)
    noop_count = max(diagnostic_counts["noop"], 1)
    geometry_count = max(diagnostic_counts["geometry"], 1)
    metrics.update(
        {
            "accuracy": float(np.mean(predicted == target)),
            "macro_f1": _macro_f1(target, predicted),
            "false_edit_rate": false_edit,
            "missed_edit_rate": missed_edit,
            "baseline_accuracy": float(np.mean(baseline == target)),
            "baseline_macro_f1": _macro_f1(target, baseline),
            "baseline_false_edit_rate": baseline_false_edit,
            "baseline_missed_edit_rate": baseline_missed_edit,
            "teacher_probability_l1": diagnostic_sums["teacher_probability_l1"]
            / all_count,
            "confidence_mae": diagnostic_sums["confidence_mae"] / all_count,
            "geometry_mae": diagnostic_sums["geometry_mae"] / geometry_count,
            "noop_probability_drift": diagnostic_sums["noop_probability_drift"]
            / noop_count,
            "noop_confidence_drift": diagnostic_sums["noop_confidence_drift"] / noop_count,
            "noop_geometry_drift": diagnostic_sums["noop_geometry_drift"] / noop_count,
            "baseline_teacher_probability_l1": diagnostic_sums[
                "baseline_teacher_probability_l1"
            ]
            / all_count,
            "baseline_confidence_mae": diagnostic_sums["baseline_confidence_mae"]
            / all_count,
            "baseline_geometry_mae": diagnostic_sums["baseline_geometry_mae"]
            / geometry_count,
            "expected_calibration_error": _expected_calibration_error(
                confidence_values, predicted == target
            ),
            "baseline_expected_calibration_error": _expected_calibration_error(
                baseline_confidence_values, baseline == target
            ),
        }
    )
    metrics.update(_classification_metrics(target, predicted))
    metrics.update(_classification_metrics(target, baseline, prefix="baseline_"))
    return metrics


def _save_checkpoint(
    path: Path,
    model: ToolBeliefResidualNetwork,
    *,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "max_logit_delta": args.max_logit_delta,
            "geometry_scale": args.geometry_scale,
            "belief_feature_dim": BELIEF_FEATURE_DIM,
            "tool_result_feature_dim": TOOL_RESULT_FEATURE_DIM,
            "tool_result_feature_names": TOOL_RESULT_FEATURE_NAMES,
            "loss_weights": asdict(args.loss_weights),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-logit-delta", type=float, default=3.0)
    parser.add_argument("--geometry-scale", type=float, default=0.25)
    parser.add_argument("--teacher-kl-weight", type=float, default=1.0)
    parser.add_argument("--operation-weight", type=float, default=0.5)
    parser.add_argument("--calibration-weight", type=float, default=0.2)
    parser.add_argument("--geometry-weight", type=float, default=0.1)
    parser.add_argument("--noop-probability-weight", type=float, default=1.0)
    parser.add_argument("--noop-confidence-weight", type=float, default=0.2)
    parser.add_argument("--noop-geometry-weight", type=float, default=0.1)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("epochs, batch-size, and patience must be positive")
    args.loss_weights = LossWeights(
        teacher_kl=args.teacher_kl_weight,
        operation=args.operation_weight,
        calibration=args.calibration_weight,
        geometry=args.geometry_weight,
        noop_probability_drift=args.noop_probability_weight,
        noop_confidence_drift=args.noop_confidence_weight,
        noop_geometry_drift=args.noop_geometry_weight,
    )
    if any(value < 0.0 for value in asdict(args.loss_weights).values()):
        raise ValueError("loss weights must be non-negative")
    if not any(value > 0.0 for value in asdict(args.loss_weights).values()):
        raise ValueError("at least one loss weight must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    train_data = ToolBeliefDataset(args.train_jsonl)
    val_data = ToolBeliefDataset(args.val_jsonl)
    if train_data.split != "train" or val_data.split != "val":
        raise ValueError("train_jsonl and val_jsonl must have train and val splits")
    audit_report = None
    if args.audit_report is not None:
        audit_report = json.loads(args.audit_report.read_text(encoding="utf-8"))
        if audit_report.get("passed") is not True:
            raise ValueError(f"data audit did not pass: {args.audit_report}")
        audited_counts = audit_report.get("record_counts", {})
        if audited_counts != {
            "train": train_data.record_count,
            "val": val_data.record_count,
        }:
            raise ValueError("audit record counts do not match training inputs")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    model = ToolBeliefResidualNetwork(
        hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(args.patience // 3, 1)
    )
    class_weights = _class_weights(train_data, device)
    protected_outputs = ("history.jsonl", "best.pt", "last.pt", "summary.json")
    existing_outputs = [name for name in protected_outputs if (args.output_dir / name).exists()]
    if existing_outputs:
        raise FileExistsError(
            f"refusing to overwrite existing run artifacts in {args.output_dir}: "
            f"{existing_outputs}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "train_jsonl": str(args.train_jsonl.resolve()),
        "val_jsonl": str(args.val_jsonl.resolve()),
        "audit_report": str(args.audit_report.resolve()) if args.audit_report else None,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "max_logit_delta": args.max_logit_delta,
        "geometry_scale": args.geometry_scale,
        "patience": args.patience,
        "seed": args.seed,
        "loss_weights": asdict(args.loss_weights),
        "belief_feature_dim": BELIEF_FEATURE_DIM,
        "tool_result_feature_dim": TOOL_RESULT_FEATURE_DIM,
        "torch_version": torch.__version__,
        "test_assets_read": False,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    data_summary = {
        "train_records": train_data.record_count,
        "train_episodes": train_data.episode_count,
        "train_class_counts": dict(train_data.class_counts),
        "val_records": val_data.record_count,
        "val_episodes": val_data.episode_count,
        "val_class_counts": dict(val_data.class_counts),
        "seed": args.seed,
        "test_assets_read": False,
    }
    (args.output_dir / "data_summary.json").write_text(
        json.dumps(data_summary, indent=2) + "\n", encoding="utf-8"
    )

    best_loss = math.inf
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    stale = 0
    history_path = args.output_dir / "history.jsonl"
    with history_path.open("w", encoding="utf-8") as history:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                train_loader,
                model,
                class_weights,
                device=device,
                optimizer=optimizer,
                max_logit_delta=args.max_logit_delta,
                geometry_scale=args.geometry_scale,
                loss_weights=args.loss_weights,
            )
            with torch.inference_mode():
                val_metrics = run_epoch(
                    val_loader,
                    model,
                    class_weights,
                    device=device,
                    optimizer=None,
                    max_logit_delta=args.max_logit_delta,
                    geometry_scale=args.geometry_scale,
                    loss_weights=args.loss_weights,
                )
            scheduler.step(val_metrics["loss"])
            row = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
            }
            history.write(json.dumps(row) + "\n")
            history.flush()
            print(
                f"epoch={epoch}/{args.epochs} train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} val_macro_f1={val_metrics['macro_f1']:.6f} "
                f"val_false_edit={val_metrics['false_edit_rate']:.6f}",
                flush=True,
            )
            _save_checkpoint(
                args.output_dir / "last.pt",
                model,
                epoch=epoch,
                metrics=val_metrics,
                args=args,
            )
            if val_metrics["loss"] < best_loss - 1e-6:
                best_loss = val_metrics["loss"]
                best_epoch = epoch
                best_metrics = dict(val_metrics)
                stale = 0
                _save_checkpoint(
                    args.output_dir / "best.pt",
                    model,
                    epoch=epoch,
                    metrics=val_metrics,
                    args=args,
                )
            else:
                stale += 1
            if stale >= args.patience:
                break

    summary = {
        **data_summary,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "best_val_metrics": best_metrics,
        "epochs_completed": epoch,
        "early_stopped": epoch < args.epochs,
        "loss_weights": asdict(args.loss_weights),
        "audit_passed": audit_report.get("passed") if audit_report else None,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
