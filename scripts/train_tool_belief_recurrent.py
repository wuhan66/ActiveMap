#!/usr/bin/env python3
"""Train the cumulative recurrent Tool-to-Belief adapter with safety control."""

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

from activemap.agent.tool_belief_data import ToolBeliefSequenceExample
from activemap.agent.tool_belief_model import (
    ANCHORED_BELIEF_FEATURE_DIM,
    BELIEF_FEATURE_DIM,
    ToolBeliefResidualNetwork,
    ToolPairBeliefResidualNetwork,
    apply_tool_belief_residual,
    encode_belief,
    encode_recommendation,
)
from activemap.agent.tool_features import (
    TOOL_RESULT_FEATURE_DIM,
    TOOL_RESULT_FEATURE_NAMES,
    encode_tool_result,
)
from activemap.models import EditOperation
from scripts.train_tool_belief import (
    _classification_metrics,
    _expected_calibration_error,
    _macro_f1,
    _rates,
)

EDIT_ORDER = list(EditOperation)


@dataclass(frozen=True)
class RecurrentLossWeights:
    teacher_kl: float = 1.0
    operation: float = 0.5
    calibration: float = 0.2
    geometry: float = 0.1
    false_edit: float = 1.0
    missed_edit: float = 0.5
    noop_probability: float = 1.0
    noop_confidence: float = 0.2
    noop_geometry: float = 0.1


DEFAULT_LOSS_WEIGHTS = RecurrentLossWeights()


class ToolBeliefSequenceDataset(Dataset):
    def __init__(self, path: Path) -> None:
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(ToolBeliefSequenceExample.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc
        if not rows:
            raise ValueError(f"empty recurrent Tool-to-Belief dataset: {path}")
        splits = {row.split for row in rows}
        if len(splits) != 1:
            raise ValueError(f"dataset mixes splits: {sorted(splits)}")
        self.split = next(iter(splits))
        self.initial = torch.tensor(
            [encode_belief(row.initial_belief) for row in rows], dtype=torch.float32
        )
        self.quality = torch.tensor(
            [
                [encode_tool_result(step.quality_result) for step in row.steps]
                for row in rows
            ],
            dtype=torch.float32,
        )
        self.temporal = torch.tensor(
            [
                [encode_tool_result(step.temporal_result) for step in row.steps]
                for row in rows
            ],
            dtype=torch.float32,
        )
        self.target_probabilities = torch.tensor(
            [
                [step.cumulative_target_belief.edit_probabilities for step in row.steps]
                for row in rows
            ],
            dtype=torch.float32,
        )
        self.target_confidence = torch.tensor(
            [
                [step.cumulative_target_belief.confidence for step in row.steps]
                for row in rows
            ],
            dtype=torch.float32,
        )
        self.target_geometry = torch.tensor(
            [
                [step.cumulative_target_belief.geometry_delta for step in row.steps]
                for row in rows
            ],
            dtype=torch.float32,
        )
        self.gt = torch.tensor(
            [EDIT_ORDER.index(row.gt_edit) for row in rows], dtype=torch.long
        )
        self.initial_prediction = torch.tensor(
            [EDIT_ORDER.index(row.initial_belief.predicted_edit) for row in rows],
            dtype=torch.long,
        )
        self.initial_recommendation = torch.tensor(
            [encode_recommendation(row.initial_belief) for row in rows],
            dtype=torch.float32,
        )
        self.sequence_count = len(rows)
        self.class_counts = Counter(row.gt_edit.value for row in rows)

    def __len__(self) -> int:
        return self.sequence_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.initial[index],
            self.quality[index],
            self.temporal[index],
            self.target_probabilities[index],
            self.target_confidence[index],
            self.target_geometry[index],
            self.gt[index],
            self.initial_prediction[index],
            self.initial_recommendation[index],
        )


def _next_belief_features(
    probabilities: torch.Tensor,
    confidence: torch.Tensor,
    geometry: torch.Tensor,
) -> torch.Tensor:
    entropy = -torch.sum(probabilities * probabilities.clamp_min(1e-7).log(), dim=1)
    entropy /= math.log(probabilities.shape[1])
    return torch.cat(
        [probabilities, confidence.unsqueeze(1), entropy.unsqueeze(1), geometry], dim=1
    )


def recurrent_losses(
    batch: tuple[torch.Tensor, ...],
    model: nn.Module,
    class_weights: torch.Tensor,
    *,
    max_logit_delta: float,
    geometry_scale: float,
    fusion_mode: str = "sequential",
    loss_weights: RecurrentLossWeights = DEFAULT_LOSS_WEIGHTS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[tuple[torch.Tensor, ...]]]:
    if fusion_mode not in {"sequential", "paired", "paired_anchored"}:
        raise ValueError(f"unsupported fusion mode: {fusion_mode}")
    initial, quality, temporal, target_p, target_c, target_g, gt, _, initial_anchor = batch
    current = initial
    current_anchor = initial_anchor
    temporal_predictions = []
    teacher_losses = []
    operation_losses = []
    calibration_losses = []
    geometry_losses = []
    false_edit_losses = []
    missed_edit_losses = []
    noop_probability_losses = []
    noop_confidence_losses = []
    noop_geometry_losses = []
    keep_index = EDIT_ORDER.index(EditOperation.KEEP)
    geometry_gt = torch.isin(
        gt,
        torch.tensor(
            [EDIT_ORDER.index(EditOperation.ADD), EDIT_ORDER.index(EditOperation.RESHAPE)],
            device=gt.device,
        ),
    )

    for step in range(quality.shape[1]):
        if fusion_mode == "sequential":
            quality_input = torch.cat([current, quality[:, step]], dim=1)
            quality_prediction = apply_tool_belief_residual(
                current,
                model(quality_input),
                max_logit_delta=max_logit_delta,
                geometry_scale=geometry_scale,
            )
            quality_p, quality_c, quality_g = quality_prediction
            noop_probability_losses.append(
                torch.mean(torch.abs(quality_p - current[:, :4]))
            )
            noop_confidence_losses.append(torch.mean(torch.abs(quality_c - current[:, 4])))
            noop_geometry_losses.append(F.smooth_l1_loss(quality_g, current[:, 6:14]))
            current = _next_belief_features(quality_p, quality_c, quality_g)
            temporal_input = torch.cat([current, temporal[:, step]], dim=1)
        else:
            zero = current.sum() * 0.0
            noop_probability_losses.append(zero)
            noop_confidence_losses.append(zero)
            noop_geometry_losses.append(zero)
            inputs = [current]
            if fusion_mode == "paired_anchored":
                inputs.append(current_anchor)
            inputs.extend([quality[:, step], temporal[:, step]])
            temporal_input = torch.cat(inputs, dim=1)
        prediction = apply_tool_belief_residual(
            current,
            model(temporal_input),
            max_logit_delta=max_logit_delta,
            geometry_scale=geometry_scale,
        )
        probabilities, confidence, geometry = prediction
        temporal_predictions.append(prediction)
        teacher_losses.append(
            F.kl_div(
                probabilities.clamp_min(1e-7).log(),
                target_p[:, step],
                reduction="batchmean",
            )
        )
        operation_losses.append(
            F.nll_loss(
                probabilities.clamp_min(1e-7).log(), gt, weight=class_weights
            )
        )
        calibration_losses.append(F.binary_cross_entropy(confidence, target_c[:, step]))
        if bool(geometry_gt.any()):
            geometry_losses.append(
                F.smooth_l1_loss(geometry[geometry_gt], target_g[geometry_gt, step])
            )
        keep_gt = gt == keep_index
        update_gt = ~keep_gt
        false_edit_losses.append(
            -torch.log(probabilities[keep_gt, keep_index].clamp_min(1e-7)).mean()
            if bool(keep_gt.any())
            else probabilities.sum() * 0.0
        )
        missed_edit_losses.append(
            -torch.log((1.0 - probabilities[update_gt, keep_index]).clamp_min(1e-7)).mean()
            if bool(update_gt.any())
            else probabilities.sum() * 0.0
        )
        current = _next_belief_features(probabilities, confidence, geometry)
        if fusion_mode == "paired_anchored":
            predicted = torch.argmax(probabilities.detach(), dim=1)
            current_anchor = torch.cat(
                [F.one_hot(predicted, num_classes=len(EDIT_ORDER)).float(),
                 torch.zeros((predicted.shape[0], 1), device=predicted.device)],
                dim=1,
            )

    zero = initial.sum() * 0.0

    def mean(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).mean() if values else zero

    components = {
        "teacher_kl": mean(teacher_losses),
        "operation": mean(operation_losses),
        "calibration": mean(calibration_losses),
        "geometry": mean(geometry_losses),
        "false_edit": mean(false_edit_losses),
        "missed_edit": mean(missed_edit_losses),
        "noop_probability": mean(noop_probability_losses),
        "noop_confidence": mean(noop_confidence_losses),
        "noop_geometry": mean(noop_geometry_losses),
    }
    total = sum(
        getattr(loss_weights, name) * value for name, value in components.items()
    )
    return total, components, temporal_predictions


def _class_weights(
    dataset: ToolBeliefSequenceDataset, device: torch.device, power: float
) -> torch.Tensor:
    counts = torch.bincount(dataset.gt, minlength=len(EDIT_ORDER)).float()
    weights = (counts.sum() / counts.clamp_min(1.0)).pow(power)
    weights /= weights.mean()
    return weights.to(device)


def _operation_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    confidence: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float]:
    false_edit, missed_edit = _rates(target, prediction)
    metrics = {
        f"{prefix}accuracy": float(np.mean(target == prediction)),
        f"{prefix}macro_f1": _macro_f1(target, prediction),
        f"{prefix}false_edit_rate": false_edit,
        f"{prefix}missed_edit_rate": missed_edit,
        f"{prefix}expected_calibration_error": _expected_calibration_error(
            confidence, target == prediction
        ),
    }
    metrics.update(_classification_metrics(target, prediction, prefix=prefix))
    return metrics


def run_epoch(
    loader: DataLoader,
    model: nn.Module,
    class_weights: torch.Tensor,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_logit_delta: float,
    geometry_scale: float,
    fusion_mode: str,
    loss_weights: RecurrentLossWeights,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums: Counter[str] = Counter()
    sample_count = 0
    targets = []
    initial_predictions = []
    initial_confidences = []
    step_predictions = [[], [], []]
    step_confidences = [[], [], []]
    teacher_l1_sum = 0.0
    iterator = tqdm(loader, leave=False, unit="batch", desc="train" if training else "val")
    for raw_batch in iterator:
        batch = tuple(item.to(device) for item in raw_batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            total, components, predictions = recurrent_losses(
                batch,
                model,
                class_weights,
                max_logit_delta=max_logit_delta,
                geometry_scale=geometry_scale,
                fusion_mode=fusion_mode,
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
        target = batch[6].cpu().numpy()
        targets.append(target)
        initial_predictions.append(batch[7].cpu().numpy())
        initial_confidences.append(batch[0][:, 4].cpu().numpy())
        for step, prediction in enumerate(predictions):
            probabilities, confidence, _ = prediction
            step_predictions[step].append(torch.argmax(probabilities, dim=1).cpu().numpy())
            step_confidences[step].append(confidence.detach().cpu().numpy())
            teacher_l1_sum += float(
                torch.abs(probabilities.detach() - batch[3][:, step]).mean(dim=1).sum()
            )
        iterator.set_postfix(loss=f"{float(total.detach()):.4f}")

    target = np.concatenate(targets)
    initial_prediction = np.concatenate(initial_predictions)
    initial_confidence = np.concatenate(initial_confidences)
    metrics = {name: value / sample_count for name, value in sums.items()}
    metrics["teacher_probability_l1"] = teacher_l1_sum / (sample_count * 3)
    metrics.update(
        _operation_metrics(
            target, initial_prediction, initial_confidence, prefix="baseline_"
        )
    )
    for step in range(3):
        metrics.update(
            _operation_metrics(
                target,
                np.concatenate(step_predictions[step]),
                np.concatenate(step_confidences[step]),
                prefix=f"step{step + 1}_",
            )
        )
    return metrics


def _save_checkpoint(
    path: Path,
    model: nn.Module,
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
            "model_kind": (
                "paired_quality_temporal_anchored"
                if args.fusion_mode == "paired_anchored"
                else (
                    "paired_quality_temporal"
                    if args.fusion_mode == "paired"
                    else "single_tool_residual"
                )
            ),
            "model_belief_feature_dim": (
                ANCHORED_BELIEF_FEATURE_DIM
                if args.fusion_mode == "paired_anchored"
                else BELIEF_FEATURE_DIM
            ),
            "training_protocol": (
                "anchored_paired_cumulative_recurrent_v4"
                if args.fusion_mode == "paired_anchored"
                else (
                    "paired_cumulative_recurrent_v3"
                    if args.fusion_mode == "paired"
                    else "cumulative_recurrent_v2"
                )
            ),
        },
        path,
    )


def _selection_score(metrics: dict[str, float], safety_margin: float) -> float:
    false_violation = max(
        metrics["step3_false_edit_rate"]
        - metrics["baseline_false_edit_rate"]
        - safety_margin,
        0.0,
    )
    missed_violation = max(
        metrics["step3_missed_edit_rate"]
        - metrics["baseline_missed_edit_rate"]
        - safety_margin,
        0.0,
    )
    return (
        metrics["step3_macro_f1"]
        - 2.0 * false_violation
        - missed_violation
        - 5.0 * metrics["noop_probability"]
    )


def _safety_feasible(metrics: dict[str, float], safety_margin: float) -> bool:
    return (
        metrics["step3_false_edit_rate"]
        <= metrics["baseline_false_edit_rate"] + safety_margin + 1e-12
        and metrics["step3_missed_edit_rate"]
        <= metrics["baseline_missed_edit_rate"] + safety_margin + 1e-12
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--fusion-mode",
        choices=("sequential", "paired", "paired_anchored"),
        default="sequential",
    )
    parser.add_argument("--max-logit-delta", type=float, default=1.5)
    parser.add_argument("--geometry-scale", type=float, default=0.15)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--safety-margin", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--teacher-kl-weight", type=float, default=1.0)
    parser.add_argument("--operation-weight", type=float, default=0.5)
    parser.add_argument("--calibration-weight", type=float, default=0.2)
    parser.add_argument("--geometry-weight", type=float, default=0.1)
    parser.add_argument("--false-edit-weight", type=float, default=1.0)
    parser.add_argument("--missed-edit-weight", type=float, default=0.5)
    parser.add_argument("--noop-probability-weight", type=float, default=1.0)
    parser.add_argument("--noop-confidence-weight", type=float, default=0.2)
    parser.add_argument("--noop-geometry-weight", type=float, default=0.1)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("epochs, batch-size, and patience must be positive")
    if not 0.0 <= args.class_weight_power <= 1.0:
        raise ValueError("class-weight-power must be between zero and one")
    if not 0.0 <= args.safety_margin <= 1.0:
        raise ValueError("safety-margin must be between zero and one")
    args.loss_weights = RecurrentLossWeights(
        teacher_kl=args.teacher_kl_weight,
        operation=args.operation_weight,
        calibration=args.calibration_weight,
        geometry=args.geometry_weight,
        false_edit=args.false_edit_weight,
        missed_edit=args.missed_edit_weight,
        noop_probability=args.noop_probability_weight,
        noop_confidence=args.noop_confidence_weight,
        noop_geometry=args.noop_geometry_weight,
    )
    if any(value < 0.0 for value in asdict(args.loss_weights).values()):
        raise ValueError("loss weights must be non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    train_data = ToolBeliefSequenceDataset(args.train_jsonl)
    val_data = ToolBeliefSequenceDataset(args.val_jsonl)
    if train_data.split != "train" or val_data.split != "val":
        raise ValueError("train_jsonl and val_jsonl must have train and val splits")
    protected = (
        "history.jsonl",
        "best.pt",
        "best_safety.pt",
        "last.pt",
        "summary.json",
    )
    existing = [name for name in protected if (args.output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {args.output_dir}: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    if args.fusion_mode == "sequential":
        model = ToolBeliefResidualNetwork(
            hidden_dim=args.hidden_dim, dropout=args.dropout
        ).to(device)
    else:
        model = ToolPairBeliefResidualNetwork(
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            belief_feature_dim=(
                ANCHORED_BELIEF_FEATURE_DIM
                if args.fusion_mode == "paired_anchored"
                else BELIEF_FEATURE_DIM
            ),
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(args.patience // 4, 1)
    )
    class_weights = _class_weights(train_data, device, args.class_weight_power)
    protocol = (
        "anchored_paired_cumulative_recurrent_v4"
        if args.fusion_mode == "paired_anchored"
        else (
            "paired_cumulative_recurrent_v3"
            if args.fusion_mode == "paired"
            else "cumulative_recurrent_v2"
        )
    )
    config = {
        "protocol": protocol,
        "fusion_mode": args.fusion_mode,
        "train_sequences": train_data.sequence_count,
        "val_sequences": val_data.sequence_count,
        "train_class_counts": dict(train_data.class_counts),
        "val_class_counts": dict(val_data.class_counts),
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "max_logit_delta": args.max_logit_delta,
        "geometry_scale": args.geometry_scale,
        "class_weight_power": args.class_weight_power,
        "safety_margin": args.safety_margin,
        "loss_weights": asdict(args.loss_weights),
        "test_assets_read": False,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    best_score = -math.inf
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    best_safety_f1 = -math.inf
    best_safety_epoch: int | None = None
    best_safety_metrics: dict[str, float] | None = None
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
                fusion_mode=args.fusion_mode,
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
                    fusion_mode=args.fusion_mode,
                    loss_weights=args.loss_weights,
                )
            scheduler.step(val_metrics["loss"])
            score = _selection_score(val_metrics, args.safety_margin)
            row = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "selection_score": score,
                "train": train_metrics,
                "val": val_metrics,
            }
            history.write(json.dumps(row) + "\n")
            history.flush()
            print(
                f"epoch={epoch}/{args.epochs} train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_step3_macro_f1={val_metrics['step3_macro_f1']:.6f} "
                f"val_false_edit={val_metrics['step3_false_edit_rate']:.6f} "
                f"val_missed_edit={val_metrics['step3_missed_edit_rate']:.6f} "
                f"selection_score={score:.6f}",
                flush=True,
            )
            _save_checkpoint(
                args.output_dir / "last.pt",
                model,
                epoch=epoch,
                metrics=val_metrics,
                args=args,
            )
            if score > best_score + 1e-8:
                best_score = score
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
            if (
                _safety_feasible(val_metrics, args.safety_margin)
                and val_metrics["step3_macro_f1"] > best_safety_f1 + 1e-8
            ):
                best_safety_f1 = val_metrics["step3_macro_f1"]
                best_safety_epoch = epoch
                best_safety_metrics = dict(val_metrics)
                _save_checkpoint(
                    args.output_dir / "best_safety.pt",
                    model,
                    epoch=epoch,
                    metrics=val_metrics,
                    args=args,
                )
            if stale >= args.patience:
                break

    summary = {
        **config,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "best_val_metrics": best_metrics,
        "best_safety_epoch": best_safety_epoch,
        "best_safety_val_metrics": best_safety_metrics,
        "epochs_completed": epoch,
        "early_stopped": epoch < args.epochs,
        "selection_criterion": (
            "step3 macro F1 minus false/missed safety violations and quality drift"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
