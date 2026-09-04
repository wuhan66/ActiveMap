#!/usr/bin/env python3
"""Train a tool-dependent residual updater after evidence acquisition."""

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

from activemap.agent.tool_belief_data import PostAcquisitionToolPairExample
from activemap.agent.tool_belief_model import (
    BELIEF_FEATURE_DIM,
    ToolPairBeliefResidualNetwork,
    apply_tool_belief_residual,
    encode_belief,
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
class LossWeights:
    teacher_kl: float = 1.0
    operation: float = 0.5
    calibration: float = 0.2
    geometry: float = 0.1
    false_edit: float = 1.0
    missed_edit: float = 0.5
    tool_contrastive: float = 0.0


class PostAcquisitionDataset(Dataset):
    def __init__(self, path: Path) -> None:
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(PostAcquisitionToolPairExample.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc
        if not rows:
            raise ValueError(f"empty post-acquisition dataset: {path}")
        splits = {row.split for row in rows}
        if len(splits) != 1:
            raise ValueError(f"dataset mixes splits: {sorted(splits)}")
        self.split = next(iter(splits))
        self.belief = torch.tensor(
            [encode_belief(row.post_acquisition_belief) for row in rows],
            dtype=torch.float32,
        )
        self.quality = torch.tensor(
            [encode_tool_result(row.quality_result) for row in rows], dtype=torch.float32
        )
        self.temporal = torch.tensor(
            [encode_tool_result(row.temporal_result) for row in rows], dtype=torch.float32
        )
        self.target_p = torch.tensor(
            [row.target_belief.edit_probabilities for row in rows], dtype=torch.float32
        )
        self.target_c = torch.tensor(
            [row.target_belief.confidence for row in rows], dtype=torch.float32
        )
        self.target_g = torch.tensor(
            [row.target_belief.geometry_delta for row in rows], dtype=torch.float32
        )
        self.gt = torch.tensor(
            [EDIT_ORDER.index(row.gt_edit) for row in rows], dtype=torch.long
        )
        self.deployed_prediction = torch.tensor(
            [EDIT_ORDER.index(row.post_acquisition_belief.predicted_edit) for row in rows],
            dtype=torch.long,
        )
        self.operation_threshold = torch.tensor(
            [float(row.metadata["operation_update_threshold"]) for row in rows],
            dtype=torch.float32,
        )
        self.count = len(rows)
        self.task_count = len({row.task_id for row in rows})
        self.class_counts = Counter(row.gt_edit.value for row in rows)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.belief[index],
            self.quality[index],
            self.temporal[index],
            self.target_p[index],
            self.target_c[index],
            self.target_g[index],
            self.gt[index],
            self.deployed_prediction[index],
            self.operation_threshold[index],
        )


def post_acquisition_losses(
    batch: tuple[torch.Tensor, ...],
    model: nn.Module,
    class_weights: torch.Tensor,
    *,
    max_logit_delta: float,
    geometry_scale: float,
    weights: LossWeights,
    zero_tool_features: bool = False,
    tool_contrastive_margin: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
    belief, quality, temporal, target_p, target_c, target_g, gt, *_ = batch
    if zero_tool_features:
        quality = torch.zeros_like(quality)
        temporal = torch.zeros_like(temporal)
    features = torch.cat([belief, quality, temporal], dim=1)
    residuals = model(features)
    prediction = apply_tool_belief_residual(
        belief,
        residuals,
        max_logit_delta=max_logit_delta,
        geometry_scale=geometry_scale,
    )
    if len(residuals) == 4:
        prediction = (*prediction, residuals[3])
    probabilities, confidence, geometry = prediction[:3]
    keep = EDIT_ORDER.index(EditOperation.KEEP)
    keep_gt = gt == keep
    update_gt = ~keep_gt
    geometry_gt = torch.isin(
        gt,
        torch.tensor(
            [EDIT_ORDER.index(EditOperation.ADD), EDIT_ORDER.index(EditOperation.RESHAPE)],
            device=gt.device,
        ),
    )
    zero = probabilities.sum() * 0.0
    tool_contrastive = zero
    if not zero_tool_features and belief.shape[0] > 1:
        shuffled_features = torch.cat(
            [belief, torch.roll(quality, 1, 0), torch.roll(temporal, 1, 0)], dim=1
        )
        shuffled_probabilities, _, _ = apply_tool_belief_residual(
            belief,
            model(shuffled_features),
            max_logit_delta=max_logit_delta,
            geometry_scale=geometry_scale,
        )
        full_nll = F.nll_loss(
            probabilities.clamp_min(1e-7).log(),
            gt,
            weight=class_weights,
            reduction="none",
        )
        shuffled_nll = F.nll_loss(
            shuffled_probabilities.clamp_min(1e-7).log(),
            gt,
            weight=class_weights,
            reduction="none",
        )
        tool_contrastive = torch.relu(
            tool_contrastive_margin + full_nll - shuffled_nll
        ).mean()
    components = {
        "teacher_kl": F.kl_div(
            probabilities.clamp_min(1e-7).log(), target_p, reduction="batchmean"
        ),
        "operation": F.nll_loss(
            probabilities.clamp_min(1e-7).log(), gt, weight=class_weights
        ),
        "calibration": F.binary_cross_entropy(confidence, target_c),
        "geometry": (
            F.smooth_l1_loss(geometry[geometry_gt], target_g[geometry_gt])
            if bool(geometry_gt.any())
            else zero
        ),
        "false_edit": (
            -torch.log(probabilities[keep_gt, keep].clamp_min(1e-7)).mean()
            if bool(keep_gt.any())
            else zero
        ),
        "missed_edit": (
            -torch.log((1.0 - probabilities[update_gt, keep]).clamp_min(1e-7)).mean()
            if bool(update_gt.any())
            else zero
        ),
        "tool_contrastive": tool_contrastive,
    }
    total = sum(getattr(weights, name) * value for name, value in components.items())
    return total, components, prediction


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


def _threshold_predictions(
    probabilities: torch.Tensor, thresholds: torch.Tensor
) -> torch.Tensor:
    update_prediction = torch.argmax(probabilities[:, 1:], dim=1) + 1
    return torch.where(
        1.0 - probabilities[:, 0] >= thresholds,
        update_prediction,
        torch.zeros_like(update_prediction),
    )


def promotion_gate(
    metrics: dict[str, float],
    *,
    safety_margin: float,
    min_quality_delta: float,
    min_tool_delta: float,
    max_calibration_degradation: float = 0.03,
) -> dict[str, bool]:
    reference_f1 = max(
        metrics["baseline_argmax_macro_f1"],
        metrics["baseline_deployed_macro_f1"],
    )
    reference_false_edit = min(
        metrics["baseline_argmax_false_edit_rate"],
        metrics["baseline_deployed_false_edit_rate"],
    )
    checks = {
        "false_edit_safe": metrics["full_false_edit_rate"]
        <= reference_false_edit + safety_margin + 1e-12,
        "improves_baseline": metrics["full_macro_f1"]
        >= reference_f1 + min_quality_delta - 1e-12,
        "tool_dependent": metrics["full_macro_f1"]
        >= metrics["zero_tool_macro_f1"] + min_tool_delta - 1e-12,
        "calibrated": metrics["full_expected_calibration_error"]
        <= metrics["baseline_deployed_expected_calibration_error"]
        + max_calibration_degradation
        + 1e-12,
    }
    checks["passed"] = all(checks.values())
    return checks


def run_epoch(
    loader: DataLoader,
    model: nn.Module,
    class_weights: torch.Tensor,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_logit_delta: float,
    geometry_scale: float,
    weights: LossWeights,
    tool_contrastive_margin: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums: Counter[str] = Counter()
    targets: list[np.ndarray] = []
    baseline_predictions: list[np.ndarray] = []
    baseline_confidences: list[np.ndarray] = []
    deployed_predictions: list[np.ndarray] = []
    full_predictions: list[np.ndarray] = []
    full_confidences: list[np.ndarray] = []
    full_raw_predictions: list[np.ndarray] = []
    zero_predictions: list[np.ndarray] = []
    zero_confidences: list[np.ndarray] = []
    zero_raw_predictions: list[np.ndarray] = []
    full_reliabilities: list[np.ndarray] = []
    zero_reliabilities: list[np.ndarray] = []
    count_total = 0
    iterator = tqdm(loader, leave=False, unit="batch", desc="train" if training else "val")
    for raw_batch in iterator:
        batch = tuple(item.to(device) for item in raw_batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            total, components, full = post_acquisition_losses(
                batch,
                model,
                class_weights,
                max_logit_delta=max_logit_delta,
                geometry_scale=geometry_scale,
                weights=weights,
                tool_contrastive_margin=tool_contrastive_margin,
            )
            if training:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        with torch.inference_mode():
            _, _, zero = post_acquisition_losses(
                batch,
                model,
                class_weights,
                max_logit_delta=max_logit_delta,
                geometry_scale=geometry_scale,
                weights=weights,
                zero_tool_features=True,
                tool_contrastive_margin=tool_contrastive_margin,
            )
        count = int(batch[0].shape[0])
        count_total += count
        sums["loss"] += float(total.detach()) * count
        for name, value in components.items():
            sums[name] += float(value.detach()) * count
        target = batch[6].cpu().numpy()
        targets.append(target)
        baseline_predictions.append(torch.argmax(batch[0][:, :4], dim=1).cpu().numpy())
        baseline_confidences.append(batch[0][:, 4].cpu().numpy())
        deployed_predictions.append(batch[7].cpu().numpy())
        full_predictions.append(
            _threshold_predictions(full[0], batch[8]).detach().cpu().numpy()
        )
        full_raw_predictions.append(torch.argmax(full[0], dim=1).detach().cpu().numpy())
        full_confidences.append(full[1].detach().cpu().numpy())
        if len(full) == 4:
            full_reliabilities.append(full[3].detach().cpu().numpy())
        zero_predictions.append(_threshold_predictions(zero[0], batch[8]).cpu().numpy())
        zero_raw_predictions.append(torch.argmax(zero[0], dim=1).cpu().numpy())
        zero_confidences.append(zero[1].cpu().numpy())
        if len(zero) == 4:
            zero_reliabilities.append(zero[3].cpu().numpy())
        iterator.set_postfix(loss=f"{float(total.detach()):.4f}")
    target = np.concatenate(targets)
    metrics = {name: value / count_total for name, value in sums.items()}
    for prefix, predictions, confidences in (
        ("baseline_argmax_", baseline_predictions, baseline_confidences),
        ("baseline_deployed_", deployed_predictions, baseline_confidences),
        ("zero_tool_", zero_predictions, zero_confidences),
        ("zero_tool_raw_", zero_raw_predictions, zero_confidences),
        ("full_", full_predictions, full_confidences),
        ("full_raw_", full_raw_predictions, full_confidences),
    ):
        metrics.update(
            _operation_metrics(
                target,
                np.concatenate(predictions),
                np.concatenate(confidences),
                prefix=prefix,
            )
        )
    for prefix, values in (
        ("full_reliability_gate_", full_reliabilities),
        ("zero_tool_reliability_gate_", zero_reliabilities),
    ):
        if values:
            joined = np.concatenate(values)
            metrics.update(
                {
                    f"{prefix}mean": float(np.mean(joined)),
                    f"{prefix}std": float(np.std(joined)),
                    f"{prefix}p10": float(np.quantile(joined, 0.10)),
                    f"{prefix}p90": float(np.quantile(joined, 0.90)),
                }
            )
    return metrics


def _class_weights(dataset: PostAcquisitionDataset, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(dataset.gt, minlength=len(EDIT_ORDER)).float()
    weights = (counts.sum() / counts.clamp_min(1.0)).sqrt()
    return (weights / weights.mean()).to(device)


def _checkpoint(
    model: nn.Module, epoch: int, metrics: dict[str, float], args: argparse.Namespace
) -> dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "max_logit_delta": args.max_logit_delta,
        "geometry_scale": args.geometry_scale,
        "belief_feature_dim": BELIEF_FEATURE_DIM,
        "model_belief_feature_dim": BELIEF_FEATURE_DIM,
        "reliability_gate": args.reliability_gate,
        "gate_bias": args.gate_bias,
        "tool_result_feature_dim": TOOL_RESULT_FEATURE_DIM,
        "tool_result_feature_names": TOOL_RESULT_FEATURE_NAMES,
        "loss_weights": asdict(args.loss_weights),
        "model_kind": "post_acquisition_paired_quality_temporal",
        "training_protocol": (
            "post_acquisition_reliability_gated_residual_v2"
            if args.reliability_gate
            else "post_acquisition_tool_dependent_residual_v1"
        ),
    }


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
        "--reliability-gate",
        action="store_true",
        help="gate belief residuals using paired evidence reliability",
    )
    parser.add_argument(
        "--gate-bias",
        type=float,
        default=-1.5,
        help="initial reliability-gate logit (negative is conservative)",
    )
    parser.add_argument("--max-logit-delta", type=float, default=1.5)
    parser.add_argument("--geometry-scale", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--safety-margin", type=float, default=0.02)
    parser.add_argument("--min-quality-delta", type=float, default=0.01)
    parser.add_argument("--min-tool-delta", type=float, default=0.01)
    parser.add_argument("--max-calibration-degradation", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--teacher-kl-weight", type=float, default=1.0)
    parser.add_argument("--operation-weight", type=float, default=0.5)
    parser.add_argument("--calibration-weight", type=float, default=0.2)
    parser.add_argument("--geometry-weight", type=float, default=0.1)
    parser.add_argument("--false-edit-weight", type=float, default=1.0)
    parser.add_argument("--missed-edit-weight", type=float, default=0.5)
    parser.add_argument("--tool-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--tool-contrastive-margin", type=float, default=0.05)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) <= 0:
        raise ValueError("epochs, batch-size, and patience must be positive")
    if min(
        args.safety_margin,
        args.min_quality_delta,
        args.min_tool_delta,
        args.max_calibration_degradation,
    ) < 0.0:
        raise ValueError("promotion thresholds must be non-negative")
    args.loss_weights = LossWeights(
        teacher_kl=args.teacher_kl_weight,
        operation=args.operation_weight,
        calibration=args.calibration_weight,
        geometry=args.geometry_weight,
        false_edit=args.false_edit_weight,
        missed_edit=args.missed_edit_weight,
        tool_contrastive=args.tool_contrastive_weight,
    )
    if min(asdict(args.loss_weights).values()) < 0.0:
        raise ValueError("loss weights must be non-negative")
    if args.tool_contrastive_margin < 0.0:
        raise ValueError("tool-contrastive-margin must be non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train_data = PostAcquisitionDataset(args.train_jsonl)
    val_data = PostAcquisitionDataset(args.val_jsonl)
    if train_data.split != "train" or val_data.split != "val":
        raise ValueError("train_jsonl and val_jsonl must have train and val splits")
    protected = ("history.jsonl", "best.pt", "best_promoted.pt", "last.pt", "summary.json")
    existing = [name for name in protected if (args.output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {args.output_dir}: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    model = ToolPairBeliefResidualNetwork(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        reliability_gate=args.reliability_gate,
        gate_bias=args.gate_bias,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(args.patience // 4, 1)
    )
    class_weights = _class_weights(train_data, device)
    protocol = (
        "post_acquisition_reliability_gated_residual_v2"
        if args.reliability_gate
        else "post_acquisition_tool_dependent_residual_v1"
    )
    config = {
        "protocol": protocol,
        "train_examples": train_data.count,
        "val_examples": val_data.count,
        "train_tasks": train_data.task_count,
        "val_tasks": val_data.task_count,
        "train_class_counts": dict(train_data.class_counts),
        "val_class_counts": dict(val_data.class_counts),
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "reliability_gate": args.reliability_gate,
        "gate_bias": args.gate_bias,
        "max_logit_delta": args.max_logit_delta,
        "geometry_scale": args.geometry_scale,
        "safety_margin": args.safety_margin,
        "min_quality_delta": args.min_quality_delta,
        "min_tool_delta": args.min_tool_delta,
        "max_calibration_degradation": args.max_calibration_degradation,
        "loss_weights": asdict(args.loss_weights),
        "tool_contrastive_margin": args.tool_contrastive_margin,
        "test_assets_read": False,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    best_score = -math.inf
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    best_promoted_epoch: int | None = None
    best_promoted_metrics: dict[str, float] | None = None
    stale = 0
    epoch = 0
    with (args.output_dir / "history.jsonl").open("x", encoding="utf-8") as history:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                train_loader, model, class_weights, device=device, optimizer=optimizer,
                max_logit_delta=args.max_logit_delta,
                geometry_scale=args.geometry_scale, weights=args.loss_weights,
                tool_contrastive_margin=args.tool_contrastive_margin,
            )
            with torch.inference_mode():
                val_metrics = run_epoch(
                    val_loader, model, class_weights, device=device, optimizer=None,
                    max_logit_delta=args.max_logit_delta,
                    geometry_scale=args.geometry_scale, weights=args.loss_weights,
                    tool_contrastive_margin=args.tool_contrastive_margin,
                )
            scheduler.step(val_metrics["loss"])
            gate = promotion_gate(
                val_metrics,
                safety_margin=args.safety_margin,
                min_quality_delta=args.min_quality_delta,
                min_tool_delta=args.min_tool_delta,
                max_calibration_degradation=args.max_calibration_degradation,
            )
            score = val_metrics["full_macro_f1"] - 2.0 * max(
                val_metrics["full_false_edit_rate"]
                - min(
                    val_metrics["baseline_argmax_false_edit_rate"],
                    val_metrics["baseline_deployed_false_edit_rate"],
                )
                - args.safety_margin,
                0.0,
            )
            row = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "selection_score": score,
                "promotion_gate": gate,
                "train": train_metrics,
                "val": val_metrics,
            }
            history.write(json.dumps(row) + "\n")
            history.flush()
            print(
                f"epoch={epoch}/{args.epochs} train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"argmax_f1={val_metrics['baseline_argmax_macro_f1']:.6f} "
                f"deployed_f1={val_metrics['baseline_deployed_macro_f1']:.6f} "
                f"zero_tool_f1={val_metrics['zero_tool_macro_f1']:.6f} "
                f"full_f1={val_metrics['full_macro_f1']:.6f} "
                f"false_edit={val_metrics['full_false_edit_rate']:.6f} "
                f"promoted={gate['passed']}",
                flush=True,
            )
            checkpoint = _checkpoint(model, epoch, val_metrics, args)
            torch.save(checkpoint, args.output_dir / "last.pt")
            if score > best_score + 1e-8:
                best_score = score
                best_epoch = epoch
                best_metrics = dict(val_metrics)
                stale = 0
                torch.save(checkpoint, args.output_dir / "best.pt")
            else:
                stale += 1
            if gate["passed"] and (
                best_promoted_metrics is None
                or val_metrics["full_macro_f1"]
                > best_promoted_metrics["full_macro_f1"] + 1e-8
            ):
                best_promoted_epoch = epoch
                best_promoted_metrics = dict(val_metrics)
                torch.save(checkpoint, args.output_dir / "best_promoted.pt")
            if stale >= args.patience:
                break
    summary = {
        **config,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "best_val_metrics": best_metrics,
        "best_promoted_epoch": best_promoted_epoch,
        "best_promoted_val_metrics": best_promoted_metrics,
        "promotion_passed": best_promoted_epoch is not None,
        "epochs_completed": epoch,
        "early_stopped": epoch < args.epochs,
        "selection_criterion": "full macro-F1 with false-edit safety penalty",
        "promotion_criterion": "safe and beats both belief baseline and zero-tool ablation",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
