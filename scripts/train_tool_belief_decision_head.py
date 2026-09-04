#!/usr/bin/env python3
"""Train a hierarchical KEEP/UPDATE then operation decision head on belief trajectories."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scripts.analyze_tool_belief_stopping import EDIT_ORDER, _summary

KEEP_INDEX = EDIT_ORDER.index("KEEP")
UPDATE_OPERATIONS = tuple(operation for operation in EDIT_ORDER if operation != "KEEP")
FEATURE_DIM = 24


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"empty trajectory details: {path}")
    return rows


def _feature_names() -> list[str]:
    names = [f"baseline_p_{operation.lower()}" for operation in EDIT_ORDER]
    names.extend(f"baseline_anchor_{operation.lower()}" for operation in EDIT_ORDER)
    names.append("baseline_has_explicit_anchor")
    for step in range(1, 4):
        names.extend(f"step{step}_p_{operation.lower()}" for operation in EDIT_ORDER)
        names.append(f"step{step}_confidence")
    return names


def _group(
    rows: list[dict[str, Any]],
    expected_split: str,
    trajectory_prefix: str = "paired",
    stages: tuple[int, ...] = (3,),
) -> list[dict[str, Any]]:
    if not stages or any(stage not in {0, 1, 2, 3} for stage in stages):
        raise ValueError("stages must be a nonempty subset of 0,1,2,3")
    if tuple(sorted(set(stages))) != stages:
        raise ValueError("stages must be unique and sorted")
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        if row.get("split", expected_split) != expected_split:
            raise ValueError(f"trajectory row is not from {expected_split}")
        grouped.setdefault(str(row["sequence_id"]), []).append(row)
    examples = []
    for sequence_id, sequence_rows in grouped.items():
        sequence_rows.sort(key=lambda item: int(item["step"]))
        if [int(item["step"]) for item in sequence_rows] != [1, 2, 3]:
            raise ValueError(f"{sequence_id} does not contain steps 1, 2, 3")
        first = sequence_rows[0]
        target = str(first["target"])
        baseline = str(first["baseline"])
        recommendation = first.get("baseline_recommended_edit")
        anchored_operation = str(recommendation or baseline)
        baseline_features = [float(value) for value in first["baseline_probabilities"]]
        baseline_features.extend(
            float(anchored_operation == operation) for operation in EDIT_ORDER
        )
        baseline_features.append(float(recommendation is not None))
        for stage in stages:
            features = list(baseline_features)
            for row in sequence_rows:
                if int(row["step"]) <= stage:
                    features.extend(
                        float(value)
                        for value in row[f"{trajectory_prefix}_probabilities"]
                    )
                    features.append(float(row[f"{trajectory_prefix}_confidence"]))
                else:
                    features.extend([0.0] * 5)
            if len(features) != FEATURE_DIM:
                raise ValueError(
                    f"{sequence_id} has {len(features)} features, expected {FEATURE_DIM}"
                )
            residual = (
                baseline
                if stage == 0
                else str(sequence_rows[stage - 1][trajectory_prefix])
            )
            examples.append(
                {
                    "sequence_id": sequence_id,
                    "stage": stage,
                    "features": features,
                    "target": EDIT_ORDER.index(target),
                    "baseline": EDIT_ORDER.index(baseline),
                    "residual": EDIT_ORDER.index(residual),
                }
            )
    return examples


class DecisionTrajectoryDataset(Dataset):
    def __init__(
        self,
        path: Path,
        expected_split: str,
        trajectory_prefix: str = "paired",
        stages: tuple[int, ...] = (3,),
    ) -> None:
        self.examples = _group(
            _read_jsonl(path), expected_split, trajectory_prefix, stages
        )
        self.features = torch.tensor(
            [example["features"] for example in self.examples], dtype=torch.float32
        )
        self.target = torch.tensor(
            [example["target"] for example in self.examples], dtype=torch.long
        )
        self.baseline = torch.tensor(
            [example["baseline"] for example in self.examples], dtype=torch.long
        )
        self.residual = torch.tensor(
            [example["residual"] for example in self.examples], dtype=torch.long
        )
        self.class_counts = Counter(EDIT_ORDER[index] for index in self.target.tolist())
        self.stages = torch.tensor(
            [example["stage"] for example in self.examples], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.features[index],
            self.target[index],
            self.baseline[index],
            self.residual[index],
            self.stages[index],
        )


class HierarchicalDecisionHead(nn.Module):
    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.update_head = nn.Linear(hidden_dim, 1)
        self.operation_head = nn.Linear(hidden_dim, len(UPDATE_OPERATIONS))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        return self.update_head(hidden).squeeze(-1), self.operation_head(hidden)


def _thresholds(values: np.ndarray) -> list[float]:
    unique = np.unique(values)
    if unique.size == 1:
        return [float(unique[0] - 1e-6), float(unique[0] + 1e-6)]
    return [
        float(unique[0] - 1e-6),
        *[float(value) for value in (unique[:-1] + unique[1:]) / 2.0],
        float(unique[-1] + 1e-6),
    ]


def _calibrate(
    target: np.ndarray,
    baseline: np.ndarray,
    update_probability: np.ndarray,
    update_operation: np.ndarray,
    *,
    safety_margin: float,
) -> dict[str, Any] | None:
    baseline_metrics = _summary(
        "identity", target.tolist(), baseline.tolist(), [0.5] * len(target), [0.0] * len(target)
    )
    candidates = []
    for threshold in _thresholds(update_probability):
        prediction = np.where(
            update_probability >= threshold, update_operation, KEEP_INDEX
        )
        confidence = np.where(
            prediction == KEEP_INDEX, 1.0 - update_probability, update_probability
        )
        metrics = _summary(
            "hierarchical",
            target.tolist(),
            prediction.tolist(),
            confidence.tolist(),
            [0.0] * len(target),
        )
        feasible = (
            metrics["false_edit_rate"]
            <= baseline_metrics["false_edit_rate"] + safety_margin + 1e-12
            and metrics["missed_edit_rate"]
            <= baseline_metrics["missed_edit_rate"] + safety_margin + 1e-12
        )
        if feasible:
            candidates.append(
                {
                    "threshold": threshold,
                    "prediction": prediction,
                    "metrics": metrics,
                    "baseline_metrics": baseline_metrics,
                }
            )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item["metrics"]["macro_f1"],
            item["metrics"]["accuracy"],
            -item["metrics"]["false_edit_rate"],
        ),
    )


def _operation_weights(dataset: DecisionTrajectoryDataset, device: torch.device) -> torch.Tensor:
    counts = []
    for operation in UPDATE_OPERATIONS:
        counts.append(max(dataset.class_counts[operation], 1))
    values = torch.tensor(counts, dtype=torch.float32)
    weights = (values.sum() / values).sqrt()
    return (weights / weights.mean()).to(device)


def _run_epoch(
    loader: DataLoader,
    model: HierarchicalDecisionHead,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    operation_weights: torch.Tensor,
    keep_weight: float,
    update_weight: float,
    operation_loss_weight: float,
    safety_margin: float,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    targets = []
    baselines = []
    residuals = []
    update_probabilities = []
    update_operations = []
    stages = []
    for features, target, baseline, residual, stage in tqdm(
        loader, leave=False, unit="batch", desc="train" if training else "val"
    ):
        features = features.to(device)
        target_device = target.to(device)
        update_target = (target_device != KEEP_INDEX).float()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            update_logit, operation_logit = model(features)
            binary_loss = F.binary_cross_entropy_with_logits(
                update_logit, update_target, reduction="none"
            )
            sample_weights = torch.where(
                update_target > 0.5,
                torch.full_like(update_target, update_weight),
                torch.full_like(update_target, keep_weight),
            )
            binary_loss = torch.mean(binary_loss * sample_weights)
            update_mask = target_device != KEEP_INDEX
            if bool(update_mask.any()):
                operation_target = target_device[update_mask] - 1
                operation_loss = F.cross_entropy(
                    operation_logit[update_mask],
                    operation_target,
                    weight=operation_weights,
                )
            else:
                operation_loss = update_logit.sum() * 0.0
            loss = binary_loss + operation_loss_weight * operation_loss
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        count = int(features.shape[0])
        total_loss += float(loss.detach()) * count
        sample_count += count
        targets.append(target.numpy())
        baselines.append(baseline.numpy())
        residuals.append(residual.numpy())
        update_probabilities.append(torch.sigmoid(update_logit).detach().cpu().numpy())
        update_operations.append(
            (torch.argmax(operation_logit, dim=1) + 1).detach().cpu().numpy()
        )
        stages.append(stage.numpy())
    target = np.concatenate(targets)
    baseline = np.concatenate(baselines)
    residual = np.concatenate(residuals)
    update_probability = np.concatenate(update_probabilities)
    update_operation = np.concatenate(update_operations)
    stage = np.concatenate(stages)
    unique_stages = sorted(int(value) for value in np.unique(stage))
    calibrated_by_stage = {
        str(stage_value): _calibrate(
            target[stage == stage_value],
            baseline[stage == stage_value],
            update_probability[stage == stage_value],
            update_operation[stage == stage_value],
            safety_margin=safety_margin,
        )
        for stage_value in unique_stages
    }
    calibrated = (
        calibrated_by_stage[str(unique_stages[0])]
        if len(unique_stages) == 1
        else None
    )
    residual_metrics = _summary(
        "residual_argmax",
        target.tolist(),
        residual.tolist(),
        [0.5] * len(target),
        [0.0] * len(target),
    )
    return {
        "loss": total_loss / sample_count,
        "calibrated": calibrated,
        "calibrated_by_stage": calibrated_by_stage,
        "residual_metrics": residual_metrics,
        "target": target,
        "update_probability": update_probability,
        "update_operation": update_operation,
    }


def _json_metrics(result: dict[str, Any]) -> dict[str, Any]:
    calibrated = result["calibrated"]
    return {
        "loss": result["loss"],
        "calibrated": (
            {
                "threshold": calibrated["threshold"],
                "metrics": calibrated["metrics"],
                "baseline_metrics": calibrated["baseline_metrics"],
            }
            if calibrated is not None
            else None
        ),
        "residual_metrics": result["residual_metrics"],
        "calibrated_by_stage": {
            stage: (
                {
                    "threshold": value["threshold"],
                    "metrics": value["metrics"],
                    "baseline_metrics": value["baseline_metrics"],
                }
                if value is not None
                else None
            )
            for stage, value in result["calibrated_by_stage"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_details", type=Path)
    parser.add_argument("val_details", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--keep-weight", type=float, default=2.0)
    parser.add_argument("--update-weight", type=float, default=1.0)
    parser.add_argument("--operation-loss-weight", type=float, default=1.0)
    parser.add_argument("--safety-margin", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--prefix-stages",
        default="3",
        help="comma-separated causal stages from 0,1,2,3; default preserves full-trajectory v1",
    )
    args = parser.parse_args()
    if any(
        value <= 0
        for value in (args.epochs, args.batch_size, args.hidden_dim, args.patience)
    ):
        raise ValueError("epochs, batch-size, hidden-dim, and patience must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    try:
        stages = tuple(int(value) for value in args.prefix_stages.split(","))
    except ValueError as exc:
        raise ValueError("prefix-stages must be comma-separated integers") from exc
    if not stages or any(stage not in {0, 1, 2, 3} for stage in stages):
        raise ValueError("prefix-stages must be a nonempty subset of 0,1,2,3")
    if tuple(sorted(set(stages))) != stages:
        raise ValueError("prefix-stages must be unique and sorted")
    train_data = DecisionTrajectoryDataset(args.train_details, "train", stages=stages)
    val_data = DecisionTrajectoryDataset(args.val_details, "val", stages=stages)
    protected = ("best_safety.pt", "last.pt", "history.jsonl", "summary.json")
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
    model = HierarchicalDecisionHead(args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(args.patience // 4, 1)
    )
    operation_weights = _operation_weights(train_data, device)
    best_f1 = -math.inf
    best_epoch: int | None = None
    best_val: dict[str, Any] | None = None
    stale = 0
    protocol = (
        "hierarchical_decision_head_v1"
        if stages == (3,)
        else "hierarchical_decision_head_prefix_v2"
    )
    config = {
        "protocol": protocol,
        "feature_dim": FEATURE_DIM,
        "feature_names": _feature_names(),
        "train_sequences": len(train_data),
        "val_sequences": len(val_data),
        "train_unique_trajectories": len(
            {example["sequence_id"] for example in train_data.examples}
        ),
        "val_unique_trajectories": len(
            {example["sequence_id"] for example in val_data.examples}
        ),
        "prefix_stages": list(stages),
        "train_class_counts": dict(train_data.class_counts),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "keep_weight": args.keep_weight,
        "update_weight": args.update_weight,
        "operation_loss_weight": args.operation_loss_weight,
        "safety_margin": args.safety_margin,
        "seed": args.seed,
        "test_assets_read": False,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "history.jsonl").open("w", encoding="utf-8") as history:
        for epoch in range(1, args.epochs + 1):
            train_result = _run_epoch(
                train_loader,
                model,
                device=device,
                optimizer=optimizer,
                operation_weights=operation_weights,
                keep_weight=args.keep_weight,
                update_weight=args.update_weight,
                operation_loss_weight=args.operation_loss_weight,
                safety_margin=args.safety_margin,
            )
            with torch.inference_mode():
                val_result = _run_epoch(
                    val_loader,
                    model,
                    device=device,
                    optimizer=None,
                    operation_weights=operation_weights,
                    keep_weight=args.keep_weight,
                    update_weight=args.update_weight,
                    operation_loss_weight=args.operation_loss_weight,
                    safety_margin=args.safety_margin,
                )
            scheduler.step(val_result["loss"])
            train_json = _json_metrics(train_result)
            val_json = _json_metrics(val_result)
            row = {"epoch": epoch, "train": train_json, "val": val_json}
            history.write(json.dumps(row) + "\n")
            history.flush()
            calibrated = val_result["calibrated"]
            stage_calibrations = val_result["calibrated_by_stage"]
            all_stages_safe = all(
                value is not None for value in stage_calibrations.values()
            )
            val_f1 = (
                float(
                    np.mean(
                        [
                            value["metrics"]["macro_f1"]
                            for value in stage_calibrations.values()
                            if value is not None
                        ]
                    )
                )
                if all_stages_safe
                else -math.inf
            )
            print(
                f"epoch={epoch}/{args.epochs} train_loss={train_result['loss']:.6f} "
                f"val_loss={val_result['loss']:.6f} val_safe_macro_f1={val_f1:.6f}",
                flush=True,
            )
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "feature_dim": FEATURE_DIM,
                "feature_names": _feature_names(),
                "decision_threshold": (
                    calibrated["threshold"] if calibrated is not None else None
                ),
                "decision_thresholds": {
                    stage: value["threshold"]
                    for stage, value in stage_calibrations.items()
                    if value is not None
                },
                "val_metrics": val_json,
                "prefix_stages": list(stages),
                "protocol": protocol,
            }
            torch.save(checkpoint, args.output_dir / "last.pt")
            if val_f1 > best_f1 + 1e-8:
                best_f1 = val_f1
                best_epoch = epoch
                best_val = val_json
                stale = 0
                torch.save(checkpoint, args.output_dir / "best_safety.pt")
            else:
                stale += 1
            if stale >= args.patience:
                break
    summary = {
        **config,
        "best_epoch": best_epoch,
        "best_safe_macro_f1": best_f1 if best_epoch is not None else None,
        "best_val": best_val,
        "epochs_completed": epoch,
        "early_stopped": epoch < args.epochs,
        "passed_hard_safety": best_epoch is not None,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
