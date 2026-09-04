#!/usr/bin/env python3
"""Train a causal STOP/CONTINUE policy over a frozen stage-aware belief head."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from activemap.features import EVIDENCE_DIM
from activemap.training.data import load_selector_samples
from scripts.analyze_tool_belief_stage_stopping import _costs, _predict
from scripts.analyze_tool_belief_stopping import _reward, _summary
from scripts.train_tool_belief_decision_head import DecisionTrajectoryDataset

POLICY_FEATURE_DIM = 34 + EVIDENCE_DIM


@dataclass(frozen=True)
class PolicyData:
    sequence_ids: list[str]
    features: np.ndarray
    targets: np.ndarray
    predictions: np.ndarray
    confidences: np.ndarray
    costs: np.ndarray
    continue_targets: np.ndarray
    future_advantages: np.ndarray


class StoppingPolicy(nn.Module):
    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(POLICY_FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.continue_head = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        return (
            self.continue_head(hidden).squeeze(-1),
            self.advantage_head(hidden).squeeze(-1),
        )


def build_policy_data(
    stage_checkpoint: Path,
    details: Path,
    selector_states: Path,
    *,
    split: str,
    device: str = "cpu",
) -> PolicyData:
    from activemap.agent.identifiers import public_evidence_id, public_task_id
    from scripts.build_muno21_tool_belief_data import _initial_samples

    dataset = DecisionTrajectoryDataset(details, split, stages=(0, 1, 2, 3))
    predictions, confidences = _predict(stage_checkpoint, dataset, device=device)
    costs_by_sequence = _costs(details, split)
    sequence_ids = list(costs_by_sequence)
    if [example["sequence_id"] for example in dataset.examples[::4]] != sequence_ids:
        raise ValueError("trajectory and cost sequence ordering differs")
    sequence_count = len(sequence_ids)
    base_features = dataset.features.numpy().reshape(sequence_count, 4, -1)
    targets = dataset.target.numpy().reshape(sequence_count, 4)[:, 0]
    stage_predictions = predictions.reshape(sequence_count, 4)
    stage_confidences = confidences.reshape(sequence_count, 4)
    costs = np.asarray(
        [costs_by_sequence[sequence_id] for sequence_id in sequence_ids],
        dtype=np.float32,
    )
    samples = _initial_samples(load_selector_samples(selector_states, split=split))
    samples_by_public_id = {
        public_task_id(raw_id): sample for raw_id, sample in samples.items()
    }
    if len(samples_by_public_id) != len(samples):
        raise ValueError("public selector episode identifier collision")
    detail_rows = _read_detail_groups(details, split)
    candidate_features = []
    for sequence_id in sequence_ids:
        rows = detail_rows[sequence_id]
        public_episode = str(rows[0]["episode_id"])
        sample = samples_by_public_id.get(public_episode)
        if sample is None:
            raise ValueError(f"{sequence_id} has no matching initial selector state")
        by_public_evidence = {
            public_evidence_id(raw_id): np.asarray(features, dtype=np.float32)
            for raw_id, features in zip(
                sample.evidence_ids, sample.evidence_features, strict=True
            )
        }
        try:
            candidate_features.append(
                [by_public_evidence[str(row["evidence_id"])] for row in rows]
            )
        except KeyError as exc:
            raise ValueError(f"{sequence_id} evidence is absent from selector metadata") from exc
    candidate_feature_array = np.asarray(candidate_features, dtype=np.float32)
    if candidate_feature_array.shape != (sequence_count, 3, EVIDENCE_DIM):
        raise ValueError("candidate evidence metadata has an unexpected shape")
    features = []
    continue_targets = []
    future_advantages = []
    for sequence_index in range(sequence_count):
        utilities = np.asarray(
            [
                _reward(
                    int(targets[sequence_index]),
                    int(stage_predictions[sequence_index, stage]),
                )
                - float(costs[sequence_index, stage])
                for stage in range(4)
            ],
            dtype=np.float32,
        )
        for stage in range(3):
            stage_one_hot = np.eye(4, dtype=np.float32)[stage]
            prediction_one_hot = np.eye(4, dtype=np.float32)[
                stage_predictions[sequence_index, stage]
            ]
            features.append(
                np.concatenate(
                    [
                        base_features[sequence_index, stage],
                        candidate_feature_array[sequence_index, stage],
                        stage_one_hot,
                        prediction_one_hot,
                        np.asarray(
                            [
                                stage_confidences[sequence_index, stage],
                                costs[sequence_index, stage],
                            ],
                            dtype=np.float32,
                        ),
                    ]
                )
            )
            advantage = float(np.max(utilities[stage + 1 :]) - utilities[stage])
            future_advantages.append(advantage)
            continue_targets.append(float(advantage > 1e-8))
    feature_array = np.asarray(features, dtype=np.float32).reshape(sequence_count, 3, -1)
    if feature_array.shape[-1] != POLICY_FEATURE_DIM:
        raise ValueError(
            f"policy feature dimension is {feature_array.shape[-1]}, expected {POLICY_FEATURE_DIM}"
        )
    return PolicyData(
        sequence_ids=sequence_ids,
        features=feature_array,
        targets=targets,
        predictions=stage_predictions,
        confidences=stage_confidences,
        costs=costs,
        continue_targets=np.asarray(continue_targets, dtype=np.float32).reshape(
            sequence_count, 3
        ),
        future_advantages=np.asarray(future_advantages, dtype=np.float32).reshape(
            sequence_count, 3
        ),
    )


def _read_detail_groups(
    path: Path, split: str
) -> OrderedDict[str, list[dict[str, Any]]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split", split) != split:
                raise ValueError(f"trajectory row is not from {split}")
            grouped.setdefault(str(row["sequence_id"]), []).append(row)
    for sequence_id, rows in grouped.items():
        rows.sort(key=lambda item: int(item["step"]))
        if [int(item["step"]) for item in rows] != [1, 2, 3]:
            raise ValueError(f"{sequence_id} does not contain steps 1, 2, 3")
    return grouped


def rollout_policy(
    data: PolicyData,
    continue_probabilities: np.ndarray,
    threshold: float,
    *,
    name: str,
) -> tuple[dict[str, Any], np.ndarray]:
    if continue_probabilities.shape != (len(data.sequence_ids), 3):
        raise ValueError("continue probabilities must have shape [sequences, 3]")
    continue_mask = continue_probabilities >= threshold
    selected_stages = np.cumprod(continue_mask, axis=1).sum(axis=1).astype(np.int64)
    indices = np.arange(len(data.sequence_ids))
    metrics = _summary(
        name,
        data.targets.tolist(),
        data.predictions[indices, selected_stages].tolist(),
        data.confidences[indices, selected_stages].tolist(),
        data.costs[indices, selected_stages].tolist(),
    )
    metrics["mean_selected_stage"] = float(np.mean(selected_stages))
    metrics["selected_stage_counts"] = {
        str(stage): int(np.sum(selected_stages == stage)) for stage in range(4)
    }
    return metrics, selected_stages


def calibrate_threshold(
    data: PolicyData,
    continue_probabilities: np.ndarray,
    *,
    safety_margin: float,
) -> tuple[float, dict[str, Any]]:
    stage0, _ = rollout_policy(
        data, np.zeros_like(continue_probabilities), 1.1, name="stage_0_head"
    )
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99),
                np.quantile(continue_probabilities, np.linspace(0.0, 1.0, 101)),
                np.asarray([1.000001]),
            ]
        )
    )
    feasible = []
    for threshold in candidates:
        metrics, _ = rollout_policy(
            data,
            continue_probabilities,
            float(threshold),
            name="learned_stopping",
        )
        if (
            metrics["false_edit_rate"]
            <= stage0["false_edit_rate"] + safety_margin + 1e-12
            and metrics["missed_edit_rate"]
            <= stage0["missed_edit_rate"] + safety_margin + 1e-12
        ):
            feasible.append((float(threshold), metrics))
    if not feasible:
        raise RuntimeError("no train-safe stopping threshold exists")
    return max(
        feasible,
        key=lambda item: (
            item[1]["mean_joint_utility"],
            -item[1]["mean_tool_cost"],
            item[0],
        ),
    )


def _infer(
    model: StoppingPolicy, data: PolicyData, *, device: torch.device
) -> np.ndarray:
    features = torch.from_numpy(data.features.reshape(-1, POLICY_FEATURE_DIM)).to(device)
    with torch.inference_mode():
        logits, _ = model(features)
    return torch.sigmoid(logits).cpu().numpy().reshape(len(data.sequence_ids), 3)


def _safety_passes(
    learned: dict[str, Any], stage0: dict[str, Any], margin: float
) -> bool:
    return bool(
        learned["false_edit_rate"] <= stage0["false_edit_rate"] + margin + 1e-12
        and learned["missed_edit_rate"]
        <= stage0["missed_edit_rate"] + margin + 1e-12
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage_checkpoint", type=Path)
    parser.add_argument("train_details", type=Path)
    parser.add_argument("val_details", type=Path)
    parser.add_argument("selector_states", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--advantage-loss-weight", type=float, default=0.5)
    parser.add_argument("--safety-margin", type=float, default=0.02)
    parser.add_argument("--minimum-utility-gain", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train_data = build_policy_data(
        args.stage_checkpoint,
        args.train_details,
        args.selector_states,
        split="train",
        device="cpu",
    )
    val_data = build_policy_data(
        args.stage_checkpoint,
        args.val_details,
        args.selector_states,
        split="val",
        device="cpu",
    )
    protected = ("best_safe.pt", "best_utility.pt", "last.pt", "history.jsonl")
    existing = [name for name in protected if (args.output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {args.output_dir}: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    flat_features = torch.from_numpy(train_data.features.reshape(-1, POLICY_FEATURE_DIM))
    flat_targets = torch.from_numpy(train_data.continue_targets.reshape(-1))
    flat_advantages = torch.from_numpy(train_data.future_advantages.reshape(-1))
    loader = DataLoader(
        TensorDataset(flat_features, flat_targets, flat_advantages),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    positives = float(torch.sum(flat_targets))
    negatives = float(len(flat_targets) - positives)
    pos_weight = math.sqrt(negatives / max(positives, 1.0))
    model = StoppingPolicy(args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(args.patience // 4, 1)
    )
    config = {
        "protocol": "tool_belief_stopping_policy_v2",
        "feature_dim": POLICY_FEATURE_DIM,
        "train_sequences": len(train_data.sequence_ids),
        "val_sequences": len(val_data.sequence_ids),
        "train_state_count": int(flat_targets.numel()),
        "train_continue_count": int(positives),
        "positive_weight": pos_weight,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "advantage_loss_weight": args.advantage_loss_weight,
        "safety_margin": args.safety_margin,
        "minimum_utility_gain": args.minimum_utility_gain,
        "seed": args.seed,
        "stage_checkpoint": str(args.stage_checkpoint),
        "selector_states": str(args.selector_states),
        "threshold_calibration_split": "train",
        "test_assets_read": False,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    best_utility = -math.inf
    best_safe_utility = -math.inf
    best_epoch = None
    best_safe_epoch = None
    stale = 0
    with (args.output_dir / "history.jsonl").open("w", encoding="utf-8") as history:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            count = 0
            for features, target, advantage in tqdm(
                loader, leave=False, unit="batch", desc="stopping-policy train"
            ):
                features = features.to(device)
                target = target.to(device)
                advantage = advantage.to(device)
                optimizer.zero_grad(set_to_none=True)
                logit, predicted_advantage = model(features)
                binary_loss = F.binary_cross_entropy_with_logits(
                    logit,
                    target,
                    pos_weight=torch.tensor(pos_weight, device=device),
                )
                advantage_loss = F.smooth_l1_loss(predicted_advantage, advantage)
                loss = binary_loss + args.advantage_loss_weight * advantage_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                batch_count = int(features.shape[0])
                total_loss += float(loss.detach()) * batch_count
                count += batch_count
            model.eval()
            train_probability = _infer(model, train_data, device=device)
            threshold, train_metrics = calibrate_threshold(
                train_data, train_probability, safety_margin=args.safety_margin
            )
            val_probability = _infer(model, val_data, device=device)
            val_metrics, _ = rollout_policy(
                val_data, val_probability, threshold, name="learned_stopping"
            )
            stage0_metrics, _ = rollout_policy(
                val_data, np.zeros_like(val_probability), 1.1, name="stage_0_head"
            )
            utility_gain = (
                val_metrics["mean_joint_utility"]
                - stage0_metrics["mean_joint_utility"]
            )
            safe = _safety_passes(val_metrics, stage0_metrics, args.safety_margin)
            eligible = safe and utility_gain >= args.minimum_utility_gain
            scheduler.step(val_metrics["mean_joint_utility"])
            row = {
                "epoch": epoch,
                "train_loss": total_loss / count,
                "train_threshold": threshold,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "val_stage0": stage0_metrics,
                "val_utility_gain": utility_gain,
                "val_safe": safe,
                "val_eligible": eligible,
            }
            history.write(json.dumps(row) + "\n")
            history.flush()
            print(
                f"epoch={epoch}/{args.epochs} loss={row['train_loss']:.6f} "
                f"threshold={threshold:.6f} val_utility={val_metrics['mean_joint_utility']:.6f} "
                f"gain={utility_gain:.6f} calls={val_metrics['mean_selected_stage']:.4f} "
                f"eligible={eligible}",
                flush=True,
            )
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "feature_dim": POLICY_FEATURE_DIM,
                "decision_threshold": threshold,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "val_stage0": stage0_metrics,
                "val_utility_gain": utility_gain,
                "protocol": "tool_belief_stopping_policy_v2",
            }
            torch.save(checkpoint, args.output_dir / "last.pt")
            if val_metrics["mean_joint_utility"] > best_utility + 1e-10:
                best_utility = val_metrics["mean_joint_utility"]
                best_epoch = epoch
                stale = 0
                torch.save(checkpoint, args.output_dir / "best_utility.pt")
            else:
                stale += 1
            if eligible and val_metrics["mean_joint_utility"] > best_safe_utility + 1e-10:
                best_safe_utility = val_metrics["mean_joint_utility"]
                best_safe_epoch = epoch
                torch.save(checkpoint, args.output_dir / "best_safe.pt")
            if stale >= args.patience:
                break
    summary = {
        **config,
        "epochs_completed": epoch,
        "early_stopped": epoch < args.epochs,
        "best_utility_epoch": best_epoch,
        "best_val_utility": best_utility,
        "best_safe_epoch": best_safe_epoch,
        "best_safe_val_utility": (
            best_safe_utility if best_safe_epoch is not None else None
        ),
        "passed_promotion_gate": best_safe_epoch is not None,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
