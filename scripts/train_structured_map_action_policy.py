#!/usr/bin/env python3
"""Train a joint evidence-acquisition and terminal-edit action policy."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from activemap.agent.evidence_value_head import (
    CONTEXT_DIM,
    EvidenceValueHead,
    EvidenceValueHeadConfig,
    fit_evidence_value_normalizer,
    risk_adjusted_scores,
)
from activemap.models import EditOperation
from scripts.train_evidence_value_head import (
    calibrate_margin,
    collate_examples,
    evidence_value_loss,
    load_examples,
)


def terminal_reward(prediction: int, target: int) -> float:
    keep = list(EditOperation).index(EditOperation.KEEP)
    if prediction == target:
        return 1.0
    if prediction == keep:
        return -0.75
    if target == keep:
        return -1.0
    return -0.5


def terminal_metrics(
    examples: list[dict[str, Any]], predictions: np.ndarray
) -> dict[str, float]:
    targets = np.asarray([row["terminal_target"] for row in examples], dtype=np.int64)
    baseline = np.asarray(
        [list(EditOperation).index(EditOperation(row["edit_type"])) for row in examples],
        dtype=np.int64,
    )
    keep = list(EditOperation).index(EditOperation.KEEP)
    f1 = []
    target_support = []
    for label in range(len(EditOperation)):
        target_support.append(int((targets == label).sum()))
        true_positive = int(((predictions == label) & (targets == label)).sum())
        false_positive = int(((predictions == label) & (targets != label)).sum())
        false_negative = int(((predictions != label) & (targets == label)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        f1.append(2 * true_positive / denominator if denominator else 0.0)
    rewards = np.asarray(
        [
            terminal_reward(int(prediction), int(target))
            for prediction, target in zip(predictions, targets, strict=True)
        ]
    )
    baseline_rewards = np.asarray(
        [
            terminal_reward(int(prediction), int(target))
            for prediction, target in zip(baseline, targets, strict=True)
        ]
    )
    return {
        "accuracy": float((predictions == targets).mean()),
        "macro_f1": float(np.mean(f1)),
        "supported_macro_f1": float(
            np.mean(
                [
                    value
                    for value, support in zip(f1, target_support, strict=True)
                    if support > 0
                ]
            )
        ),
        "target_support": {
            edit.value: target_support[index]
            for index, edit in enumerate(EditOperation)
        },
        "false_edit_rate": float(((targets == keep) & (predictions != keep)).mean()),
        "missed_edit_rate": float(((targets != keep) & (predictions == keep)).mean()),
        "wrong_edit_rate": float(
            ((targets != keep) & (predictions != keep) & (predictions != targets)).mean()
        ),
        "mean_reward": float(rewards.mean()),
        "baseline_accuracy": float((baseline == targets).mean()),
        "baseline_mean_reward": float(baseline_rewards.mean()),
        "reward_gain_over_fixed_terminal": float(
            rewards.mean() - baseline_rewards.mean()
        ),
    }


def load_terminal_replay(
    paths: list[Path],
    split: str,
) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw.get("split") != split:
                    continue
                metadata = raw.get("metadata")
                if not isinstance(metadata, dict):
                    raise ValueError(f"terminal replay lacks metadata: {path}:{line_number}")
                if metadata.get("test_assets_read") or split == "test":
                    raise ValueError("terminal replay cannot contain test data")
                context = np.asarray(
                    raw["hypothesis_features"] + raw["state_features"],
                    dtype=np.float32,
                )
                if context.shape != (CONTEXT_DIM,) or not np.isfinite(context).all():
                    raise ValueError(
                        f"invalid terminal replay context: {path}:{line_number}"
                    )
                target = EditOperation(str(metadata["gt_edit"]))
                rows.append(
                    {
                        "context": context,
                        "terminal_target": list(EditOperation).index(target),
                        "edit_type": str(raw["edit_type"]),
                    }
                )
    return rows


def collate_terminal_replay(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "context": torch.as_tensor(
            np.stack([row["context"] for row in rows]),
            dtype=torch.float32,
        ),
        "terminal_target": torch.as_tensor(
            [row["terminal_target"] for row in rows],
            dtype=torch.long,
        ),
    }


@torch.no_grad()
def infer_terminal_replay(
    model: EvidenceValueHead,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    predictions = []
    for start in range(0, len(rows), batch_size):
        context = torch.as_tensor(
            np.stack([row["context"] for row in rows[start : start + batch_size]]),
            dtype=torch.float32,
            device=device,
        )
        predictions.extend(model.terminal_logits(context).argmax(dim=-1).cpu().tolist())
    return np.asarray(predictions, dtype=np.int64)


@torch.no_grad()
def infer(
    model: EvidenceValueHead,
    examples: list[dict[str, Any]],
    *,
    normalizer: Any,
    device: torch.device,
    unsafe_penalty: float,
    missed_penalty: float,
    batch_size: int = 512,
) -> tuple[list[np.ndarray], np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    terminal_predictions = []
    for start in range(0, len(examples), batch_size):
        rows = []
        for row in examples[start : start + batch_size]:
            context, candidates = normalizer.transform(
                row["context"], row["candidates"]
            )
            rows.append({**row, "context": context, "candidates": candidates})
        batch = {
            key: value.to(device) for key, value in collate_examples(rows).items()
        }
        outputs = model(batch["context"], batch["candidates"])
        values = risk_adjusted_scores(
            outputs,
            unsafe_weight=unsafe_penalty,
            missed_weight=missed_penalty,
        )
        terminal_predictions.extend(
            outputs["terminal_edit_logits"].argmax(dim=-1).cpu().tolist()
        )
        for row_index, row in enumerate(rows):
            scores.append(
                values[row_index, : len(row["utility_gains"])].cpu().numpy()
            )
    return scores, np.asarray(terminal_predictions, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("states_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--terminal-loss-weight", type=float, default=1.0)
    parser.add_argument("--terminal-score-weight", type=float, default=0.1)
    parser.add_argument(
        "--terminal-replay",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--terminal-replay-loss-weight", type=float, default=1.0)
    parser.add_argument("--terminal-replay-score-weight", type=float, default=0.1)
    parser.add_argument("--unsafe-penalty", type=float, default=0.10)
    parser.add_argument("--missed-penalty", type=float, default=0.05)
    parser.add_argument("--maximum-false-call-rate", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-train-states", type=int)
    parser.add_argument("--max-val-states", type=int)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train = load_examples(args.states_jsonl, "train")
    validation = load_examples(args.states_jsonl, "val")
    train_terminal_replay = load_terminal_replay(args.terminal_replay, "train")
    val_terminal_replay = load_terminal_replay(args.terminal_replay, "val")
    if bool(train_terminal_replay) != bool(val_terminal_replay):
        raise ValueError("terminal replay requires both train and validation rows")
    if args.max_train_states is not None:
        train = train[: args.max_train_states]
    if args.max_val_states is not None:
        validation = validation[: args.max_val_states]
    normalizer = fit_evidence_value_normalizer(train)
    for row in train:
        row["context"], row["candidates"] = normalizer.transform(
            row["context"], row["candidates"]
        )
    for row in train_terminal_replay:
        row["context"] = (
            row["context"] - normalizer.context_mean
        ) / normalizer.context_std
    for row in val_terminal_replay:
        row["context"] = (
            row["context"] - normalizer.context_mean
        ) / normalizer.context_std

    positive_states = np.asarray(
        [float(row["utility_gains"].max()) > 0 for row in train]
    )
    if positive_states.all() or not positive_states.any():
        raise ValueError("training requires both ACQUIRE and STOP states")
    positive_weight = (~positive_states).sum() / positive_states.sum()
    sampler = WeightedRandomSampler(
        torch.as_tensor(
            np.where(positive_states, positive_weight, 1.0), dtype=torch.double
        ),
        num_samples=len(train),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        train,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_examples,
    )
    replay_loader = (
        DataLoader(
            train_terminal_replay,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_terminal_replay,
            generator=torch.Generator().manual_seed(args.seed),
        )
        if train_terminal_replay
        else None
    )
    target_counts = np.bincount(
        [row["terminal_target"] for row in train]
        + [row["terminal_target"] for row in train_terminal_replay],
        minlength=len(EditOperation),
    )
    terminal_weights = np.sqrt(target_counts.sum() / np.maximum(target_counts, 1))
    terminal_weights /= terminal_weights.mean()
    terminal_weights_tensor = torch.as_tensor(
        terminal_weights, dtype=torch.float32, device=device
    )

    config = EvidenceValueHeadConfig(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        terminal_classes=len(EditOperation),
    )
    model = EvidenceValueHead(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )
    args.output_dir.mkdir(parents=True)
    best_score = -float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["context"], batch["candidates"])
            acquisition_loss, _ = evidence_value_loss(outputs, batch)
            terminal_loss = F.cross_entropy(
                outputs["terminal_edit_logits"],
                batch["terminal_target"],
                weight=terminal_weights_tensor,
            )
            loss = acquisition_loss + args.terminal_loss_weight * terminal_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals.append(float(loss.detach()))
        replay_totals = []
        if replay_loader is not None:
            for batch in replay_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                replay_loss = F.cross_entropy(
                    model.terminal_logits(batch["context"]),
                    batch["terminal_target"],
                    weight=terminal_weights_tensor,
                )
                loss = args.terminal_replay_loss_weight * replay_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                replay_totals.append(float(replay_loss.detach()))

        validation_scores, terminal_predictions = infer(
            model,
            validation,
            normalizer=normalizer,
            device=device,
            unsafe_penalty=args.unsafe_penalty,
            missed_penalty=args.missed_penalty,
        )
        margin, acquisition_metrics = calibrate_margin(
            validation,
            validation_scores,
            args.maximum_false_call_rate,
        )
        edit_metrics = terminal_metrics(validation, terminal_predictions)
        replay_metrics = None
        if val_terminal_replay:
            replay_predictions = infer_terminal_replay(
                model,
                val_terminal_replay,
                device=device,
            )
            replay_metrics = terminal_metrics(
                val_terminal_replay,
                replay_predictions,
            )
        score = (
            acquisition_metrics["utility_gain_over_stop_mean"]
            + args.terminal_score_weight
            * edit_metrics["reward_gain_over_fixed_terminal"]
            + (
                args.terminal_replay_score_weight
                * replay_metrics["supported_macro_f1"]
                if replay_metrics is not None
                else 0.0
            )
        )
        scheduler.step(-score)
        record = {
            "epoch": epoch,
            "loss": float(np.mean(totals)),
            "terminal_replay_loss": (
                float(np.mean(replay_totals)) if replay_totals else None
            ),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "margin": margin,
            "joint_selection_score": score,
            "acquisition": acquisition_metrics,
            "terminal": edit_metrics,
            "terminal_replay": replay_metrics,
        }
        with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(json.dumps(record, separators=(",", ":")), flush=True)
        if score > best_score + 1e-8:
            best_score = score
            stale = 0
            torch.save(
                {
                    "protocol": "structured_map_action_policy_v1",
                    "state_dict": model.state_dict(),
                    "model_config": config.as_dict(),
                    "normalizer": normalizer.as_dict(),
                    "safety_margin": margin,
                    "unsafe_penalty": args.unsafe_penalty,
                    "missed_penalty": args.missed_penalty,
                    "terminal_loss_weight": args.terminal_loss_weight,
                    "terminal_score_weight": args.terminal_score_weight,
                    "terminal_replay_loss_weight": args.terminal_replay_loss_weight,
                    "terminal_replay_score_weight": args.terminal_replay_score_weight,
                    "terminal_replay_train_rows": len(train_terminal_replay),
                    "terminal_replay_val_rows": len(val_terminal_replay),
                    "terminal_class_weights": terminal_weights.tolist(),
                    "epoch": epoch,
                    "seed": args.seed,
                    "val_acquisition_metrics": acquisition_metrics,
                    "val_terminal_metrics": edit_metrics,
                    "val_terminal_replay_metrics": replay_metrics,
                    "joint_selection_score": score,
                    "test_assets_read": False,
                },
                args.output_dir / "best.pt",
            )
        else:
            stale += 1
        if stale >= args.patience:
            break

    checkpoint = torch.load(
        args.output_dir / "best.pt", map_location="cpu", weights_only=False
    )
    summary = {
        "schema_version": "structured-map-action-training-v1",
        "best_epoch": checkpoint["epoch"],
        "train_states": len(train),
        "val_states": len(validation),
        "joint_selection_score": checkpoint["joint_selection_score"],
        "val_acquisition_metrics": checkpoint["val_acquisition_metrics"],
        "val_terminal_metrics": checkpoint["val_terminal_metrics"],
        "terminal_replay_train_rows": checkpoint["terminal_replay_train_rows"],
        "terminal_replay_val_rows": checkpoint["terminal_replay_val_rows"],
        "val_terminal_replay_metrics": checkpoint["val_terminal_replay_metrics"],
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
