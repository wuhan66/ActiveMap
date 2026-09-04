#!/usr/bin/env python3
"""Train a hierarchical terminal action adapter on grounded Tool-Belief pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from activemap.agent.post_tool_action_adapter import (
    PostToolActionAdapter,
    PostToolActionAdapterConfig,
    SEMANTIC_CONDITIONED_FEATURE_DIM,
    SEMANTIC_ONLY_FEATURE_DIM,
    TOOL_CONDITIONED_FEATURE_DIM,
    UPDATE_OPERATIONS,
    encode_post_tool_action_features,
    encode_semantic_action_features,
)
from activemap.agent.tool_belief_data import PostAcquisitionToolPairExample
from activemap.agent.tool_belief_model import PairedToolBeliefUpdater, encode_belief
from activemap.agent.tool_features import encode_semantic_tool_result
from activemap.models import EditOperation


def load_rows(
    path: Path,
    updater: PairedToolBeliefUpdater,
    expected_split: str,
    include_tool_results: bool,
    include_semantic_result: bool,
    use_post_acquisition_belief: bool,
    semantic_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    targets = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = PostAcquisitionToolPairExample.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid {path}:{line_number}") from exc
            if row.split != expected_split or row.metadata.get("test_assets_read"):
                raise ValueError(f"invalid {expected_split} row at {path}:{line_number}")
            updated = (
                row.post_acquisition_belief
                if include_semantic_result
                or use_post_acquisition_belief
                or semantic_only
                else updater.update_pair(
                    row.post_acquisition_belief,
                    row.quality_result,
                    row.temporal_result,
                )
            )
            if (include_semantic_result or semantic_only) and row.semantic_result is None:
                raise ValueError(f"missing semantic result at {path}:{line_number}")
            features.append(
                encode_semantic_tool_result(row.semantic_result)
                if semantic_only and row.semantic_result is not None
                else encode_semantic_action_features(updated, row.semantic_result)
                if include_semantic_result and row.semantic_result is not None
                else encode_post_tool_action_features(
                    updated,
                    [row.quality_result, row.temporal_result],
                )
                if include_tool_results
                else encode_belief(updated)
            )
            targets.append(list(EditOperation).index(row.gt_edit))
    if not features:
        raise ValueError(f"empty {expected_split} adapter dataset")
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.int64)


def predictions(
    model: PostToolActionAdapter,
    features: np.ndarray,
    threshold: float,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        update_logit, operation_logit = model(
            torch.from_numpy(features).to(device)
        )
    update = torch.sigmoid(update_logit).cpu().numpy() >= threshold
    operation = operation_logit.argmax(dim=-1).cpu().numpy() + 1
    return np.where(update, operation, 0).astype(np.int64)


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    f1 = []
    support = []
    for label in range(len(EditOperation)):
        support.append(int((target == label).sum()))
        tp = int(((prediction == label) & (target == label)).sum())
        fp = int(((prediction == label) & (target != label)).sum())
        fn = int(((prediction != label) & (target == label)).sum())
        denominator = 2 * tp + fp + fn
        f1.append(2 * tp / denominator if denominator else 0.0)
    return {
        "accuracy": float((prediction == target).mean()),
        "macro_f1": float(np.mean(f1)),
        "false_edit_rate": float(((target == 0) & (prediction != 0)).mean()),
        "missed_edit_rate": float(((target != 0) & (prediction == 0)).mean()),
        "wrong_edit_rate": float(
            ((target != 0) & (prediction != 0) & (prediction != target)).mean()
        ),
        "target_support": {
            operation.value: support[index]
            for index, operation in enumerate(EditOperation)
        },
    }


def select_train_threshold(
    model: PostToolActionAdapter,
    features: np.ndarray,
    target: np.ndarray,
    device: torch.device,
    false_edit_penalty: float,
    missed_edit_penalty: float,
) -> tuple[float, dict[str, Any]]:
    candidates = np.linspace(0.1, 0.9, 33)
    scored = []
    for threshold in candidates:
        result = metrics(target, predictions(model, features, float(threshold), device))
        score = (
            result["macro_f1"]
            - false_edit_penalty * result["false_edit_rate"]
            - missed_edit_penalty * result["missed_edit_rate"]
        )
        scored.append((score, -float(threshold), result, float(threshold)))
    _, _, result, threshold = max(scored, key=lambda item: (item[0], item[1]))
    return threshold, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_pairs", type=Path)
    parser.add_argument("val_pairs", type=Path)
    parser.add_argument("tool_belief_checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--operation-loss-weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--include-tool-results", action="store_true")
    parser.add_argument("--include-semantic-result", action="store_true")
    parser.add_argument("--use-post-acquisition-belief", action="store_true")
    parser.add_argument("--semantic-only", action="store_true")
    parser.add_argument("--threshold-false-edit-penalty", type=float, default=2.0)
    parser.add_argument("--threshold-missed-edit-penalty", type=float, default=0.5)
    args = parser.parse_args()
    if (
        args.threshold_false_edit_penalty < 0.0
        or args.threshold_missed_edit_penalty < 0.0
    ):
        raise ValueError("threshold penalties must be nonnegative")
    if sum(
        (
            args.include_tool_results,
            args.include_semantic_result,
            args.use_post_acquisition_belief,
            args.semantic_only,
        )
    ) > 1:
        raise ValueError("adapter feature modes are mutually exclusive")
    feature_mode = (
        "semantic"
        if args.include_semantic_result
        else "semantic_only"
        if args.semantic_only
        else "weak_tools"
        if args.include_tool_results
        else "base_belief"
        if args.use_post_acquisition_belief
        else "updated_belief"
    )
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    updater = PairedToolBeliefUpdater.from_checkpoint(
        args.tool_belief_checkpoint,
        device=device,
    )
    train_x, train_y = load_rows(
        args.train_pairs,
        updater,
        "train",
        args.include_tool_results,
        args.include_semantic_result,
        args.use_post_acquisition_belief,
        args.semantic_only,
    )
    val_x, val_y = load_rows(
        args.val_pairs,
        updater,
        "val",
        args.include_tool_results,
        args.include_semantic_result,
        args.use_post_acquisition_belief,
        args.semantic_only,
    )
    feature_mean = train_x.mean(axis=0)
    feature_std = np.maximum(train_x.std(axis=0), 1e-6)
    train_x = (train_x - feature_mean) / feature_std
    val_x = (val_x - feature_mean) / feature_std

    train_target_counts = np.bincount(train_y, minlength=len(EditOperation))
    update_target = train_y != 0
    positive_weight = float((~update_target).sum() / max(update_target.sum(), 1))
    operation_counts = np.maximum(train_target_counts[1:], 1)
    operation_weights = np.sqrt(operation_counts.sum() / operation_counts)
    operation_weights /= operation_weights.mean()
    operation_weights_tensor = torch.as_tensor(
        operation_weights,
        dtype=torch.float32,
        device=device,
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x),
            torch.from_numpy(train_y),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    config = PostToolActionAdapterConfig(
        input_dim=(
            SEMANTIC_ONLY_FEATURE_DIM
            if args.semantic_only
            else SEMANTIC_CONDITIONED_FEATURE_DIM
            if args.include_semantic_result
            else TOOL_CONDITIONED_FEATURE_DIM
            if args.include_tool_results
            else len(feature_mean)
        ),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        include_tool_results=args.include_tool_results,
        include_semantic_result=args.include_semantic_result,
        semantic_only=args.semantic_only,
    )
    model = PostToolActionAdapter(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True)
    best_score = -float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            update_logit, operation_logit = model(features)
            binary_target = (target != 0).float()
            binary_loss = F.binary_cross_entropy_with_logits(
                update_logit,
                binary_target,
                pos_weight=torch.tensor(positive_weight, device=device),
            )
            operation_mask = target != 0
            operation_loss = F.cross_entropy(
                operation_logit[operation_mask],
                target[operation_mask] - 1,
                weight=operation_weights_tensor,
            )
            loss = binary_loss + args.operation_loss_weight * operation_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        threshold, train_metrics = select_train_threshold(
            model,
            train_x,
            train_y,
            device,
            args.threshold_false_edit_penalty,
            args.threshold_missed_edit_penalty,
        )
        val_metrics = metrics(
            val_y,
            predictions(model, val_x, threshold, device),
        )
        score = (
            val_metrics["macro_f1"]
            - args.threshold_false_edit_penalty * val_metrics["false_edit_rate"]
            - args.threshold_missed_edit_penalty * val_metrics["missed_edit_rate"]
        )
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "update_threshold": threshold,
            "selection_score": score,
            "train": train_metrics,
            "val": val_metrics,
        }
        with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(json.dumps(record, separators=(",", ":")), flush=True)
        if score > best_score + 1e-8:
            best_score = score
            stale = 0
            torch.save(
                {
                    "protocol": "post_tool_action_adapter_v1",
                    "model_config": config.as_dict(),
                    "state_dict": model.state_dict(),
                    "feature_mean": feature_mean.tolist(),
                    "feature_std": feature_std.tolist(),
                    "update_threshold": threshold,
                    "epoch": epoch,
                    "seed": args.seed,
                    "feature_mode": feature_mode,
                    "threshold_false_edit_penalty": (
                        args.threshold_false_edit_penalty
                    ),
                    "threshold_missed_edit_penalty": (
                        args.threshold_missed_edit_penalty
                    ),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "test_assets_read": False,
                },
                args.output_dir / "best.pt",
            )
        else:
            stale += 1
        if stale >= args.patience:
            break
    checkpoint = torch.load(
        args.output_dir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    summary = {
        "schema_version": "post-tool-action-adapter-training-v1",
        "train_examples": len(train_y),
        "val_examples": len(val_y),
        "best_epoch": checkpoint["epoch"],
        "update_threshold": checkpoint["update_threshold"],
        "feature_mode": checkpoint["feature_mode"],
        "threshold_false_edit_penalty": checkpoint[
            "threshold_false_edit_penalty"
        ],
        "threshold_missed_edit_penalty": checkpoint[
            "threshold_missed_edit_penalty"
        ],
        "train_metrics": checkpoint["train_metrics"],
        "val_metrics": checkpoint["val_metrics"],
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
