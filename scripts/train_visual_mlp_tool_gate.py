#!/usr/bin/env python3
"""Train a nonlinear task-grouped utility-aware gate on frozen VLM states."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from joblib import dump
from sklearn.model_selection import StratifiedGroupKFold

from activemap.agent.visual_gate import NumpyMLPGate, binary_call_metrics
from scripts.train_visual_tool_gate import _load, utility_proxy_metrics, utility_risk_weights


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit(
    features: np.ndarray,
    labels: np.ndarray,
    utilities: np.ndarray,
    *,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    seed: int,
    device: str,
) -> NumpyMLPGate:
    import torch
    from torch import nn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-5] = 1.0
    normalized = ((features - mean) / scale).astype(np.float32)
    x = torch.from_numpy(normalized)
    y = torch.from_numpy(labels.astype(np.float32))
    weights = torch.from_numpy(utility_risk_weights(utilities).astype(np.float32))
    model = nn.Sequential(
        nn.Linear(features.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        permutation = torch.randperm(len(x), generator=generator)
        model.train()
        for start in range(0, len(x), batch_size):
            indices = permutation[start : start + batch_size]
            batch_x = x[indices].to(device)
            batch_y = y[indices].to(device)
            batch_weights = weights[indices].to(device)
            logits = model(batch_x).squeeze(1)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, batch_y, reduction="none"
            )
            loss = (losses * batch_weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    first = model[0]
    second = model[3]
    return NumpyMLPGate(
        mean=mean,
        scale=scale,
        weight1=first.weight.detach().float().cpu().numpy().T,
        bias1=first.bias.detach().float().cpu().numpy(),
        weight2=second.weight.detach().float().cpu().numpy().reshape(-1),
        bias2=float(second.bias.detach().float().cpu().item()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_features", type=Path)
    parser.add_argument("val_features", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument(
        "--thresholds",
        default="0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.975",
    )
    parser.add_argument("--max-call-rate", type=float, default=0.50)
    parser.add_argument("--min-oof-recall", type=float, default=0.10)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    train_x, train_y, train_u, train_groups, _, train_summary = _load(
        args.train_features, "train"
    )
    val_x, val_y, val_u, _, val_ids, val_summary = _load(args.val_features, "val")
    if train_x.shape[1] != val_x.shape[1]:
        raise ValueError("train and validation feature dimensions differ")
    if train_summary.get("utility_metadata") != val_summary.get("utility_metadata"):
        raise ValueError("train and validation utility metadata differ")
    folds = min(
        5,
        len(np.unique(train_groups[train_y == 1])),
        len(np.unique(train_groups[train_y == 0])),
    )
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=args.seed)
    oof = np.full(len(train_y), np.nan, dtype=np.float64)
    for fold, (fit_indices, holdout_indices) in enumerate(
        splitter.split(train_x, train_y, train_groups)
    ):
        model = _fit(
            train_x[fit_indices],
            train_y[fit_indices],
            train_u[fit_indices],
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            seed=args.seed + fold,
            device=args.device,
        )
        oof[holdout_indices] = model.predict_proba(train_x[holdout_indices])[:, 1]
    thresholds = tuple(float(value) for value in args.thresholds.split(","))
    grid = []
    for threshold in thresholds:
        metrics = binary_call_metrics(train_y, oof, threshold)
        metrics.update(utility_proxy_metrics(oof, train_u, threshold))
        grid.append({"threshold": threshold, "metrics": metrics})
    feasible = [
        row
        for row in grid
        if 0.0 < row["metrics"]["call_rate"] <= args.max_call_rate
        and row["metrics"]["recall"] >= args.min_oof_recall
    ]
    if not feasible:
        raise RuntimeError("no nonlinear OOF gate satisfies call-rate and recall constraints")
    selected = max(
        feasible,
        key=lambda row: (
            row["metrics"]["proxy_utility_mean"],
            -row["metrics"]["proxy_risk_mean"],
            row["metrics"]["precision"],
            -row["metrics"]["call_rate"],
        ),
    )
    model = _fit(
        train_x,
        train_y,
        train_u,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
    )
    val_probability = model.predict_proba(val_x)[:, 1]
    threshold = float(selected["threshold"])
    validation = binary_call_metrics(val_y, val_probability, threshold)
    validation.update(utility_proxy_metrics(val_probability, val_u, threshold))
    promotion = {
        "positive_train_oof_utility": selected["metrics"]["proxy_utility_sum"] > 0.0,
        "validation_nonzero_calls": validation["predicted_calls"] > 0,
        "validation_call_rate_within_cap": validation["call_rate"] <= args.max_call_rate,
        "validation_recall_at_least_0_10": validation["recall"] >= 0.10,
    }
    args.output_dir.mkdir(parents=True)
    dump(model, args.output_dir / "gate.joblib")
    (args.output_dir / "selection_grid.json").write_text(
        json.dumps(grid, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "validation_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for example_id, target, utility, probability in zip(
            val_ids, val_y, val_u, val_probability, strict=True
        ):
            handle.write(
                json.dumps(
                    {
                        "example_id": example_id,
                        "target_use_tool": bool(target),
                        "policy_relative_realized_advantage": float(utility),
                        "call_probability": float(probability),
                        "predicted_use_tool": bool(probability >= threshold),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    summary: dict[str, Any] = {
        "schema_version": "semantic-vlm-visual-mlp-tool-gate-v1",
        "selection_protocol": "stratified-task-grouped-OOF-train-only",
        "feature_protocol": "frozen-adapter-PRE_TOOL-last-mean-hidden-state",
        "score_type": "call_probability",
        "selection_objective": "proxy_utility",
        "fit_weighting": "utility_risk",
        "architecture": {
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
        },
        "seed": args.seed,
        "train_examples": len(train_y),
        "validation_examples": len(val_y),
        "feature_dim": int(train_x.shape[1]),
        "selected": selected,
        "validation": validation,
        "promotion_gate": {**promotion, "passed": all(promotion.values())},
        "sources": {
            "train": {"summary_sha256": _sha256(args.train_features / "summary.json")},
            "val": {"summary_sha256": _sha256(args.val_features / "summary.json")},
        },
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
