#!/usr/bin/env python3
"""Train and calibrate a no-test C5 candidate false-edit risk head.

The model is intentionally narrow: it predicts an executable false edit for a
single updater-f1 candidate from features available before acquisition.  It is
not trained on selected-decision outcomes or future map metrics.  A separate
runner applies its development-frozen threshold to the selector policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class CandidateExample:
    sample_id: str
    split: str
    source_episode: str
    features: tuple[float, ...]
    false_edit: float


class CandidateRiskHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _candidate_features(row: dict[str, Any], index: int) -> tuple[float, ...]:
    # hypothesis, state, candidate descriptor, updater-derived risk and cost
    return tuple(
        float(value)
        for value in (
            *row["hypothesis_features"],
            *row["state_features"],
            *row["evidence_features"][index],
            row["false_edit_risks"][index],
            row["evidence_costs"][index],
        )
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("empty selector state cache")
    if any(str(row.get("split")) == "test" for row in rows):
        raise ValueError("candidate risk training forbids test records")
    if any(str(row.get("split")) not in {"train", "val"} for row in rows):
        raise ValueError("candidate risk support requires train/val records only")
    return rows


def _candidate_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"train_positive": 0, "train_negative": 0}
    for row in rows:
        if str(row["split"]) != "train":
            continue
        outcomes = row["metadata"]["executable_outcomes"]
        for evidence_id in row["evidence_ids"]:
            key = "train_positive" if bool(outcomes[evidence_id]["false_edit"]) else "train_negative"
            counts[key] += 1
    return counts


def _keep_negative(sample_id: str, evidence_id: str, probability: float) -> bool:
    if probability >= 1.0:
        return True
    digest = hashlib.sha256(f"candidate-risk-v1:{sample_id}:{evidence_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < probability


def build_examples(rows: list[dict[str, Any]], *, max_negatives_per_positive: int) -> tuple[list[CandidateExample], list[CandidateExample], dict[str, int]]:
    if max_negatives_per_positive < 1:
        raise ValueError("max_negatives_per_positive must be positive")
    counts = _candidate_counts(rows)
    positive_count = counts["train_positive"]
    if positive_count == 0:
        raise ValueError("train support contains no false-edit labels")
    keep_probability = min(1.0, positive_count * max_negatives_per_positive / max(counts["train_negative"], 1))
    train: list[CandidateExample] = []
    val: list[CandidateExample] = []
    for row in rows:
        split = str(row["split"])
        outcomes = row["metadata"]["executable_outcomes"]
        group = str(row["metadata"].get("source_episode", row["sample_id"]))
        for index, evidence_id in enumerate(row["evidence_ids"]):
            label = float(bool(outcomes[evidence_id]["false_edit"]))
            if split == "train" and not label and not _keep_negative(str(row["sample_id"]), str(evidence_id), keep_probability):
                continue
            example = CandidateExample(
                sample_id=f"{row['sample_id']}::{evidence_id}", split=split,
                source_episode=group, features=_candidate_features(row, index), false_edit=label,
            )
            (train if split == "train" else val).append(example)
    if not train or not val:
        raise ValueError("empty train or development candidate support")
    train_groups = {row.source_episode for row in train}
    val_groups = {row.source_episode for row in val}
    if train_groups & val_groups:
        raise ValueError("candidate support leaks source episodes across train and development")
    counts.update({
        "sampled_train": len(train), "development": len(val),
        "development_positive": int(sum(row.false_edit for row in val)),
        "development_negative": int(sum(not row.false_edit for row in val)),
    })
    return train, val, counts


def _arrays(rows: list[CandidateExample], mean: np.ndarray | None = None, std: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray([row.features for row in rows], dtype=np.float32)
    labels = np.asarray([row.false_edit for row in rows], dtype=np.float32)
    if mean is None or std is None:
        mean = features.mean(axis=0)
        std = features.std(axis=0).clip(min=1e-6)
    return (features - mean) / std, labels, mean, std


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Stable scores are continuous in practice; average rank correction is
    # unnecessary for this development-only diagnostic.
    return float((ranks[labels > 0.5].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def calibrate_threshold(labels: np.ndarray, probabilities: np.ndarray, *, target_false_edit_recall: float) -> dict[str, float]:
    if not 0.0 < target_false_edit_recall <= 1.0:
        raise ValueError("target_false_edit_recall must be in (0, 1]")
    positives = probabilities[labels > 0.5]
    if not len(positives):
        raise ValueError("development support contains no false-edit labels")
    threshold = float(np.quantile(positives, 1.0 - target_false_edit_recall, method="higher"))
    rejected = probabilities >= threshold
    return {
        "risk_threshold": threshold,
        "false_edit_recall": float(rejected[labels > 0.5].mean()),
        "false_positive_rate": float(rejected[labels <= 0.5].mean()),
        "precision": float(labels[rejected].mean()) if bool(rejected.any()) else 0.0,
        "rejection_rate": float(rejected.mean()),
        "development_examples": int(len(labels)),
        "development_false_edits": int(labels.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-negatives-per-positive", type=int, default=32)
    parser.add_argument("--target-false-edit-recall", type=float, default=0.95)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = _read_rows(args.states)
    train, val, counts = build_examples(rows, max_negatives_per_positive=args.max_negatives_per_positive)
    fit_x, fit_y, mean, std = _arrays(train)
    val_x, val_y, _, _ = _arrays(val, mean, std)
    device = torch.device(args.device)
    model = CandidateRiskHead(fit_x.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    positive_weight = (len(fit_y) - float(fit_y.sum())) / max(float(fit_y.sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    loader = DataLoader(TensorDataset(torch.from_numpy(fit_x), torch.from_numpy(fit_y)), batch_size=args.batch_size, shuffle=True, num_workers=0)
    best_state: dict[str, torch.Tensor] | None = None
    best_auc = float("-inf")
    history: list[dict[str, float]] = []
    val_tensor = torch.from_numpy(val_x).to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features.to(device)), labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(val_tensor)).cpu().numpy()
        auc = _auroc(val_y, probabilities)
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "development_auroc": auc})
        if auc > best_auc:
            best_auc = auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probabilities = torch.sigmoid(model(val_tensor)).cpu().numpy()
    calibration = calibrate_threshold(val_y, val_probabilities, target_false_edit_recall=args.target_false_edit_recall)
    args.output_dir.mkdir(parents=True)
    checkpoint = {
        "schema_version": "updater-conditioned-candidate-risk-head-v1",
        "state_dict": model.state_dict(),
        "input_dim": int(fit_x.shape[1]),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "feature_mean": mean.tolist(), "feature_std": std.tolist(),
        "calibration": calibration,
        "states": str(args.states.resolve()), "test_assets_read": False,
    }
    torch.save(checkpoint, args.output_dir / "best.pt")
    summary = {
        "schema_version": "updater-conditioned-candidate-risk-training-v1",
        "counts": counts, "feature_dim": int(fit_x.shape[1]),
        "best_development_auroc": best_auc, "calibration": calibration,
        "history": history, "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
