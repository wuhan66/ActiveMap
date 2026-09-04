#!/usr/bin/env python3
"""Train a frozen-feature evidence selector without consulting test assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260841)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _held_out(episode_id: str) -> bool:
    value = int(hashlib.sha256(episode_id.encode()).hexdigest()[:8], 16)
    return value % 5 == 0


def _metrics(records: list[dict[str, Any]], scores: list[float]) -> dict[str, float | int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        grouped[str(row["episode_id"])].append(index)
    selected_utility = oracle_utility = exact = 0.0
    rankings: list[dict[str, Any]] = []
    for episode_id, indices in sorted(grouped.items()):
        ranked = sorted(indices, key=lambda index: (-scores[index], records[index]["evidence_id"]))
        oracle = max(
            indices,
            key=lambda index: (records[index]["utility"], records[index]["evidence_id"]),
        )
        selected_utility += float(records[ranked[0]]["utility"])
        oracle_utility += float(records[oracle]["utility"])
        exact += float(ranked[0] == oracle)
        rankings.append(
            {
                "episode_id": episode_id,
                "ranked_evidence_ids": [records[index]["evidence_id"] for index in ranked],
                "scores": [scores[index] for index in ranked],
            }
        )
    count = len(grouped)
    return {
        "episodes": count,
        "mean_selected_utility": selected_utility / count,
        "mean_oracle_utility": oracle_utility / count,
        "mean_regret": (oracle_utility - selected_utility) / count,
        "exact_top1": exact / count,
        "rankings": rankings,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = [
        json.loads(line)
        for line in (args.features / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    features = np.load(args.features / "features.npy").astype(np.float32)
    if len(records) != len(features):
        raise ValueError("feature and record counts differ")
    costs = np.asarray([float(row["cost"]) for row in records], dtype=np.float32)
    costs = (costs - costs.mean()) / max(float(costs.std()), 1e-6)
    features = np.concatenate([features, costs[:, None]], axis=1)
    targets = np.asarray([float(row["utility"]) for row in records], dtype=np.float32)
    held_out = np.asarray([_held_out(str(row["episode_id"])) for row in records])
    if not held_out.any() or held_out.all():
        raise ValueError("deterministic AOI split is empty")
    mean = features[~held_out].mean(axis=0)
    std = np.maximum(features[~held_out].std(axis=0), 1e-5)
    features = (features - mean) / std

    model = nn.Sequential(
        nn.Linear(features.shape[1], 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1),
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    train_x = torch.from_numpy(features[~held_out])
    train_y = torch.from_numpy(targets[~held_out])
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    val_x = torch.from_numpy(features[held_out]).to(args.device)
    val_y = torch.from_numpy(targets[held_out]).to(args.device)
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            prediction = model(batch_x.to(args.device)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(prediction, batch_y.to(args.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            val_loss = float(nn.functional.smooth_l1_loss(model(val_x).squeeze(1), val_y))
        history.append(
            {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val_loss": val_loss}
        )
        if val_loss < best_loss - 1e-6:
            best_loss, stale = val_loss, 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        scores = model(torch.from_numpy(features).to(args.device)).squeeze(1).cpu().tolist()
    fit_records = [row for row, flag in zip(records, held_out, strict=True) if not flag]
    fit_scores = [score for score, flag in zip(scores, held_out, strict=True) if not flag]
    validation_records = [row for row, flag in zip(records, held_out, strict=True) if flag]
    validation_scores = [score for score, flag in zip(scores, held_out, strict=True) if flag]
    fit_metrics = _metrics(fit_records, fit_scores)
    validation_metrics = _metrics(validation_records, validation_scores)
    args.output_dir.mkdir(parents=True)
    checkpoint = {
        "state_dict": best_state, "feature_mean": mean, "feature_std": std,
        "seed": args.seed, "input_dim": int(features.shape[1]),
    }
    torch.save(checkpoint, args.output_dir / "best.pt")
    (args.output_dir / "history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history), encoding="utf-8"
    )
    rankings = validation_metrics.pop("rankings")
    fit_metrics.pop("rankings")
    (args.output_dir / "calibration_rankings.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rankings),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "activemap-argotweak-visual-selector-v1",
        "seed": args.seed, "fit": fit_metrics, "calibration": validation_metrics,
        "best_validation_loss": best_loss, "epochs_completed": len(history),
        "split_policy": "sha256(episode_id) modulo 5; AOI-disjoint; train only",
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
