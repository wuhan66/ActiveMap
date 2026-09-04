#!/usr/bin/env python3
"""Train an AOI-disjoint listwise selector on executable-map utility."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _held_out(episode_id: str) -> bool:
    return int(hashlib.sha256(episode_id.encode()).hexdigest()[:8], 16) % 5 == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261521)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-temperature", type=float, default=0.08)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    import numpy as np
    import torch
    from torch import nn

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = [
        json.loads(line)
        for line in (args.features / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw = np.load(args.features / "features.npy").astype(np.float32)
    costs = np.asarray([float(row["cost"]) for row in records], dtype=np.float32)
    costs = (costs - costs.mean()) / max(float(costs.std()), 1e-6)
    features = np.concatenate([raw, costs[:, None]], axis=1)
    targets = np.asarray([float(row["utility"]) for row in records], dtype=np.float32)
    held_out = np.asarray([_held_out(str(row["episode_id"])) for row in records])
    mean = features[~held_out].mean(axis=0)
    std = np.maximum(features[~held_out].std(axis=0), 1e-5)
    features = (features - mean) / std
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        groups[str(row["episode_id"])].append(index)
    train_groups = [indices for episode, indices in groups.items() if not _held_out(episode)]
    val_groups = [indices for episode, indices in groups.items() if _held_out(episode)]
    if not train_groups or not val_groups:
        raise ValueError("AOI-disjoint train/calibration split is empty")

    model = nn.Sequential(
        nn.Linear(features.shape[1], 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1),
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    feature_tensor = torch.from_numpy(features).to(args.device)
    target_tensor = torch.from_numpy(targets).to(args.device)

    def group_loss(indices: list[int]) -> Any:
        index = torch.tensor(indices, device=args.device)
        scores = model(feature_tensor[index]).squeeze(1)
        values = target_tensor[index]
        target_distribution = torch.softmax(values / args.target_temperature, dim=0)
        rank_loss = -(target_distribution * torch.log_softmax(scores, dim=0)).sum()
        regression = nn.functional.smooth_l1_loss(scores, values)
        return rank_loss + 0.1 * regression

    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_groups)
        train_losses = []
        for indices in train_groups:
            loss = group_loss(indices)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            val_loss = sum(float(group_loss(indices)) for indices in val_groups) / len(val_groups)
        history.append({"epoch": epoch, "train_loss": sum(train_losses) / len(train_losses), "val_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            stale = 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    assert best_state is not None
    args.output_dir.mkdir(parents=True)
    torch.save(
        {"state_dict": best_state, "feature_mean": mean, "feature_std": std, "seed": args.seed, "input_dim": int(features.shape[1])},
        args.output_dir / "best.pt",
    )
    (args.output_dir / "history.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in history), encoding="utf-8"
    )
    summary = {
        "schema_version": "activemap-argotweak-listwise-selector-v1",
        "seed": args.seed,
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "train_episodes": len(train_groups),
        "calibration_episodes": len(val_groups),
        "objective": "listwise executable-map utility",
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
