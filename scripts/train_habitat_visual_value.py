#!/usr/bin/env python3
"""Train a causal RGB-D value head on train-only Habitat occupancy episodes."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from activemap.data.navigation_map import NavigationMapEpisode
from activemap.evaluation.navigation_rollout import (
    NavigationState,
    _load_occupancy,
    _map_quality,
    _merge_local_observation,
    _valid_mask,
    load_navigation_episodes,
)
from activemap.nn.navigation_visual_value import (
    NavigationVisualValueConfig,
    NavigationVisualValueNet,
    candidate_features,
    current_rgbd_array,
    occupancy_array,
)

TARGET_MODES = ("absolute_gain", "state_relative_gain")


@dataclass(frozen=True)
class ValueExample:
    rgb_path: str | None
    depth_path: str | None
    committed_map: np.ndarray
    candidate: np.ndarray
    target_gain: float


def state_targets(gains: list[float], *, mode: str) -> np.ndarray:
    """Return absolute gains or state-normalized candidate ranking targets."""
    if mode not in TARGET_MODES:
        raise ValueError(f"unsupported target mode: {mode}")
    values = np.asarray(gains, dtype=np.float32)
    if not len(values):
        raise ValueError("state gains cannot be empty")
    if mode == "absolute_gain":
        return values
    return (values - values.mean()) / max(float(values.std()), 1e-6)


def build_value_examples(
    episodes: list[NavigationMapEpisode], *, target_mode: str = "absolute_gain"
) -> list[ValueExample]:
    """Teacher-force observed history while labeling future candidate value.

    Reference maps are opened only to form regression labels. At inference the
    model receives neither target occupancy nor candidate RGB-D.
    """
    examples: list[ValueExample] = []
    for episode in episodes:
        if episode.split == "test":
            raise PermissionError("test episodes cannot train visual value")
        target = _load_occupancy(episode.target_map_path)
        committed = _load_occupancy(episode.initial_map_path, expected_shape=target.shape)
        valid = _valid_mask(episode, target.shape)
        ordered = sorted(episode.evidence, key=lambda row: (row.timestamp, row.evidence_id))
        current_rgb, current_depth = episode.initial_rgb_path, episode.initial_depth_path
        current_pose = episode.start_pose
        spent_cost = 0.0
        for index, observed in enumerate(ordered):
            state = NavigationState(
                committed_map=committed,
                pose=current_pose,
                remaining_budget=max(episode.budget - spent_cost, 0.0),
                acquired_evidence_ids=tuple(row.evidence_id for row in ordered[:index]),
                step=index,
                pixels_per_meter=float(episode.metadata.get("pixels_per_meter", 1.0)),
                grid_x_min=float(episode.metadata.get("grid_x_min", 0.0)),
                grid_z_max=float(episode.metadata.get("grid_z_max", 0.0)),
                visual_rgb_path=current_rgb,
                visual_depth_path=current_depth,
            )
            quality_before = _map_quality(committed, target, valid)
            candidate_rows: list[tuple[np.ndarray, float]] = []
            for candidate in ordered[index:]:
                proposed, _ = _merge_local_observation(
                    committed, _load_occupancy(candidate.path, expected_shape=target.shape)
                )
                candidate_rows.append(
                    (
                        candidate_features(state, candidate),
                        _map_quality(proposed, target, valid) - quality_before,
                    )
                )
            targets = state_targets(
                [gain for _, gain in candidate_rows], mode=target_mode
            )
            for (candidate, _), target_gain in zip(candidate_rows, targets, strict=True):
                examples.append(
                    ValueExample(
                        rgb_path=current_rgb,
                        depth_path=current_depth,
                        committed_map=committed.copy(),
                        candidate=candidate,
                        target_gain=float(target_gain),
                    )
                )
            committed, _ = _merge_local_observation(
                committed, _load_occupancy(observed.path, expected_shape=target.shape)
            )
            current_rgb = observed.rgb_path or current_rgb
            current_depth = observed.depth_path or current_depth
            current_pose = observed.pose or current_pose
            spent_cost += observed.cost
    if not examples:
        raise ValueError("no visual-value training examples were built")
    return examples


class ValueDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        examples: list[ValueExample],
        *,
        image_size: int,
        target_mean: float,
        target_std: float,
    ) -> None:
        self.examples = examples
        self.image_size = image_size
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.examples[index]
        return {
            "rgbd": torch.from_numpy(
                current_rgbd_array(row.rgb_path, row.depth_path, image_size=self.image_size)
            ),
            "map": torch.from_numpy(occupancy_array(row.committed_map, image_size=self.image_size)),
            "candidate": torch.from_numpy(row.candidate),
            "target": torch.tensor((row.target_gain - self.target_mean) / self.target_std),
        }


@torch.no_grad()
def evaluate(
    model: NavigationVisualValueNet,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        prediction = model(
            batch["rgbd"].to(device), batch["map"].to(device), batch["candidate"].to(device)
        )
        losses.append(float(nn.functional.mse_loss(prediction, batch["target"].to(device)).item()))
    return float(np.mean(losses)) if losses else float("inf")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_index", type=Path)
    parser.add_argument("val_index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20270831)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--cost-weight", type=float, default=0.0)
    parser.add_argument("--stop-margin", type=float, default=0.0)
    parser.add_argument("--target-mode", choices=TARGET_MODES, default="absolute_gain")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, and batch-size must be positive")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = build_value_examples(
        load_navigation_episodes(args.train_index, split="train"), target_mode=args.target_mode
    )
    val = build_value_examples(
        load_navigation_episodes(args.val_index, split="val"), target_mode=args.target_mode
    )
    target_values = np.asarray([row.target_gain for row in train], dtype=np.float32)
    target_mean = float(target_values.mean())
    target_std = max(float(target_values.std()), 1e-6)
    train_loader = DataLoader(
        ValueDataset(
            train,
            image_size=args.image_size,
            target_mean=target_mean,
            target_std=target_std,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        ValueDataset(
            val,
            image_size=args.image_size,
            target_mean=target_mean,
            target_std=target_std,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=args.device.startswith("cuda"),
    )
    device = torch.device(args.device)
    config = NavigationVisualValueConfig(image_size=args.image_size)
    model = NavigationVisualValueNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history: list[dict[str, float | int]] = []
    args.output.mkdir(parents=True)
    checkpoint = args.output / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch["rgbd"].to(device), batch["map"].to(device), batch["candidate"].to(device)
            )
            loss = nn.functional.mse_loss(prediction, batch["target"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss = evaluate(model, val_loader, device)
        history.append(
            {"epoch": epoch, "train_mse": float(np.mean(train_losses)), "val_mse": val_loss}
        )
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save(
                {
                    "protocol": "navigation_visual_value_v1",
                    "model_config": config.as_dict(),
                    "state_dict": model.state_dict(),
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "cost_weight": args.cost_weight,
                    "stop_margin": args.stop_margin,
                    "target_mode": args.target_mode,
                    "train_index": str(args.train_index.resolve()),
                    "val_index": str(args.val_index.resolve()),
                    "seed": args.seed,
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    (args.output / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "protocol": "navigation_visual_value_v1",
                "development_only": True,
                "train_examples": len(train),
                "val_examples": len(val),
                "best_epoch": best_epoch,
                "best_val_mse": best_loss,
                "parameter_count": model.parameter_count(),
                "candidate_visual_access": "forbidden before ACQUIRE",
                "target_mode": args.target_mode,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
