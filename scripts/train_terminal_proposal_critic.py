#!/usr/bin/env python3
"""Train a no-leak critic that arbitrates VLM proposals against a safe policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ACTIONS = ("KEEP", "ADD", "DELETE", "RESHAPE")


def action_index(value: str) -> int:
    operation = value.split(":", 1)[1] if value.startswith("COMMIT:") else "KEEP"
    return ACTIONS.index(operation)


def encode(features: list[float], baseline: int, candidate: int) -> list[float]:
    return (
        [float(value) for value in features]
        + [float(index == baseline) for index in range(len(ACTIONS))]
        + [float(index == candidate) for index in range(len(ACTIONS))]
    )


def split_bucket(group: str) -> int:
    return int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5


def load_states(path: Path, split: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == split:
                rows.append(row)
    return rows


def build_counterfactual_rows(
    states: list[dict[str, Any]], calibration: bool
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    features, labels, actions = [], [], []
    for row in states:
        group = str(row["metadata"]["source_episode"])
        if (split_bucket(group) == 0) != calibration:
            continue
        target = ACTIONS.index(str(row["metadata"]["gt_edit"]))
        for baseline in range(len(ACTIONS)):
            for candidate in range(len(ACTIONS)):
                if baseline == candidate:
                    continue
                features.append(encode(row["hypothesis_features"], baseline, candidate))
                labels.append(float(candidate == target and baseline != target))
                actions.append((baseline, candidate, target))
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        actions,
    )


def action_score(action: int, target: int) -> float:
    if action == target:
        return 1.0
    if target == 0 and action != 0:
        return -1.25
    if action == 0 and target != 0:
        return -0.75
    return -1.0


def select_threshold(
    probabilities: np.ndarray,
    actions: list[tuple[int, int, int]],
) -> tuple[float, dict[str, float]]:
    thresholds = np.unique(
        np.quantile(probabilities, np.linspace(0.0, 1.0, 201))
    )
    baseline_false = np.mean(
        [target == 0 and baseline != 0 for baseline, _, target in actions]
    )
    best: tuple[tuple[float, float], float, dict[str, float]] | None = None
    for threshold in thresholds:
        selected = [
            candidate if probability >= threshold else baseline
            for probability, (baseline, candidate, _) in zip(probabilities, actions)
        ]
        false_edit = np.mean(
            [
                target == 0 and prediction != 0
                for prediction, (_, _, target) in zip(selected, actions)
            ]
        )
        if false_edit > baseline_false + 0.005:
            continue
        utility = np.mean(
            [
                action_score(prediction, target)
                for prediction, (_, _, target) in zip(selected, actions)
            ]
        )
        accuracy = np.mean(
            [
                prediction == target
                for prediction, (_, _, target) in zip(selected, actions)
            ]
        )
        result = {
            "utility": float(utility),
            "accuracy": float(accuracy),
            "false_edit_rate": float(false_edit),
            "baseline_false_edit_rate": float(baseline_false),
        }
        key = (float(utility), float(accuracy))
        if best is None or key > best[0]:
            best = (key, float(threshold), result)
    if best is None:
        raise RuntimeError("no threshold satisfies the train-only safety constraint")
    return best[1], best[2]


class ProposalCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


def rollout_map(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {str(row["sample_id"]): row for row in rows}


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "terminal_correct",
        "false_edit",
        "missed_edit",
        "episode_utility_v2_proxy_balanced",
        "episode_utility_v2_proxy_safety",
        "spent_cost",
        "tool_calls",
    )
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_states = load_states(args.states, "train")
    val_states = load_states(args.states, "val")
    fit_x, fit_y, _ = build_counterfactual_rows(train_states, calibration=False)
    cal_x, _, cal_actions = build_counterfactual_rows(train_states, calibration=True)
    mean = fit_x.mean(axis=0)
    std = fit_x.std(axis=0).clip(min=1e-6)
    fit_x = (fit_x - mean) / std
    cal_x = (cal_x - mean) / std

    model = ProposalCritic(fit_x.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    positives = fit_y.sum()
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(len(fit_y) - positives) / positives], device=device)
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(fit_x), torch.from_numpy(fit_y)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model.train()
    history = []
    for epoch in range(args.epochs):
        losses = []
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})

    model.eval()
    with torch.no_grad():
        cal_probabilities = torch.sigmoid(
            model(torch.from_numpy(cal_x).to(device))
        ).cpu().numpy()
    threshold, calibration = select_threshold(cal_probabilities, cal_actions)

    state_by_id = {str(row["sample_id"]): row for row in val_states}
    baseline = rollout_map(args.baseline)
    candidate = rollout_map(args.candidate)
    common = sorted(baseline.keys() & candidate.keys() & state_by_id.keys())
    if common != sorted(baseline):
        raise ValueError("validation state/rollout sample mismatch")
    inference_x = np.asarray(
        [
            encode(
                state_by_id[sample_id]["hypothesis_features"],
                action_index(str(baseline[sample_id]["prediction"])),
                action_index(str(candidate[sample_id]["prediction"])),
            )
            for sample_id in common
        ],
        dtype=np.float32,
    )
    inference_x = (inference_x - mean) / std
    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(torch.from_numpy(inference_x).to(device))
        ).cpu().numpy()

    hybrid, accepted = [], 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "hybrid.jsonl").open("x", encoding="utf-8") as handle:
        for sample_id, probability in zip(common, probabilities):
            base_row, candidate_row = baseline[sample_id], candidate[sample_id]
            disagreement = base_row["prediction"] != candidate_row["prediction"]
            use_candidate = disagreement and float(probability) >= threshold
            accepted += int(use_candidate)
            row = dict(candidate_row if use_candidate else base_row)
            row["proposal_critic"] = {
                "probability": float(probability),
                "threshold": threshold,
                "accepted_candidate": use_candidate,
                "baseline_prediction": base_row["prediction"],
                "candidate_prediction": candidate_row["prediction"],
                "fit_split": "train",
                "calibration_split": "train_grouped_holdout",
            }
            hybrid.append(row)
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    checkpoint = {
        "protocol": "terminal_proposal_critic_v1",
        "seed": args.seed,
        "input_dim": fit_x.shape[1],
        "hidden_dim": args.hidden_dim,
        "threshold": threshold,
        "feature_mean": mean,
        "feature_std": std,
        "state_dict": model.cpu().state_dict(),
    }
    torch.save(checkpoint, args.output_dir / "best.pt")
    result = {
        "schema_version": "terminal-proposal-critic-v1",
        "seed": args.seed,
        "train_states": len(train_states),
        "fit_pairs": len(fit_y),
        "calibration_pairs": len(cal_actions),
        "threshold": threshold,
        "calibration": calibration,
        "validation_rows": len(common),
        "accepted_candidate_disagreements": accepted,
        "baseline": summarize(list(baseline.values())),
        "candidate": summarize(list(candidate.values())),
        "hybrid": summarize(hybrid),
        "test_assets_read": False,
    }
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
