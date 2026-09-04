#!/usr/bin/env python3
"""Train terminal arbitration from out-of-fold Qwen proposals."""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from scripts.train_terminal_proposal_critic import (
        ACTIONS,
        ProposalCritic,
        action_index,
        encode,
        load_states,
        rollout_map,
        select_threshold,
        split_bucket,
        summarize,
    )
except ModuleNotFoundError:
    from train_terminal_proposal_critic import (
        ACTIONS,
        ProposalCritic,
        action_index,
        encode,
        load_states,
        rollout_map,
        select_threshold,
        split_bucket,
        summarize,
    )


def belief_features(belief: dict[str, Any]) -> list[float]:
    probabilities = [float(value) for value in belief["edit_probabilities"]]
    geometry = [float(value) for value in belief["geometry_delta"]]
    return (
        probabilities
        + [float(belief["confidence"])]
        + geometry
        + [float(belief["uncertainty"])]
    )


def state_belief_features(row: dict[str, Any]) -> list[float]:
    metadata = row["metadata"]
    prediction = metadata["evidence_predictions"][metadata["initial_evidence_id"]]
    probabilities = np.asarray(prediction["edit_probabilities"], dtype=np.float64)
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * np.log(positive)).sum() / math.log(len(ACTIONS)))
    return belief_features(
        {
            "edit_probabilities": probabilities.tolist(),
            "confidence": prediction["confidence"],
            "geometry_delta": prediction["geometry_delta"],
            "uncertainty": entropy,
        }
    )


def terminal_operation(action: dict[str, Any]) -> str | None:
    if action["action"] == "REJECT":
        return "KEEP"
    if action["action"] == "COMMIT":
        return str(action["edit"])
    return None


def target_map(path: Path) -> dict[str, str]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[str(row["task_id"])] = str(row["target_operation"])
    return result


def parse_oof_rows(
    root: Path,
    targets: dict[str, str],
    generated_root: Path | None = None,
) -> list[tuple[str, list[float], int, int, int]]:
    rows = []
    for name in sorted(glob.glob(str(root / "fold*" / "holdout_rollout.jsonl"))):
        fold = Path(name).parent.name
        generated = None
        if generated_root is not None:
            generated_path = generated_root / fold / "predictions.jsonl"
            if not generated_path.is_file():
                raise FileNotFoundError(generated_path)
            generated = {}
            with generated_path.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    generated[str(record["example_id"])] = record
        with Path(name).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["stage"] != "PRE_TOOL":
                    continue
                task_id = str(row["task_id"])
                user_payload = None
                for message in row["messages"]:
                    if message["role"] != "user":
                        continue
                    for content in message["content"]:
                        text = content.get("text")
                        if text and text.startswith("{"):
                            user_payload = json.loads(text)
                if generated is None:
                    assistant = json.loads(row["messages"][-1]["content"][0]["text"])
                    candidate_operation = terminal_operation(assistant)
                    target_operation = targets[task_id]
                else:
                    prediction = generated[str(row["example_id"])]
                    if bool(prediction["predicted_use_tool"]):
                        continue
                    candidate_operation = str(prediction["predicted_operation"])
                    target_operation = str(prediction["target_operation"])
                if user_payload is None or candidate_operation is None:
                    continue
                baseline = ACTIONS.index(str(row["consensus"]["baseline_operation"]))
                candidate = ACTIONS.index(candidate_operation)
                if baseline == candidate:
                    continue
                rows.append(
                    (
                        task_id,
                        belief_features(user_payload["belief"]),
                        baseline,
                        candidate,
                        ACTIONS.index(target_operation),
                    )
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("oof_root", type=Path)
    parser.add_argument("target_trajectories", type=Path)
    parser.add_argument("states", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--generated-root", type=Path)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    oof = parse_oof_rows(
        args.oof_root,
        target_map(args.target_trajectories),
        generated_root=args.generated_root,
    )
    fit = [row for row in oof if split_bucket(row[0]) != 0]
    calibration_rows = [row for row in oof if split_bucket(row[0]) == 0]
    fit_x = np.asarray(
        [encode(features, baseline, candidate) for _, features, baseline, candidate, _ in fit],
        dtype=np.float32,
    )
    fit_y = np.asarray(
        [float(candidate == target) for _, _, _, candidate, target in fit],
        dtype=np.float32,
    )
    cal_x = np.asarray(
        [
            encode(features, baseline, candidate)
            for _, features, baseline, candidate, _ in calibration_rows
        ],
        dtype=np.float32,
    )
    cal_actions = [
        (baseline, candidate, target)
        for _, _, baseline, candidate, target in calibration_rows
    ]
    mean = fit_x.mean(axis=0)
    std = fit_x.std(axis=0).clip(min=1e-6)
    fit_x = (fit_x - mean) / std
    cal_x = (cal_x - mean) / std

    model = ProposalCritic(fit_x.shape[1], args.hidden_dim).to(device)
    positives = fit_y.sum()
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(len(fit_y) - positives) / positives], device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(fit_x), torch.from_numpy(fit_y)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    history = []
    for epoch in range(args.epochs):
        model.train()
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

    states = {str(row["sample_id"]): row for row in load_states(args.states, "val")}
    baseline = rollout_map(args.baseline)
    candidate = rollout_map(args.candidate)
    common = sorted(baseline.keys() & candidate.keys() & states.keys())
    inference_x = np.asarray(
        [
            encode(
                state_belief_features(states[sample_id]),
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

    args.output_dir.mkdir(parents=True, exist_ok=False)
    hybrid, accepted = [], 0
    with (args.output_dir / "hybrid.jsonl").open("x", encoding="utf-8") as handle:
        for sample_id, probability in zip(common, probabilities):
            base_row, candidate_row = baseline[sample_id], candidate[sample_id]
            disagreement = base_row["prediction"] != candidate_row["prediction"]
            use_candidate = disagreement and float(probability) >= threshold
            accepted += int(use_candidate)
            row = dict(candidate_row if use_candidate else base_row)
            row["proposal_critic"] = {
                "protocol": "oof_terminal_proposal_critic_v1",
                "probability": float(probability),
                "threshold": threshold,
                "accepted_candidate": use_candidate,
                "fit_split": "train_oof",
                "calibration_split": "train_oof_grouped_holdout",
            }
            hybrid.append(row)
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    torch.save(
        {
            "protocol": "oof_terminal_proposal_critic_v1",
            "seed": args.seed,
            "input_dim": fit_x.shape[1],
            "hidden_dim": args.hidden_dim,
            "threshold": threshold,
            "feature_mean": mean,
            "feature_std": std,
            "state_dict": model.cpu().state_dict(),
        },
        args.output_dir / "best.pt",
    )
    result = {
        "schema_version": "oof-terminal-proposal-critic-v1",
        "seed": args.seed,
        "oof_disagreements": len(oof),
        "oof_source": "generated_predictions"
        if args.generated_root is not None
        else "supervised_targets_invalid_for_promotion",
        "fit_rows": len(fit),
        "calibration_rows": len(calibration_rows),
        "fit_positive_rate": float(fit_y.mean()),
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
