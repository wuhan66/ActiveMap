#!/usr/bin/env python3
"""Train a no-test residual-utility and false-edit risk terminal scorer."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from scripts.train_oof_terminal_proposal_critic import (
        belief_features,
        state_belief_features,
    )
    from scripts.train_terminal_proposal_critic import (
        ACTIONS,
        action_index,
        encode,
        load_states,
        rollout_map,
        split_bucket,
        summarize,
    )
except ModuleNotFoundError:
    from train_oof_terminal_proposal_critic import (
        belief_features,
        state_belief_features,
    )
    from train_terminal_proposal_critic import (
        ACTIONS,
        action_index,
        encode,
        load_states,
        rollout_map,
        split_bucket,
        summarize,
    )


@dataclass(frozen=True)
class OOFExample:
    task_id: str
    features: list[float]
    baseline: int
    candidate: int
    target: int
    baseline_utility: float
    candidate_utility: float

    @property
    def utility_delta(self) -> float:
        return self.candidate_utility - self.baseline_utility

    @property
    def candidate_false_edit(self) -> float:
        return float(self.target == 0 and self.candidate != 0)

    @property
    def baseline_false_edit(self) -> float:
        return float(self.target == 0 and self.baseline != 0)

    @property
    def candidate_missed_edit(self) -> float:
        return float(self.target != 0 and self.candidate == 0)

    @property
    def baseline_missed_edit(self) -> float:
        return float(self.target != 0 and self.baseline == 0)


class ResidualRiskScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.utility_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        return (
            self.utility_head(hidden).squeeze(-1),
            self.risk_head(hidden).squeeze(-1),
        )


def _user_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    for message in row["messages"]:
        if message["role"] != "user":
            continue
        for content in message["content"]:
            text = content.get("text")
            if text and text.startswith("{"):
                return json.loads(text)
    return None


def parse_generated_oof_rows(
    oof_root: Path,
    generated_root: Path,
) -> list[OOFExample]:
    rows: list[OOFExample] = []
    for name in sorted(glob.glob(str(oof_root / "fold*" / "holdout_rollout.jsonl"))):
        fold = Path(name).parent.name
        generated_path = generated_root / fold / "predictions.jsonl"
        if not generated_path.is_file():
            raise FileNotFoundError(generated_path)
        generated: dict[str, dict[str, Any]] = {}
        with generated_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                generated[str(record["example_id"])] = record
        with Path(name).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["stage"] != "PRE_TOOL":
                    continue
                prediction = generated.get(str(row["example_id"]))
                payload = _user_payload(row)
                if prediction is None or payload is None:
                    continue
                baseline = ACTIONS.index(str(prediction["baseline_operation"]))
                candidate = ACTIONS.index(str(prediction["predicted_operation"]))
                if baseline == candidate:
                    continue
                rows.append(
                    OOFExample(
                        task_id=str(row["task_id"]),
                        features=belief_features(payload["belief"]),
                        baseline=baseline,
                        candidate=candidate,
                        target=ACTIONS.index(str(prediction["target_operation"])),
                        baseline_utility=float(prediction["baseline_utility"]),
                        candidate_utility=float(prediction["policy_utility"]),
                    )
                )
    if not rows:
        raise ValueError("no generated OOF disagreements were found")
    return rows


def build_dagger_risk_support(
    states: list[dict[str, Any]], *, calibration: bool
) -> tuple[np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    labels: list[float] = []
    for row in states:
        group = str(row["metadata"]["source_episode"])
        if (split_bucket(group) == 0) != calibration:
            continue
        metadata = row["metadata"]
        initial = metadata["evidence_predictions"][metadata["initial_evidence_id"]]
        baseline = ACTIONS.index(str(initial["gated_edit"]))
        target = ACTIONS.index(str(metadata["gt_edit"]))
        state_features = state_belief_features(row)
        for candidate in range(len(ACTIONS)):
            if candidate == baseline:
                continue
            features.append(encode(state_features, baseline, candidate))
            labels.append(float(target == 0 and candidate != 0))
    if not features:
        raise ValueError("empty DAgger risk support")
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def select_risk_cap(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    positives = probabilities[labels > 0.5]
    if not len(positives):
        raise ValueError("risk calibration has no false-edit positives")
    positives = np.sort(positives)
    cap = float(positives[int(math.floor(0.05 * (len(positives) - 1)))])
    predicted_harmful = probabilities >= cap
    recall = float(predicted_harmful[labels > 0.5].mean())
    false_positive_rate = float(predicted_harmful[labels <= 0.5].mean())
    return {
        "risk_cap": cap,
        "false_edit_recall": recall,
        "false_positive_rate": false_positive_rate,
        "positive_rows": int((labels > 0.5).sum()),
        "rows": len(labels),
    }


def _conditional_rate(values: np.ndarray, targets: np.ndarray, keep: bool) -> float:
    mask = targets == 0 if keep else targets != 0
    return float(values[mask].mean()) if bool(mask.any()) else 0.0


def select_safe_policy(
    predicted_delta: np.ndarray,
    predicted_risk: np.ndarray,
    rows: list[OOFExample],
    *,
    lambdas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    false_edit_tolerance: float = 0.0,
    risk_cap: float = 1.0,
) -> dict[str, float]:
    targets = np.asarray([row.target for row in rows])
    baseline_false = np.asarray([row.baseline_false_edit for row in rows])
    candidate_false = np.asarray([row.candidate_false_edit for row in rows])
    baseline_missed = np.asarray([row.baseline_missed_edit for row in rows])
    candidate_missed = np.asarray([row.candidate_missed_edit for row in rows])
    utility_delta = np.asarray([row.utility_delta for row in rows])
    baseline_false_rate = _conditional_rate(baseline_false, targets, keep=True)
    best: tuple[tuple[float, float, float, float], dict[str, float]] | None = None
    for risk_lambda in lambdas:
        scores = predicted_delta - risk_lambda * predicted_risk
        margins = np.unique(
            np.concatenate(
                ([float(scores.max()) + 1e-6], np.quantile(scores, np.linspace(0, 1, 101)))
            )
        )
        for margin in margins:
            accepted = (scores >= margin) & (predicted_risk <= risk_cap)
            selected_false = np.where(accepted, candidate_false, baseline_false)
            false_rate = _conditional_rate(selected_false, targets, keep=True)
            if false_rate > baseline_false_rate + false_edit_tolerance + 1e-12:
                continue
            selected_missed = np.where(accepted, candidate_missed, baseline_missed)
            missed_rate = _conditional_rate(selected_missed, targets, keep=False)
            utility_gain = float(np.where(accepted, utility_delta, 0.0).mean())
            acceptance_rate = float(accepted.mean())
            result = {
                "risk_lambda": float(risk_lambda),
                "margin": float(margin),
                "utility_gain": utility_gain,
                "false_edit_rate": false_rate,
                "baseline_false_edit_rate": baseline_false_rate,
                "missed_edit_rate": missed_rate,
                "acceptance_rate": acceptance_rate,
                "risk_cap": float(risk_cap),
            }
            key = (utility_gain, -false_rate, -missed_rate, -acceptance_rate)
            if best is None or key > best[0]:
                best = (key, result)
    if best is None:
        raise RuntimeError("no calibration policy satisfies the false-edit constraint")
    return best[1]


def _metric_deltas(
    baseline: dict[str, float], hybrid: dict[str, float]
) -> dict[str, float]:
    return {key: float(hybrid[key] - baseline[key]) for key in baseline}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("oof_root", type=Path)
    parser.add_argument("generated_root", type=Path)
    parser.add_argument("states", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--risk-loss-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-weight", type=float, default=2.0)
    parser.add_argument("--false-edit-tolerance", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    oof = parse_generated_oof_rows(args.oof_root, args.generated_root)
    fit = [row for row in oof if split_bucket(row.task_id) != 0]
    calibration_rows = [row for row in oof if split_bucket(row.task_id) == 0]
    if not fit or not calibration_rows:
        raise ValueError("grouped OOF fit/calibration split is empty")

    oof_fit_x = np.asarray(
        [encode(row.features, row.baseline, row.candidate) for row in fit],
        dtype=np.float32,
    )
    cal_x = np.asarray(
        [encode(row.features, row.baseline, row.candidate) for row in calibration_rows],
        dtype=np.float32,
    )
    fit_delta_oof = np.asarray([row.utility_delta for row in fit], dtype=np.float32)
    fit_risk_oof = np.asarray(
        [row.candidate_false_edit for row in fit], dtype=np.float32
    )
    train_states = load_states(args.states, "train")
    risk_fit_x, risk_fit_y = build_dagger_risk_support(
        train_states, calibration=False
    )
    risk_cal_x, risk_cal_y = build_dagger_risk_support(
        train_states, calibration=True
    )
    fit_x = np.concatenate((oof_fit_x, risk_fit_x), axis=0)
    fit_delta = np.concatenate(
        (fit_delta_oof, np.zeros(len(risk_fit_y), dtype=np.float32))
    )
    fit_risk = np.concatenate((fit_risk_oof, risk_fit_y))
    utility_mask = np.concatenate(
        (
            np.ones(len(fit_delta_oof), dtype=np.float32),
            np.zeros(len(risk_fit_y), dtype=np.float32),
        )
    )
    utility_weight = np.concatenate(
        (
            1.0
            + args.hard_negative_weight
            * (fit_delta_oof < 0).astype(np.float32),
            np.zeros(len(risk_fit_y), dtype=np.float32),
        )
    )
    oof_risk_weight = len(risk_fit_y) / max(len(fit_risk_oof), 1)
    risk_sample_weight = np.concatenate(
        (
            np.full(len(fit_risk_oof), oof_risk_weight, dtype=np.float32),
            np.ones(len(risk_fit_y), dtype=np.float32),
        )
    )
    mean = fit_x.mean(axis=0)
    std = fit_x.std(axis=0).clip(min=1e-6)
    fit_x = (fit_x - mean) / std
    cal_x = (cal_x - mean) / std
    risk_cal_x = (risk_cal_x - mean) / std

    model = ResidualRiskScorer(fit_x.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    positives = float(fit_risk.sum())
    risk_pos_weight = (len(fit_risk) - positives) / max(positives, 1.0)
    risk_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([risk_pos_weight], device=device), reduction="none"
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(fit_x),
            torch.from_numpy(fit_delta),
            torch.from_numpy(fit_risk),
            torch.from_numpy(utility_mask),
            torch.from_numpy(utility_weight),
            torch.from_numpy(risk_sample_weight),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    history: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, float, float], dict[str, Any], dict[str, Any]] | None = None
    for epoch in range(args.epochs):
        model.train()
        losses, utility_losses, risk_losses = [], [], []
        for features, delta, risk, utility_valid, utility_weight_batch, risk_weight in loader:
            features = features.to(device)
            delta = delta.to(device)
            risk = risk.to(device)
            utility_valid = utility_valid.to(device)
            utility_weight_batch = utility_weight_batch.to(device)
            risk_weight = risk_weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted_delta, risk_logit = model(features)
            utility_terms = nn.functional.smooth_l1_loss(
                predicted_delta, delta, reduction="none"
            ) * utility_weight_batch
            utility_loss = utility_terms.sum() / utility_valid.sum().clamp_min(1.0)
            risk_loss = (risk_loss_fn(risk_logit, risk) * risk_weight).sum() / risk_weight.sum()
            loss = utility_loss + args.risk_loss_weight * risk_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            utility_losses.append(float(utility_loss.item()))
            risk_losses.append(float(risk_loss.item()))
        model.eval()
        with torch.no_grad():
            cal_delta, cal_risk_logit = model(torch.from_numpy(cal_x).to(device))
            cal_delta_np = cal_delta.cpu().numpy()
            cal_risk_np = torch.sigmoid(cal_risk_logit).cpu().numpy()
            _, risk_support_logit = model(torch.from_numpy(risk_cal_x).to(device))
            risk_support_prob = torch.sigmoid(risk_support_logit).cpu().numpy()
        deployment_risk_calibration = select_risk_cap(
            cal_risk_np,
            np.asarray(
                [row.candidate_false_edit for row in calibration_rows],
                dtype=np.float32,
            ),
        )
        dagger_risk_calibration = select_risk_cap(risk_support_prob, risk_cal_y)
        calibration = select_safe_policy(
            cal_delta_np,
            cal_risk_np,
            calibration_rows,
            false_edit_tolerance=args.false_edit_tolerance,
            risk_cap=deployment_risk_calibration["risk_cap"],
        )
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "utility_loss": float(np.mean(utility_losses)),
            "risk_loss": float(np.mean(risk_losses)),
            "calibration": calibration,
            "deployment_risk_calibration": deployment_risk_calibration,
            "dagger_risk_calibration": dagger_risk_calibration,
        }
        history.append(row)
        key = (
            calibration["utility_gain"],
            -calibration["false_edit_rate"],
            -calibration["missed_edit_rate"],
            -calibration["acceptance_rate"],
        )
        if best is None or key > best[0]:
            best = (key, copy.deepcopy(model.state_dict()), row)
    assert best is not None
    model.load_state_dict(best[1])
    calibration = best[2]["calibration"]
    deployment_risk_calibration = best[2]["deployment_risk_calibration"]
    dagger_risk_calibration = best[2]["dagger_risk_calibration"]

    states = {str(row["sample_id"]): row for row in load_states(args.states, "val")}
    baseline = rollout_map(args.baseline)
    candidate = rollout_map(args.candidate)
    common = sorted(baseline.keys() & candidate.keys() & states.keys())
    if common != sorted(baseline):
        raise ValueError("validation state/rollout sample mismatch")
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
    model.eval()
    with torch.no_grad():
        predicted_delta, risk_logit = model(torch.from_numpy(inference_x).to(device))
        predicted_delta = predicted_delta.cpu().numpy()
        predicted_risk = torch.sigmoid(risk_logit).cpu().numpy()
    scores = predicted_delta - calibration["risk_lambda"] * predicted_risk

    args.output_dir.mkdir(parents=True, exist_ok=False)
    hybrid: list[dict[str, Any]] = []
    accepted = 0
    disagreements = 0
    with (args.output_dir / "hybrid.jsonl").open("x", encoding="utf-8") as handle:
        for sample_id, delta, risk, score in zip(
            common, predicted_delta, predicted_risk, scores, strict=True
        ):
            base_row, candidate_row = baseline[sample_id], candidate[sample_id]
            disagreement = base_row["prediction"] != candidate_row["prediction"]
            use_candidate = (
                disagreement
                and float(score) >= calibration["margin"]
                and float(risk) <= deployment_risk_calibration["risk_cap"]
            )
            disagreements += int(disagreement)
            accepted += int(use_candidate)
            row = dict(candidate_row if use_candidate else base_row)
            row["residual_risk_scorer"] = {
                "protocol": "residual_risk_terminal_scorer_v1",
                "predicted_utility_delta": float(delta),
                "predicted_false_edit_risk": float(risk),
                "risk_adjusted_score": float(score),
                "risk_lambda": calibration["risk_lambda"],
                "margin": calibration["margin"],
                "risk_cap": deployment_risk_calibration["risk_cap"],
                "accepted_candidate": use_candidate,
                "fit_split": "train_oof",
                "calibration_split": "train_oof_grouped_holdout",
            }
            hybrid.append(row)
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    baseline_summary = summarize(list(baseline.values()))
    candidate_summary = summarize(list(candidate.values()))
    hybrid_summary = summarize(hybrid)
    deltas = _metric_deltas(baseline_summary, hybrid_summary)
    acceptance_rate = accepted / max(disagreements, 1)
    pilot_gate = {
        "positive_balanced_utility": deltas["episode_utility_v2_proxy_balanced"] > 0,
        "false_edit_not_increased": deltas["false_edit"] <= 0,
        "missed_edit_not_materially_worse": deltas["missed_edit"] <= 0.005,
        "nonzero_bounded_acceptance": 0 < acceptance_rate <= 0.5,
    }
    pilot_gate["pass"] = bool(all(pilot_gate.values()))
    checkpoint = {
        "protocol": "residual_risk_terminal_scorer_v1",
        "seed": args.seed,
        "best_epoch": best[2]["epoch"],
        "input_dim": fit_x.shape[1],
        "hidden_dim": args.hidden_dim,
        "feature_mean": mean,
        "feature_std": std,
        "calibration": calibration,
        "deployment_risk_calibration": deployment_risk_calibration,
        "dagger_risk_calibration": dagger_risk_calibration,
        "state_dict": model.cpu().state_dict(),
    }
    torch.save(checkpoint, args.output_dir / "best.pt")
    result = {
        "schema_version": "residual-risk-terminal-scorer-v1",
        "seed": args.seed,
        "objective": "residual_terminal_utility_plus_false_edit_risk",
        "oof_disagreements": len(oof),
        "fit_rows": len(fit),
        "calibration_rows": len(calibration_rows),
        "hard_negative_rows": int((fit_delta_oof < 0).sum()),
        "oof_risk_positive_rows": int(fit_risk_oof.sum()),
        "dagger_risk_fit_rows": len(risk_fit_y),
        "dagger_risk_calibration_rows": len(risk_cal_y),
        "oof_risk_weight": oof_risk_weight,
        "best_epoch": best[2]["epoch"],
        "calibration": calibration,
        "validation_rows": len(common),
        "validation_disagreements": disagreements,
        "accepted_candidate_disagreements": accepted,
        "accepted_candidate_rate": acceptance_rate,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "hybrid": hybrid_summary,
        "hybrid_minus_baseline": deltas,
        "pilot_gate": pilot_gate,
        "test_assets_read": False,
        "promotion_status": "single_seed_pilot_only",
    }
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / ("PILOT_PASS" if pilot_gate["pass"] else "PILOT_REJECT")).write_text(
        json.dumps(pilot_gate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
