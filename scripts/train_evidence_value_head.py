#!/usr/bin/env python3
"""Train the structured Evidence Value Head on executable counterfactual states."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from activemap.agent.evidence_value_head import (
    SUPPORTED_FEATURE_SETS,
    EvidenceValueHead,
    EvidenceValueHeadConfig,
    candidate_feature_dim,
    executable_value_example,
    fit_evidence_value_normalizer,
    risk_adjusted_scores,
)
from activemap.training.data import load_selector_samples


def load_examples(
    path: Path,
    split: str,
    *,
    feature_set: str = "evidence-value",
) -> list[dict[str, Any]]:
    return [
        executable_value_example(sample, feature_set=feature_set)
        for sample in load_selector_samples(path, split=split)
    ]


def collate_examples(rows: list[dict[str, Any]]) -> dict[str, Tensor]:
    batch = len(rows)
    maximum = max(len(row["utility_gains"]) for row in rows)
    context = torch.from_numpy(np.stack([row["context"] for row in rows]))
    candidates = torch.zeros(batch, maximum, rows[0]["candidates"].shape[1])
    mask = torch.zeros(batch, maximum, dtype=torch.bool)
    names = ("utility_gains", "quality_gains", "beneficial", "unsafe", "missed")
    result = {name: torch.zeros(batch, maximum) for name in names}
    for index, row in enumerate(rows):
        count = len(row["utility_gains"])
        candidates[index, :count] = torch.from_numpy(row["candidates"])
        mask[index, :count] = True
        for name in names:
            result[name][index, :count] = torch.from_numpy(row[name])
    return {
        "context": context,
        "candidates": candidates,
        "mask": mask,
        "terminal_target": torch.as_tensor(
            [int(row["terminal_target"]) for row in rows], dtype=torch.long
        ),
        **result,
    }


def evidence_value_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    *,
    temperature: float = 0.1,
    utility_weight: float = 1.0,
    quality_weight: float = 0.5,
    listwise_weight: float = 1.0,
    pairwise_weight: float = 0.0,
    beneficial_weight: float = 0.5,
    unsafe_weight: float = 0.5,
    missed_weight: float = 0.5,
    positive_weight: float = 3.0,
    beneficial_positive_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    if pairwise_weight < 0:
        raise ValueError("pairwise loss weight must be non-negative")
    if positive_weight <= 0 or beneficial_positive_weight <= 0:
        raise ValueError("positive loss weights must be positive")
    mask = batch["mask"]
    count = mask.sum().clamp_min(1)
    positive = torch.where(batch["utility_gains"] > 0, positive_weight, 1.0)
    utility = (
        F.smooth_l1_loss(
            outputs["utility"], batch["utility_gains"], reduction="none"
        )
        * positive
        * mask
    ).sum() / (positive * mask).sum().clamp_min(1)
    quality = (
        F.smooth_l1_loss(
            outputs["quality"], batch["quality_gains"], reduction="none"
        )
        * mask
    ).sum() / count

    stop = torch.zeros(outputs["utility"].shape[0], 1, device=mask.device)
    predicted = torch.cat(
        [stop, outputs["utility"].masked_fill(~mask, -1e4)], dim=1
    )
    target = torch.cat(
        [stop, batch["utility_gains"].masked_fill(~mask, -1e4)], dim=1
    )
    target_distribution = F.softmax(target / temperature, dim=1)
    listwise = -(
        target_distribution * F.log_softmax(predicted / temperature, dim=1)
    ).sum(dim=1).mean()

    positive_pairs = (batch["utility_gains"] > 0.0) & mask
    nonpositive_pairs = (batch["utility_gains"] <= 0.0) & mask
    pair_mask = positive_pairs.unsqueeze(2) & nonpositive_pairs.unsqueeze(1)
    pairwise_differences = outputs["utility"].unsqueeze(2) - outputs[
        "utility"
    ].unsqueeze(1)
    pairwise = (
        F.softplus(-pairwise_differences) * pair_mask
    ).sum() / pair_mask.sum().clamp_min(1)

    def binary(
        name: str, target_name: str, *, positive_class_weight: float = 1.0
    ) -> Tensor:
        values = F.binary_cross_entropy_with_logits(
            outputs[name],
            batch[target_name],
            reduction="none",
            pos_weight=torch.as_tensor(
                positive_class_weight,
                dtype=outputs[name].dtype,
                device=outputs[name].device,
            ),
        )
        return (values * mask).sum() / count

    components = {
        "utility": utility,
        "quality": quality,
        "listwise": listwise,
        "pairwise": pairwise,
        "beneficial": binary(
            "beneficial_logit",
            "beneficial",
            positive_class_weight=beneficial_positive_weight,
        ),
        "unsafe": binary("unsafe_logit", "unsafe"),
        "missed": binary("missed_logit", "missed"),
    }
    total = (
        utility_weight * components["utility"]
        + quality_weight * components["quality"]
        + listwise_weight * components["listwise"]
        + pairwise_weight * components["pairwise"]
        + beneficial_weight * components["beneficial"]
        + unsafe_weight * components["unsafe"]
        + missed_weight * components["missed"]
    )
    return total, components


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
) -> list[np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
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
        for row_index, row in enumerate(rows):
            scores.append(
                values[row_index, : len(row["utility_gains"])].cpu().numpy()
            )
    return scores


def policy_metrics(
    examples: list[dict[str, Any]], scores: list[np.ndarray], margin: float
) -> dict[str, float]:
    calls = false_calls = target_calls = true_calls = exact = 0
    utility = stop = oracle = unsafe = missed = 0.0
    for row, values in zip(examples, scores, strict=True):
        oracle_index = int(np.argmax(row["utility_gains"]))
        target_call = float(row["utility_gains"][oracle_index]) > 0.0
        predicted_index = int(np.argmax(values))
        predicted_call = float(values[predicted_index]) > margin
        calls += int(predicted_call)
        target_calls += int(target_call)
        true_calls += int(predicted_call and target_call)
        false_calls += int(predicted_call and not target_call)
        exact += int(predicted_call and target_call and predicted_index == oracle_index)
        stop += row["stop_utility"]
        utility += (
            float(row["utilities"][predicted_index])
            if predicted_call
            else row["stop_utility"]
        )
        oracle += max(row["stop_utility"], float(row["utilities"][oracle_index]))
        unsafe += float(row["unsafe"][predicted_index]) if predicted_call else 0.0
        missed += float(row["missed"][predicted_index]) if predicted_call else 0.0
    states = len(examples)
    negatives = states - target_calls
    return {
        "states": float(states),
        "predicted_call_rate": calls / states,
        "target_call_rate": target_calls / states,
        "acquire_recall": true_calls / max(target_calls, 1),
        "false_call_rate": false_calls / max(negatives, 1),
        "exact_evidence_recall": exact / max(target_calls, 1),
        "utility_gain_over_stop_mean": (utility - stop) / states,
        "mean_regret": (oracle - utility) / states,
        "unsafe_call_rate": unsafe / max(calls, 1),
        "missed_call_rate": missed / max(calls, 1),
    }


def calibrate_margin(
    examples: list[dict[str, Any]],
    scores: list[np.ndarray],
    maximum_false_call_rate: float,
) -> tuple[float, dict[str, float]]:
    best = np.asarray([float(values.max()) for values in scores])
    target = np.asarray([float(row["utility_gains"].max()) > 0 for row in examples])
    negatives = best[~target]
    if not len(negatives):
        raise ValueError("margin calibration requires STOP states")
    quantiles = sorted(
        {1.0 - maximum_false_call_rate, 0.90, 0.95, 0.975, 0.99, 1.0}
    )
    margins = {0.0}
    margins.update(
        float(np.quantile(negatives, q, method="higher")) + 1e-7
        for q in quantiles
        if q >= 1.0 - maximum_false_call_rate
    )
    candidates = [
        (margin, policy_metrics(examples, scores, margin))
        for margin in sorted(margins)
    ]
    feasible = [
        item
        for item in candidates
        if item[1]["false_call_rate"] <= maximum_false_call_rate
    ]
    return max(
        feasible or candidates,
        key=lambda item: (
            item[1]["utility_gain_over_stop_mean"],
            -item[1]["mean_regret"],
            item[0],
        ),
    )


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
    parser.add_argument(
        "--feature-set",
        choices=SUPPORTED_FEATURE_SETS,
        default="evidence-value",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--utility-weight", type=float, default=1.0)
    parser.add_argument("--quality-weight", type=float, default=0.5)
    parser.add_argument("--listwise-weight", type=float, default=1.0)
    parser.add_argument("--beneficial-weight", type=float, default=0.5)
    parser.add_argument("--utility-positive-weight", type=float, default=3.0)
    parser.add_argument("--beneficial-positive-weight", type=float, default=1.0)
    parser.add_argument("--unsafe-loss-weight", type=float, default=0.5)
    parser.add_argument("--missed-loss-weight", type=float, default=0.5)
    parser.add_argument("--unsafe-penalty", type=float, default=0.10)
    parser.add_argument("--missed-penalty", type=float, default=0.05)
    parser.add_argument("--maximum-false-call-rate", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.utility_positive_weight <= 0 or args.beneficial_positive_weight <= 0:
        raise ValueError("positive loss weights must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train = load_examples(args.states_jsonl, "train", feature_set=args.feature_set)
    validation = load_examples(
        args.states_jsonl, "val", feature_set=args.feature_set
    )
    normalizer = fit_evidence_value_normalizer(train)
    for row in train:
        row["context"], row["candidates"] = normalizer.transform(
            row["context"], row["candidates"]
        )
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
    config = EvidenceValueHeadConfig(
        candidate_dim=candidate_feature_dim(args.feature_set),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
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
        totals: list[float] = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["context"], batch["candidates"])
            loss, _ = evidence_value_loss(
                outputs,
                batch,
                temperature=args.temperature,
                utility_weight=args.utility_weight,
                quality_weight=args.quality_weight,
                listwise_weight=args.listwise_weight,
                beneficial_weight=args.beneficial_weight,
                unsafe_weight=args.unsafe_loss_weight,
                missed_weight=args.missed_loss_weight,
                positive_weight=args.utility_positive_weight,
                beneficial_positive_weight=args.beneficial_positive_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals.append(float(loss.detach()))
        validation_scores = infer(
            model,
            validation,
            normalizer=normalizer,
            device=device,
            unsafe_penalty=args.unsafe_penalty,
            missed_penalty=args.missed_penalty,
        )
        margin, metrics = calibrate_margin(
            validation,
            validation_scores,
            args.maximum_false_call_rate,
        )
        score = metrics["utility_gain_over_stop_mean"]
        scheduler.step(-score)
        record = {
            "epoch": epoch,
            "loss": float(np.mean(totals)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "margin": margin,
            "val": metrics,
        }
        with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(json.dumps(record, separators=(",", ":")), flush=True)
        if score > best_score + 1e-8:
            best_score = score
            stale = 0
            torch.save(
                {
                    "protocol": "evidence_value_head_v1",
                    "state_dict": model.state_dict(),
                    "model_config": config.as_dict(),
                    "normalizer": normalizer.as_dict(),
                    "feature_set": args.feature_set,
                    "safety_margin": margin,
                    "unsafe_penalty": args.unsafe_penalty,
                    "missed_penalty": args.missed_penalty,
                    "loss_weights": {
                        "utility": args.utility_weight,
                        "quality": args.quality_weight,
                        "listwise": args.listwise_weight,
                        "beneficial": args.beneficial_weight,
                        "utility_positive": args.utility_positive_weight,
                        "beneficial_positive": args.beneficial_positive_weight,
                        "unsafe": args.unsafe_loss_weight,
                        "missed": args.missed_loss_weight,
                    },
                    "epoch": epoch,
                    "seed": args.seed,
                    "val_metrics": metrics,
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
        "schema_version": "evidence-value-head-training-v1",
        "best_epoch": checkpoint["epoch"],
        "feature_set": checkpoint.get("feature_set", "evidence-value"),
        "train_states": len(train),
        "val_states": len(validation),
        "safety_margin": checkpoint["safety_margin"],
        "val_metrics": checkpoint["val_metrics"],
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
