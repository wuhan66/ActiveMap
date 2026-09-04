#!/usr/bin/env python3
"""Train a train-internal top-k, set-aware Evidence Value reranker."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

from activemap.agent.evidence_value_head import (
    SUPPORTED_FEATURE_SETS,
    TopKSetEvidenceValueConfig,
    TopKSetEvidenceValueHead,
    candidate_feature_dim,
    executable_value_example,
    fit_evidence_value_normalizer,
    risk_adjusted_scores,
)
from activemap.features import ONLINE_OBSERVABLE_STATE_CONTRACT
from activemap.selector_records import SelectorSample
from activemap.training.data import load_selector_samples
from activemap.training.selector import split_fit_calibration_samples
from scripts.train_evidence_value_head import collate_examples, evidence_value_loss

TARGET_NAMES = ("utility_gains", "quality_gains", "beneficial", "unsafe", "missed")


def resolve_data_contract(samples: list[SelectorSample]) -> dict[str, str] | None:
    """Return a shared online contract without inventing one for legacy states."""

    contracts = [sample.metadata.get("online_state_contract") for sample in samples]
    if all(contract is None for contract in contracts):
        return None
    if any(contract != ONLINE_OBSERVABLE_STATE_CONTRACT for contract in contracts):
        raise ValueError("selector states have mixed or unsupported online data contracts")
    return dict(ONLINE_OBSERVABLE_STATE_CONTRACT)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_weight_grid(value: str) -> tuple[float, ...]:
    weights = tuple(sorted({float(item) for item in value.split(",") if item.strip()}))
    if not weights or any(item < 0.0 for item in weights):
        raise ValueError("beneficial score weights must be non-negative")
    return weights


def parse_probability_grid(value: str) -> tuple[float, ...]:
    thresholds = tuple(
        sorted({float(item) for item in value.split(",") if item.strip()})
    )
    if not thresholds or any(item < 0.0 or item >= 1.0 for item in thresholds):
        raise ValueError("beneficial gate thresholds must be in [0, 1)")
    return thresholds


def normalize_examples(
    rows: list[dict[str, Any]], normalizer: Any
) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        context, candidates = normalizer.transform(row["context"], row["candidates"])
        normalized.append({**row, "context": context, "candidates": candidates})
    return normalized


def shortlist_targets(
    batch: dict[str, Tensor], outputs: dict[str, Any]
) -> dict[str, Tensor]:
    indices = outputs["shortlist_indices"]
    return {
        "mask": outputs["shortlist_mask"],
        "terminal_target": batch["terminal_target"],
        **{name: torch.gather(batch[name], 1, indices) for name in TARGET_NAMES},
    }


def loss_for_outputs(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    args: argparse.Namespace,
) -> tuple[Tensor, dict[str, Tensor]]:
    return evidence_value_loss(
        outputs,
        batch,
        temperature=args.temperature,
        utility_weight=args.utility_weight,
        quality_weight=args.quality_weight,
        listwise_weight=args.listwise_weight,
        pairwise_weight=args.pairwise_weight,
        beneficial_weight=args.beneficial_loss_weight,
        unsafe_weight=args.unsafe_loss_weight,
        missed_weight=args.missed_loss_weight,
        positive_weight=args.utility_positive_weight,
        beneficial_positive_weight=args.beneficial_positive_weight,
    )


@torch.no_grad()
def infer_score_grid(
    model: TopKSetEvidenceValueHead,
    examples: list[dict[str, Any]],
    *,
    device: torch.device,
    beneficial_weights: tuple[float, ...],
    beneficial_gate_thresholds: tuple[float, ...],
    unsafe_penalty: float,
    missed_penalty: float,
    batch_size: int = 512,
) -> tuple[dict[tuple[float, float], list[np.ndarray]], float]:
    model.eval()
    grids: dict[tuple[float, float], list[np.ndarray]] = {
        (weight, threshold): []
        for weight in beneficial_weights
        for threshold in beneficial_gate_thresholds
    }
    target_states = 0
    shortlist_hits = 0
    for start in range(0, len(examples), batch_size):
        rows = examples[start : start + batch_size]
        batch = {
            key: value.to(device) for key, value in collate_examples(rows).items()
        }
        outputs = model(batch["context"], batch["candidates"], batch["mask"])
        shortlist_indices = outputs["shortlist_indices"]
        for row_index, row in enumerate(rows):
            oracle_index = int(np.argmax(row["utility_gains"]))
            target = float(row["utility_gains"][oracle_index]) > 0.0
            target_states += int(target)
            shortlist = shortlist_indices[row_index].cpu().numpy()
            shortlist_hits += int(target and oracle_index in shortlist)
        for weight in beneficial_weights:
            shortlist_scores = risk_adjusted_scores(
                outputs["reranker"],
                beneficial_weight=weight,
                unsafe_weight=unsafe_penalty,
                missed_weight=missed_penalty,
            )
            beneficial_probability = torch.sigmoid(
                outputs["reranker"]["beneficial_logit"]
            )
            for threshold in beneficial_gate_thresholds:
                gated_scores = shortlist_scores.masked_fill(
                    beneficial_probability < threshold, -1e4
                ).masked_fill(~outputs["shortlist_mask"], -1e4)
                full_scores = torch.full_like(outputs["proposer_scores"], -1e4)
                full_scores.scatter_(1, shortlist_indices, gated_scores)
                for row_index, row in enumerate(rows):
                    count = len(row["utility_gains"])
                    grids[(weight, threshold)].append(
                        full_scores[row_index, :count].cpu().numpy()
                    )
    return grids, shortlist_hits / max(target_states, 1)


def policy_point(
    *,
    threshold: float,
    state_count: int,
    call_count: int,
    target_count: int,
    true_calls: int,
    false_calls: int,
    harmful_calls: int,
    exact_calls: int,
    cumulative_gain: float,
    stop_utility_sum: float,
) -> dict[str, float]:
    return {
        "stop_margin": float(threshold),
        "calls": float(call_count),
        "acquire_rate": call_count / state_count,
        "acquire_recall": true_calls / max(target_count, 1),
        "false_call_rate": false_calls / state_count,
        "harmful_call_fraction": harmful_calls / max(call_count, 1),
        "exact_evidence_recall": exact_calls / max(target_count, 1),
        "utility_gain_over_stop_mean": cumulative_gain / state_count,
        "mean_chosen_utility": (stop_utility_sum + cumulative_gain) / state_count,
    }


def constraint_distance(
    point: dict[str, float],
    *,
    maximum_false_call_rate: float,
    maximum_harmful_call_fraction: float,
    minimum_acquire_recall: float,
) -> float:
    return (
        max(point["false_call_rate"] - maximum_false_call_rate, 0.0)
        / max(maximum_false_call_rate, 1e-12)
        + max(point["harmful_call_fraction"] - maximum_harmful_call_fraction, 0.0)
        / max(maximum_harmful_call_fraction, 1e-12)
        + max(minimum_acquire_recall - point["acquire_recall"], 0.0)
        / max(minimum_acquire_recall, 1e-12)
        + float(point["utility_gain_over_stop_mean"] <= 0.0)
    )


def scan_stop_frontier(
    examples: list[dict[str, Any]],
    scores: list[np.ndarray],
    *,
    maximum_false_call_rate: float,
    maximum_harmful_call_fraction: float,
    minimum_acquire_recall: float,
) -> dict[str, Any]:
    if len(examples) != len(scores) or not examples:
        raise ValueError("frontier examples and scores must be aligned and non-empty")
    selected_indices = np.asarray([int(np.argmax(values)) for values in scores])
    maximum_scores = np.asarray(
        [float(values[index]) for values, index in zip(scores, selected_indices, strict=True)]
    )
    selected_gains = np.asarray(
        [
            float(row["utility_gains"][index])
            for row, index in zip(examples, selected_indices, strict=True)
        ]
    )
    oracle_indices = np.asarray(
        [int(np.argmax(row["utility_gains"])) for row in examples]
    )
    target = np.asarray(
        [
            float(row["utility_gains"][index]) > 0.0
            for row, index in zip(examples, oracle_indices, strict=True)
        ]
    )
    exact = target & (selected_indices == oracle_indices)
    order = np.argsort(-maximum_scores, kind="stable")
    ordered_scores = maximum_scores[order]
    ordered_target = target[order]
    ordered_harmful = selected_gains[order] <= 0.0
    ordered_exact = exact[order]
    ordered_gains = selected_gains[order]
    target_count = int(target.sum())
    stop_utility_sum = float(sum(row["stop_utility"] for row in examples))
    frontier = [
        policy_point(
            threshold=float(np.nextafter(ordered_scores[0], np.inf)),
            state_count=len(examples),
            call_count=0,
            target_count=target_count,
            true_calls=0,
            false_calls=0,
            harmful_calls=0,
            exact_calls=0,
            cumulative_gain=0.0,
            stop_utility_sum=stop_utility_sum,
        )
    ]
    cumulative_target = np.cumsum(ordered_target)
    cumulative_false = np.cumsum(~ordered_target)
    cumulative_harmful = np.cumsum(ordered_harmful)
    cumulative_exact = np.cumsum(ordered_exact)
    cumulative_gain = np.cumsum(ordered_gains)
    group_ends = np.flatnonzero(
        np.r_[ordered_scores[:-1] != ordered_scores[1:], True]
    )
    for end in group_ends:
        frontier.append(
            policy_point(
                threshold=float(np.nextafter(ordered_scores[end], -np.inf)),
                state_count=len(examples),
                call_count=int(end + 1),
                target_count=target_count,
                true_calls=int(cumulative_target[end]),
                false_calls=int(cumulative_false[end]),
                harmful_calls=int(cumulative_harmful[end]),
                exact_calls=int(cumulative_exact[end]),
                cumulative_gain=float(cumulative_gain[end]),
                stop_utility_sum=stop_utility_sum,
            )
        )

    def feasible(point: dict[str, float]) -> bool:
        return (
            point["false_call_rate"] <= maximum_false_call_rate
            and point["harmful_call_fraction"] <= maximum_harmful_call_fraction
            and point["acquire_recall"] >= minimum_acquire_recall
            and point["utility_gain_over_stop_mean"] > 0.0
        )

    feasible_points = [point for point in frontier if feasible(point)]
    selected = (
        max(
            feasible_points,
            key=lambda point: (
                point["utility_gain_over_stop_mean"],
                point["acquire_recall"],
                -point["acquire_rate"],
            ),
        )
        if feasible_points
        else min(
            frontier,
            key=lambda point: (
                constraint_distance(
                    point,
                    maximum_false_call_rate=maximum_false_call_rate,
                    maximum_harmful_call_fraction=maximum_harmful_call_fraction,
                    minimum_acquire_recall=minimum_acquire_recall,
                ),
                -point["utility_gain_over_stop_mean"],
            ),
        )
    )
    return {
        "constraints_satisfied": bool(feasible_points),
        "feasible_point_count": len(feasible_points),
        "constraint_distance": constraint_distance(
            selected,
            maximum_false_call_rate=maximum_false_call_rate,
            maximum_harmful_call_fraction=maximum_harmful_call_fraction,
            minimum_acquire_recall=minimum_acquire_recall,
        ),
        "selected": selected,
        "frontier": frontier,
    }


def selection_key(value: dict[str, Any]) -> tuple[float, ...]:
    point = value["selected"]
    return (
        float(value["constraints_satisfied"]),
        -float(value["constraint_distance"]),
        float(point["utility_gain_over_stop_mean"]),
        float(point["acquire_recall"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("states_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--feature-set",
        choices=SUPPORTED_FEATURE_SETS,
        default="evidence-value",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument(
        "--calibration-group-key",
        choices=("source_episode", "aoi_id"),
        default="source_episode",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--base-loss-weight", type=float, default=0.5)
    parser.add_argument("--utility-weight", type=float, default=1.0)
    parser.add_argument("--quality-weight", type=float, default=0.5)
    parser.add_argument("--listwise-weight", type=float, default=2.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.0)
    parser.add_argument("--beneficial-loss-weight", type=float, default=1.0)
    parser.add_argument("--utility-positive-weight", type=float, default=16.0)
    parser.add_argument("--beneficial-positive-weight", type=float, default=16.0)
    parser.add_argument("--unsafe-loss-weight", type=float, default=0.5)
    parser.add_argument("--missed-loss-weight", type=float, default=0.5)
    parser.add_argument("--unsafe-penalty", type=float, default=0.10)
    parser.add_argument("--missed-penalty", type=float, default=0.05)
    parser.add_argument("--beneficial-score-grid", default="0,0.025,0.05,0.1")
    parser.add_argument("--beneficial-gate-grid", default="0,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--maximum-false-call-rate", type=float, default=0.02)
    parser.add_argument("--maximum-harmful-call-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-acquire-recall", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if not 0.0 < args.calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be in (0, 0.5)")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")
    beneficial_weights = parse_weight_grid(args.beneficial_score_grid)
    beneficial_gate_thresholds = parse_probability_grid(args.beneficial_gate_grid)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    samples = load_selector_samples(args.states_jsonl, split="train")
    if any(sample.metadata.get("test_assets_read") is True for sample in samples):
        raise ValueError("test-derived selector states are forbidden")
    data_contract = resolve_data_contract(samples)
    fit_samples, calibration_samples = split_fit_calibration_samples(
        samples,
        fraction=args.calibration_fraction,
        seed=args.seed,
        group_key=args.calibration_group_key,
    )
    fit_groups = {
        str(sample.metadata.get(args.calibration_group_key, sample.sample_id))
        for sample in fit_samples
    }
    calibration_groups = {
        str(sample.metadata.get(args.calibration_group_key, sample.sample_id))
        for sample in calibration_samples
    }
    if fit_groups & calibration_groups:
        raise RuntimeError("calibration groups overlap across fit and calibration")
    fit_raw = [
        executable_value_example(sample, feature_set=args.feature_set)
        for sample in fit_samples
    ]
    calibration_raw = [
        executable_value_example(sample, feature_set=args.feature_set)
        for sample in calibration_samples
    ]
    normalizer = fit_evidence_value_normalizer(fit_raw)
    fit = normalize_examples(fit_raw, normalizer)
    calibration = normalize_examples(calibration_raw, normalizer)
    positive_states = np.asarray(
        [float(row["utility_gains"].max()) > 0.0 for row in fit]
    )
    if positive_states.all() or not positive_states.any():
        raise ValueError("fit split requires both ACQUIRE and STOP states")
    positive_state_weight = (~positive_states).sum() / positive_states.sum()
    sampler = WeightedRandomSampler(
        torch.as_tensor(
            np.where(positive_states, positive_state_weight, 1.0),
            dtype=torch.double,
        ),
        num_samples=len(fit),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        fit,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_examples,
    )
    config = TopKSetEvidenceValueConfig(
        candidate_dim=candidate_feature_dim(args.feature_set),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        top_k=args.top_k,
        proposer_beneficial_weight=0.10,
        proposer_unsafe_penalty=args.unsafe_penalty,
        proposer_missed_penalty=args.missed_penalty,
    )
    model = TopKSetEvidenceValueHead(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )
    args.output_dir.mkdir(parents=True)
    best_key: tuple[float, ...] | None = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["context"], batch["candidates"], batch["mask"])
            proposer_loss, _ = loss_for_outputs(outputs["proposer"], batch, args)
            reranker_loss, _ = loss_for_outputs(
                outputs["reranker"], shortlist_targets(batch, outputs), args
            )
            loss = (
                proposer_loss
                if epoch <= args.warmup_epochs
                else args.base_loss_weight * proposer_loss + reranker_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals.append(float(loss.detach()))
        if epoch <= args.warmup_epochs:
            record = {
                "epoch": epoch,
                "phase": "proposer_warmup",
                "loss": float(np.mean(totals)),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            print(json.dumps(record, separators=(",", ":")), flush=True)
            continue

        score_grid, topk_oracle_recall = infer_score_grid(
            model,
            calibration,
            device=device,
            beneficial_weights=beneficial_weights,
            beneficial_gate_thresholds=beneficial_gate_thresholds,
            unsafe_penalty=args.unsafe_penalty,
            missed_penalty=args.missed_penalty,
        )
        candidates = {}
        for (weight, gate_threshold), scores in score_grid.items():
            result = scan_stop_frontier(
                calibration,
                scores,
                maximum_false_call_rate=args.maximum_false_call_rate,
                maximum_harmful_call_fraction=args.maximum_harmful_call_fraction,
                minimum_acquire_recall=args.minimum_acquire_recall,
            )
            result["beneficial_score_weight"] = weight
            result["beneficial_probability_threshold"] = gate_threshold
            candidates[(weight, gate_threshold)] = result
        selected = max(candidates.values(), key=selection_key)
        current_key = selection_key(selected)
        scheduler.step(
            -selected["selected"]["utility_gain_over_stop_mean"]
            + selected["constraint_distance"]
        )
        record = {
            "epoch": epoch,
            "phase": "joint",
            "loss": float(np.mean(totals)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "topk_oracle_recall": topk_oracle_recall,
            "beneficial_score_weight": selected["beneficial_score_weight"],
            "beneficial_probability_threshold": selected[
                "beneficial_probability_threshold"
            ],
            "constraints_satisfied": selected["constraints_satisfied"],
            "calibration": selected["selected"],
        }
        with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(json.dumps(record, separators=(",", ":")), flush=True)
        if best_key is None or current_key > best_key:
            best_key = current_key
            stale = 0
            torch.save(
                {
                    "protocol": "topk_set_evidence_reranker_v1",
                    "state_dict": model.state_dict(),
                    "model_config": config.as_dict(),
                    "normalizer": normalizer.as_dict(),
                    "feature_set": args.feature_set,
                    "safety_margin": selected["selected"]["stop_margin"],
                    "stop_margin": selected["selected"]["stop_margin"],
                    "data_contract": data_contract,
                    "beneficial_score_weight": selected["beneficial_score_weight"],
                    "beneficial_probability_threshold": selected[
                        "beneficial_probability_threshold"
                    ],
                    "unsafe_penalty": args.unsafe_penalty,
                    "missed_penalty": args.missed_penalty,
                    "epoch": epoch,
                    "seed": args.seed,
                    "calibration_metrics": selected["selected"],
                    "calibration_constraints_satisfied": selected[
                        "constraints_satisfied"
                    ],
                    "topk_oracle_recall": topk_oracle_recall,
                    "pairwise_weight": args.pairwise_weight,
                    "utility_positive_weight": args.utility_positive_weight,
                    "beneficial_positive_weight": args.beneficial_positive_weight,
                    "states_sha256": sha256(args.states_jsonl),
                    "calibration_group_key": args.calibration_group_key,
                    "formal_validation_used_for_training_or_calibration": False,
                    "test_assets_read": False,
                },
                args.output_dir / "best.pt",
            )
            (args.output_dir / "best_calibration_frontier.json").write_text(
                json.dumps(selected, indent=2) + "\n", encoding="utf-8"
            )
        else:
            stale += 1
        if stale >= args.patience:
            break

    checkpoint = torch.load(
        args.output_dir / "best.pt", map_location="cpu", weights_only=False
    )
    summary = {
        "schema_version": "topk-set-evidence-reranker-training-v1",
        "states": str(args.states_jsonl.resolve()),
        "states_sha256": sha256(args.states_jsonl),
        "seed": args.seed,
        "top_k": args.top_k,
        "feature_set": checkpoint.get("feature_set", "evidence-value"),
        "data_contract": checkpoint.get("data_contract"),
        "best_epoch": checkpoint["epoch"],
        "fit_states": len(fit),
        "calibration_states": len(calibration),
        "calibration_group_key": args.calibration_group_key,
        "fit_group_count": len(fit_groups),
        "calibration_group_count": len(calibration_groups),
        "calibration_group_overlap": 0,
        "fit_source_episodes": len({row["source_episode"] for row in fit}),
        "calibration_source_episodes": len(
            {row["source_episode"] for row in calibration}
        ),
        "topk_oracle_recall": checkpoint["topk_oracle_recall"],
        "pairwise_weight": checkpoint.get("pairwise_weight", 0.0),
        "utility_positive_weight": checkpoint.get("utility_positive_weight"),
        "beneficial_positive_weight": checkpoint.get(
            "beneficial_positive_weight"
        ),
        "beneficial_score_weight": checkpoint["beneficial_score_weight"],
        "beneficial_probability_threshold": checkpoint[
            "beneficial_probability_threshold"
        ],
        "safety_margin": checkpoint["safety_margin"],
        "calibration_constraints_satisfied": checkpoint[
            "calibration_constraints_satisfied"
        ],
        "calibration_metrics": checkpoint["calibration_metrics"],
        "formal_validation_used_for_training_or_calibration": False,
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
