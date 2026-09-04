#!/usr/bin/env python3
"""Train an observable utility ranker for Active-Catalog candidates."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from activemap.agent.active_catalog_candidate_ranker import (
    RANKER_FEATURE_NAMES,
    CandidateUtilityRanker,
    CandidateUtilityRankerConfig,
    observable_ranker_features,
)
from activemap.agent.vlm_sft import load_vlm_sft_rows


def _message_text(message: dict[str, Any]) -> str:
    return next(part["text"] for part in message["content"] if part.get("type") == "text")


def _load_index(path: Path, split: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != split or row.get("model_visible") is not False:
                raise ValueError(f"evaluation index is not hidden {split} data")
            if row.get("test_assets_read") is not False:
                raise ValueError("evaluation index reports test access")
            example_id = str(row["example_id"])
            if example_id in result:
                raise ValueError(f"duplicate evaluation example: {example_id}")
            result[example_id] = row
    if not result:
        raise ValueError(f"empty evaluation index: {path}")
    return result


def load_examples(
    sft_path: Path, index_path: Path, split: str
) -> list[dict[str, Any]]:
    sft_rows = load_vlm_sft_rows(sft_path)
    if {str(row["split"]) for row in sft_rows} != {split}:
        raise ValueError(f"SFT file is not exclusively {split}")
    index = _load_index(index_path, split)
    if {str(row["example_id"]) for row in sft_rows} != set(index):
        raise ValueError("SFT rows and evaluation index do not have identical examples")
    examples = []
    for row in sft_rows:
        example_id = str(row["example_id"])
        evaluation = index[example_id]
        state = json.loads(_message_text(row["messages"][1]))
        candidates = list(state["candidate_evidence"])
        utility_by_id = {
            str(item["evidence_id"]): float(item["utility"])
            for item in evaluation["candidates"]
        }
        cost_by_id = {
            str(item["evidence_id"]): float(item["cost"])
            for item in evaluation["candidates"]
        }
        candidate_ids = [str(item["evidence_id"]) for item in candidates]
        if set(candidate_ids) != set(utility_by_id):
            raise ValueError(f"prompt/evaluation candidates disagree: {example_id}")
        stop = float(evaluation["stop_utility"])
        examples.append(
            {
                "example_id": example_id,
                "aoi_id": str(evaluation["aoi_id"]),
                "source_episode": str(evaluation["source_episode"]),
                "state": state,
                "candidate_ids": candidate_ids,
                "features": np.stack(
                    [observable_ranker_features(state, candidate) for candidate in candidates]
                ),
                "gains": np.asarray(
                    [utility_by_id[evidence_id] - stop for evidence_id in candidate_ids],
                    dtype=np.float32,
                ),
                "utilities": np.asarray(
                    [utility_by_id[evidence_id] for evidence_id in candidate_ids],
                    dtype=np.float32,
                ),
                "costs": np.asarray(
                    [cost_by_id[evidence_id] for evidence_id in candidate_ids],
                    dtype=np.float32,
                ),
                "stop_utility": stop,
            }
        )
    return examples


def load_online_examples(
    traces_path: Path, index_path: Path, split: str
) -> list[dict[str, Any]]:
    """Build current-policy residual targets from observable rollout states."""

    index_rows: dict[tuple[str, float, int], dict[str, Any]] = {}
    with index_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != split or row.get("model_visible") is not False:
                raise ValueError(f"online ranker requires hidden {split} index data")
            if row.get("test_assets_read") is not False:
                raise ValueError("online ranker index reports test access")
            key = (
                str(row["source_episode"]),
                float(row["budget"]),
                int(row["oracle_step"]),
            )
            if key in index_rows:
                raise ValueError(f"duplicate online evaluation key: {key}")
            index_rows[key] = row

    examples = []
    with traces_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            trace = json.loads(line)
            if trace.get("split") != split or trace.get("test_assets_read") is not False:
                raise ValueError(f"online ranker requires {split} rollout traces")
            match = re.search(r"__s(\d+)$", str(trace["sample_id"]))
            if match is None:
                raise ValueError(f"trace sample ID lacks oracle step: {trace['sample_id']}")
            key = (
                str(trace["source_episode"]),
                float(trace["budget"]),
                int(match.group(1)),
            )
            evaluation = index_rows.get(key)
            if evaluation is None:
                raise ValueError(f"trace has no evaluation row: {key}")
            events = list(trace.get("events", []))
            if not events or "observable_state" not in events[0]:
                raise ValueError(f"trace lacks initial observable state: {trace['sample_id']}")
            state = events[0]["observable_state"]
            candidates = list(state["candidate_evidence"])
            utility_by_id = {
                str(item["evidence_id"]): float(item["utility"])
                for item in evaluation["candidates"]
            }
            cost_by_id = {
                str(item["evidence_id"]): float(item["cost"])
                for item in evaluation["candidates"]
            }
            candidate_ids = [str(item["evidence_id"]) for item in candidates]
            if set(candidate_ids) != set(utility_by_id):
                raise ValueError(f"trace/evaluation candidates disagree: {trace['sample_id']}")
            stop = float(evaluation["stop_utility"])
            examples.append(
                {
                    "example_id": str(trace["sample_id"]),
                    "aoi_id": str(trace["aoi_id"]),
                    "source_episode": str(trace["source_episode"]),
                    "state": state,
                    "candidate_ids": candidate_ids,
                    "features": np.stack(
                        [observable_ranker_features(state, candidate) for candidate in candidates]
                    ),
                    "gains": np.asarray(
                        [utility_by_id[evidence_id] - stop for evidence_id in candidate_ids],
                        dtype=np.float32,
                    ),
                    "utilities": np.asarray(
                        [utility_by_id[evidence_id] for evidence_id in candidate_ids],
                        dtype=np.float32,
                    ),
                    "costs": np.asarray(
                        [cost_by_id[evidence_id] for evidence_id in candidate_ids],
                        dtype=np.float32,
                    ),
                    "stop_utility": stop,
                }
            )
    if not examples:
        raise ValueError(f"empty online rollout traces: {traces_path}")
    return examples


def attach_online_event_utility(
    examples: list[dict[str, Any]], traces_path: Path
) -> list[dict[str, Any]]:
    utility_by_id: dict[str, float] = {}
    with traces_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            trace = json.loads(line)
            events = list(trace.get("events", []))
            if not events or "predicted_acquire_utility" not in events[0]:
                raise ValueError(
                    f"trace lacks predicted acquire utility: {trace.get('sample_id')}"
                )
            utility_by_id[str(trace["sample_id"])] = float(
                events[0]["predicted_acquire_utility"]
            )
    result = []
    for row in examples:
        value = utility_by_id.get(str(row["example_id"]))
        if value is None:
            raise ValueError(f"missing online utility feature: {row['example_id']}")
        scalar = np.full((len(row["features"]), 1), value, dtype=np.float32)
        result.append(
            {
                **row,
                "features": np.concatenate([row["features"], scalar], axis=1),
            }
        )
    return result


def feature_statistics(examples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    features = np.concatenate([example["features"] for example in examples], axis=0)
    mean = features.mean(axis=0).astype(np.float32)
    standard_deviation = features.std(axis=0).astype(np.float32)
    standard_deviation[standard_deviation < 1e-6] = 1.0
    return mean, standard_deviation


def load_state_features(root: Path, split: str) -> dict[str, np.ndarray]:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("test_assets_read") is not False:
        raise ValueError("state features violate frozen-test isolation")
    features = np.load(root / "features.npy").astype(np.float32)
    records = [
        json.loads(line)
        for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(features) != len(records) or {row["split"] for row in records} != {split}:
        raise ValueError("state feature records are misaligned")
    ids = [str(row["example_id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("state feature example IDs are duplicated")
    return dict(zip(ids, features, strict=True))


def attach_state_features(
    examples: list[dict[str, Any]], state_features: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    by_id = {str(row["example_id"]): row for row in examples}
    missing = set(state_features) - set(by_id)
    if missing:
        raise ValueError(f"state features lack candidate examples: {len(missing)}")
    result = []
    for example_id, embedding in state_features.items():
        row = by_id[example_id]
        repeated = np.repeat(embedding.reshape(1, -1), len(row["features"]), axis=0)
        result.append(
            {
                **row,
                "features": np.concatenate([row["features"], repeated], axis=1).astype(
                    np.float32
                ),
            }
        )
    return result


def _collate(rows: list[dict[str, Any]]) -> dict[str, Tensor]:
    maximum = max(len(row["gains"]) for row in rows)
    feature_dim = rows[0]["features"].shape[1]
    features = torch.zeros(len(rows), maximum, feature_dim)
    gains = torch.zeros(len(rows), maximum)
    mask = torch.zeros(len(rows), maximum, dtype=torch.bool)
    for index, row in enumerate(rows):
        count = len(row["gains"])
        features[index, :count] = torch.from_numpy(row["features"])
        gains[index, :count] = torch.from_numpy(row["gains"])
        mask[index, :count] = True
    return {"features": features, "gains": gains, "mask": mask}


def ranker_loss(
    predicted: Tensor,
    gains: Tensor,
    mask: Tensor,
    *,
    temperature: float,
    regression_weight: float,
    listwise_weight: float,
    pairwise_weight: float,
    gate_weight: float,
    positive_weight: float,
) -> Tensor:
    regression_elements = F.smooth_l1_loss(predicted, gains, reduction="none")
    candidate_weights = torch.where(gains > 0.0, positive_weight, 1.0) * mask
    regression = (regression_elements * candidate_weights).sum() / candidate_weights.sum()
    stop = torch.zeros(predicted.shape[0], 1, device=predicted.device)
    predicted_actions = torch.cat([stop, predicted.masked_fill(~mask, -1e4)], dim=1)
    target_actions = torch.cat([stop, gains.masked_fill(~mask, -1e4)], dim=1)
    target_distribution = F.softmax(target_actions / temperature, dim=1)
    listwise = -(target_distribution * F.log_softmax(predicted_actions / temperature, dim=1))
    target_call = gains.masked_fill(~mask, -1e4).amax(dim=1) > 0.0
    state_weights = torch.where(target_call, positive_weight, 1.0)
    listwise_loss = (listwise.sum(dim=1) * state_weights).sum() / state_weights.sum()
    predicted_best = predicted.masked_fill(~mask, -1e4).amax(dim=1)
    oracle_index = gains.masked_fill(~mask, -1e4).argmax(dim=1)
    oracle_score = predicted.gather(1, oracle_index[:, None])
    oracle_gain = gains.gather(1, oracle_index[:, None])
    pair_mask = mask.clone()
    pair_mask.scatter_(1, oracle_index[:, None], False)
    utility_gap = (oracle_gain - gains).clamp_min(0.0)
    pair_weights = (utility_gap + 0.1) * pair_mask
    pairwise = (
        F.softplus(-(oracle_score - predicted) / temperature) * pair_weights
    ).sum() / pair_weights.sum().clamp_min(1.0)
    gate = F.binary_cross_entropy_with_logits(
        predicted_best / temperature,
        target_call.float(),
    )
    return (
        regression_weight * regression
        + listwise_weight * listwise_loss
        + pairwise_weight * pairwise
        + gate_weight * gate
    )


@torch.no_grad()
def infer_scores(
    model: CandidateUtilityRanker,
    examples: list[dict[str, Any]],
    *,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    device: torch.device,
) -> list[np.ndarray]:
    model.eval()
    lengths = [len(example["gains"]) for example in examples]
    normalized = np.concatenate(
        [(example["features"] - mean) / standard_deviation for example in examples],
        axis=0,
    ).astype(np.float32)
    chunks = []
    for start in range(0, len(normalized), 65536):
        batch = torch.from_numpy(normalized[start : start + 65536]).to(device)
        chunks.append(model(batch).cpu().numpy())
    flattened = np.concatenate(chunks)
    boundaries = np.cumsum(lengths)[:-1]
    return list(np.split(flattened, boundaries))


def policy_metrics(
    examples: list[dict[str, Any]], scores: list[np.ndarray], margin: float
) -> dict[str, float]:
    false_calls = calls = exact = candidate_exact = target_calls = true_calls = 0
    utility_sum = stop_sum = regret_sum = oracle_gate_utility_sum = 0.0
    for example, values in zip(examples, scores, strict=True):
        oracle_index = int(np.argmax(example["gains"]))
        target_call = float(example["gains"][oracle_index]) > 0.0
        target_calls += int(target_call)
        predicted_index = int(np.argmax(values))
        predicted_call = float(values[predicted_index]) > margin
        calls += int(predicted_call)
        true_calls += int(predicted_call and target_call)
        false_calls += int(predicted_call and not target_call)
        exact += int(predicted_call and target_call and predicted_index == oracle_index)
        candidate_exact += int(target_call and predicted_index == oracle_index)
        stop = float(example["stop_utility"])
        utility = float(example["utilities"][predicted_index]) if predicted_call else stop
        oracle = max(stop, float(example["utilities"][oracle_index]))
        utility_sum += utility
        oracle_gate_utility_sum += (
            float(example["utilities"][predicted_index]) if target_call else stop
        )
        stop_sum += stop
        regret_sum += oracle - utility
    states = len(examples)
    negatives = states - target_calls
    return {
        "states": float(states),
        "target_call_rate": target_calls / states,
        "predicted_call_rate": calls / states,
        "acquire_recall": true_calls / max(target_calls, 1),
        "false_call_rate": false_calls / max(negatives, 1),
        "exact_evidence_recall": exact / max(target_calls, 1),
        "candidate_exact_recall_given_oracle_gate": candidate_exact
        / max(target_calls, 1),
        "realized_utility_mean": utility_sum / states,
        "utility_gain_over_stop_mean": (utility_sum - stop_sum) / states,
        "oracle_gate_ranker_utility_mean": oracle_gate_utility_sum / states,
        "mean_regret": regret_sum / states,
    }


def calibrate_margin(
    examples: list[dict[str, Any]],
    scores: list[np.ndarray],
    margins: list[float],
    maximum_false_call_rate: float,
) -> tuple[float, dict[str, float]]:
    best_scores = np.asarray([float(np.max(values)) for values in scores])
    target_calls = np.asarray([float(np.max(row["gains"])) > 0.0 for row in examples])
    negative_scores = best_scores[~target_calls]
    if len(negative_scores) == 0:
        raise ValueError("margin calibration requires STOP states")
    minimum_safe_quantile = 1.0 - maximum_false_call_rate
    probabilities = sorted(
        {minimum_safe_quantile, 0.925, 0.95, 0.975, 0.99, 0.995, 1.0}
    )
    dynamic_margins = [
        float(np.quantile(negative_scores, probability, method="higher")) + 1e-7
        for probability in probabilities
        if probability >= minimum_safe_quantile
    ]
    unique_scores = np.unique(best_scores)
    if len(unique_scores) > 257:
        unique_scores = np.quantile(
            unique_scores, np.linspace(0.0, 1.0, 257), method="nearest"
        )
    empirical_margins = [float(score) - 1e-7 for score in unique_scores] + [
        float(score) + 1e-7 for score in unique_scores
    ]
    tested_margins = sorted(set(margins + dynamic_margins + empirical_margins))
    candidates = [
        (margin, policy_metrics(examples, scores, margin))
        for margin in tested_margins
    ]
    feasible = [
        item
        for item in candidates
        if item[1]["false_call_rate"] <= maximum_false_call_rate
    ]
    pool = feasible or candidates
    return max(
        pool,
        key=lambda item: (
            item[1]["utility_gain_over_stop_mean"],
            -item[1]["mean_regret"],
            item[0],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("train_evaluation_index", type=Path)
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("val_evaluation_index", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--input-format", choices=("sft", "online_trace"), default="sft"
    )
    parser.add_argument(
        "--online-event-feature",
        choices=("none", "predicted_acquire_utility"),
        default="none",
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--fusion-type", choices=("concat", "low_rank"), default="concat"
    )
    parser.add_argument("--fusion-dim", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--listwise-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.0)
    parser.add_argument("--gate-weight", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--positive-sampling-fraction", type=float, default=0.25)
    parser.add_argument("--maximum-false-call-rate", type=float, default=0.10)
    parser.add_argument("--margin-grid", default="0,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--train-state-features", type=Path)
    parser.add_argument("--val-state-features", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if min(args.epochs, args.batch_size, args.patience) <= 0:
        raise ValueError("epochs, batch size, and patience must be positive")
    if not 0.0 < args.positive_sampling_fraction < 1.0:
        raise ValueError("positive sampling fraction must be between zero and one")
    margins = [float(value) for value in args.margin_grid.split(",")]
    if not margins or min(margins) < 0.0:
        raise ValueError("margin grid must contain non-negative values")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    loader = load_online_examples if args.input_format == "online_trace" else load_examples
    train = loader(args.train_jsonl, args.train_evaluation_index, "train")
    validation = loader(args.val_jsonl, args.val_evaluation_index, "val")
    if args.online_event_feature != "none":
        if args.input_format != "online_trace":
            raise ValueError("online event features require online_trace input")
        train = attach_online_event_utility(train, args.train_jsonl)
        validation = attach_online_event_utility(validation, args.val_jsonl)
    if (args.train_state_features is None) != (args.val_state_features is None):
        raise ValueError("train and validation state features must be provided together")
    state_feature_dim = int(train[0]["features"].shape[1] - len(RANKER_FEATURE_NAMES))
    if args.train_state_features is not None and args.val_state_features is not None:
        train_state_features = load_state_features(args.train_state_features, "train")
        val_state_features = load_state_features(args.val_state_features, "val")
        embedding_dim = len(next(iter(train_state_features.values())))
        state_feature_dim += embedding_dim
        if len(next(iter(val_state_features.values()))) != embedding_dim:
            raise ValueError("train and validation state feature dimensions differ")
        train = attach_state_features(train, train_state_features)
        validation = attach_state_features(validation, val_state_features)
    mean, standard_deviation = feature_statistics(train)
    normalized_train = [
        {**row, "features": ((row["features"] - mean) / standard_deviation).astype(np.float32)}
        for row in train
    ]
    generator = torch.Generator().manual_seed(args.seed)
    positive_states = np.asarray(
        [float(np.max(row["gains"])) > 0.0 for row in normalized_train]
    )
    positive_count = int(positive_states.sum())
    negative_count = len(positive_states) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ranker training requires both ACQUIRE and STOP states")
    positive_sampling_weight = (
        args.positive_sampling_fraction
        / (1.0 - args.positive_sampling_fraction)
        * negative_count
        / positive_count
    )
    sampling_weights = torch.as_tensor(
        np.where(positive_states, positive_sampling_weight, 1.0),
        dtype=torch.double,
    )
    sampler = WeightedRandomSampler(
        sampling_weights,
        num_samples=len(normalized_train),
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        normalized_train,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=_collate,
        generator=generator,
    )
    config = CandidateUtilityRankerConfig(
        input_dim=train[0]["features"].shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        fusion_type=args.fusion_type,
        fusion_dim=args.fusion_dim,
    )
    model = CandidateUtilityRanker(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    args.output_dir.mkdir(parents=True)
    history_path = args.output_dir / "history.jsonl"
    best_score = -float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            features = batch["features"].to(device)
            gains = batch["gains"].to(device)
            mask = batch["mask"].to(device)
            loss = ranker_loss(
                model(features),
                gains,
                mask,
                temperature=args.temperature,
                regression_weight=args.regression_weight,
                listwise_weight=args.listwise_weight,
                pairwise_weight=args.pairwise_weight,
                gate_weight=args.gate_weight,
                positive_weight=args.positive_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        train_scores = infer_scores(
            model,
            train,
            mean=mean,
            standard_deviation=standard_deviation,
            device=device,
        )
        margin, train_metrics = calibrate_margin(
            train,
            train_scores,
            margins,
            args.maximum_false_call_rate,
        )
        validation_scores = infer_scores(
            model,
            validation,
            mean=mean,
            standard_deviation=standard_deviation,
            device=device,
        )
        validation_metrics = policy_metrics(validation, validation_scores, margin)
        score = validation_metrics["oracle_gate_ranker_utility_mean"]
        scheduler.step(-score)
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "safety_margin": margin,
            "train": train_metrics,
            "val": validation_metrics,
            "selection_score": score,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(json.dumps(record, separators=(",", ":")), flush=True)
        if score > best_score + 1e-8:
            best_score = score
            stale = 0
            torch.save(
                {
                    "protocol": "active_catalog_candidate_ranker_v4",
                    "input_format": args.input_format,
                    "online_event_feature": args.online_event_feature,
                    "state_dict": model.state_dict(),
                    "model_config": config.as_dict(),
                    "feature_mean": mean,
                    "feature_std": standard_deviation,
                    "safety_margin": margin,
                    "epoch": epoch,
                    "seed": args.seed,
                    "loss_weights": {
                        "regression": args.regression_weight,
                        "listwise": args.listwise_weight,
                        "pairwise": args.pairwise_weight,
                        "gate": args.gate_weight,
                        "positive": args.positive_weight,
                    },
                    "positive_sampling_fraction": args.positive_sampling_fraction,
                    "maximum_false_call_rate": args.maximum_false_call_rate,
                    "state_feature_dim": state_feature_dim,
                    "train_metrics": train_metrics,
                    "val_metrics": validation_metrics,
                    "test_assets_read": False,
                },
                args.output_dir / "best.pt",
            )
        else:
            stale += 1
        if stale >= args.patience:
            break
    checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    summary = {
        "schema_version": "active-catalog-candidate-ranker-training-v1",
        "best_epoch": checkpoint["epoch"],
        "safety_margin": checkpoint["safety_margin"],
        "train_metrics": checkpoint["train_metrics"],
        "val_metrics": checkpoint["val_metrics"],
        "train_states": len(train),
        "val_states": len(validation),
        "state_feature_dim": int(checkpoint.get("state_feature_dim", 0)),
        "input_format": checkpoint.get("input_format", "sft"),
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
