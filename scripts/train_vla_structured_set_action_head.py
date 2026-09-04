#!/usr/bin/env python3
"""Train a structured STOP-or-candidate action head on frozen VLM states.

This is deliberately a bounded diagnostic.  The Qwen3-VL encoder is frozen;
the trainable policy sees only public SELECT fields and candidate-wise encoder
states.  Hyperparameters and the conservative STOP margin are chosen from
task-grouped train-only out-of-fold predictions.  Validation is evaluated once
after that choice is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from scripts.train_sequential_set_utility_ranker import (
        _audit,
        action_metrics,
        action_traces,
        candidate_thresholds,
        load_task_examples,
        task_bootstrap,
    )
except ModuleNotFoundError as error:
    if error.name is None or not error.name.startswith("scripts"):
        raise
    from train_sequential_set_utility_ranker import (
        _audit,
        action_metrics,
        action_traces,
        candidate_thresholds,
        load_task_examples,
        task_bootstrap,
    )


PUBLIC_FEATURE_NAMES = (
    "draft_keep",
    "draft_delete",
    "draft_add",
    "draft_reshape",
    "belief_keep",
    "belief_delete",
    "belief_add",
    "belief_reshape",
    "belief_confidence",
    "geometry_delta_0",
    "geometry_delta_1",
    "geometry_delta_2",
    "geometry_delta_3",
    "geometry_delta_4",
    "geometry_delta_5",
    "geometry_delta_6",
    "geometry_delta_7",
    "belief_uncertainty",
    "draft_confidence",
    "budget_initial",
    "budget_spent",
    "budget_remaining",
    "false_edit_risk_limit",
    "raster_segment_available",
)
EDIT_VALUES = ("KEEP", "DELETE", "ADD", "RESHAPE")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _state_from_row(row: dict[str, Any]) -> dict[str, Any]:
    content = row["messages"][1]["content"]
    texts = [part["text"] for part in content if part.get("type") == "text"]
    if len(texts) != 1:
        raise ValueError("SELECT row must expose exactly one public state payload")
    state = json.loads(str(texts[0]))
    if state.get("controller_stage") != "SELECT":
        raise ValueError("manifest contains a non-SELECT state")
    return state


def public_state_features(state: dict[str, Any]) -> np.ndarray:
    """Encode only fields exposed in the SELECT user message."""

    draft = state.get("direct_draft")
    belief = state.get("belief")
    budget = state.get("budget")
    safety = state.get("safety")
    if not all(isinstance(value, dict) for value in (draft, belief, budget, safety)):
        raise ValueError("SELECT state lacks public structured fields")
    edit = str(draft.get("edit"))
    probabilities = [float(value) for value in belief.get("edit_probabilities", [])]
    geometry = [float(value) for value in belief.get("geometry_delta", [])]
    if edit not in EDIT_VALUES or len(probabilities) != 4 or len(geometry) != 8:
        raise ValueError("SELECT state has unsupported draft or belief shape")
    values = np.asarray(
        [float(edit == value) for value in EDIT_VALUES]
        + probabilities
        + [
            float(belief["confidence"]),
            *geometry,
            float(belief["uncertainty"]),
            float(draft["confidence"]),
            float(budget["initial"]),
            float(budget["spent"]),
            float(budget["remaining"]),
            float(safety["false_edit_risk_limit"]),
            float("RASTER_SEGMENT" in state.get("available_tools", [])),
        ],
        dtype=np.float32,
    )
    if values.shape != (len(PUBLIC_FEATURE_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError("invalid public SELECT state features")
    return values


def attach_public_features(tasks: list[dict[str, Any]], manifest: Path, split: str) -> list[dict[str, Any]]:
    rows = _load_jsonl(manifest)
    if {str(row.get("split")) for row in rows} != {split}:
        raise ValueError("manifest split does not match feature split")
    structured_by_key: dict[tuple[str, str], np.ndarray] = {}
    for row in rows:
        state = _state_from_row(row)
        evidence_id = state.get("evidence_id")
        if not isinstance(evidence_id, str):
            raise ValueError("SELECT state lacks public evidence ID")
        key = (str(row["task_id"]), evidence_id)
        if key in structured_by_key:
            raise ValueError("duplicate task/candidate state")
        structured_by_key[key] = public_state_features(state)
    result: list[dict[str, Any]] = []
    for task in tasks:
        structured = []
        for evidence_id in task["candidate_ids"]:
            value = structured_by_key.get((str(task["task_id"]), str(evidence_id)))
            if value is None:
                raise ValueError("feature task lacks matching public state")
            structured.append(value)
        visual = np.asarray(task["features"], dtype=np.float32)
        metadata = np.stack(structured).astype(np.float32)
        if visual.shape[0] != metadata.shape[0]:
            raise ValueError("candidate feature counts differ")
        result.append({**task, "features": np.concatenate((metadata, visual), axis=1)})
    return result


def feature_statistics(tasks: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([np.asarray(task["features"], dtype=np.float32) for task in tasks])
    mean = values.mean(axis=0).astype(np.float32)
    standard_deviation = values.std(axis=0).astype(np.float32)
    standard_deviation[standard_deviation < 1e-6] = 1.0
    return mean, standard_deviation


def normalize(tasks: list[dict[str, Any]], mean: np.ndarray, standard_deviation: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            **task,
            "features": ((task["features"] - mean) / standard_deviation).astype(np.float32),
        }
        for task in tasks
    ]


def _collate(tasks: list[dict[str, Any]]) -> dict[str, Tensor]:
    candidate_count = max(len(task["advantages"]) for task in tasks)
    feature_dim = int(tasks[0]["features"].shape[1])
    features = torch.zeros(len(tasks), candidate_count, feature_dim)
    advantages = torch.zeros(len(tasks), candidate_count)
    mask = torch.zeros(len(tasks), candidate_count, dtype=torch.bool)
    for index, task in enumerate(tasks):
        count = len(task["advantages"])
        features[index, :count] = torch.from_numpy(task["features"])
        advantages[index, :count] = torch.from_numpy(task["advantages"].astype(np.float32))
        mask[index, :count] = True
    return {"features": features, "advantages": advantages, "mask": mask}


@dataclass(frozen=True)
class StructuredSetActionHeadConfig:
    input_dim: int
    metadata_dim: int = len(PUBLIC_FEATURE_NAMES)
    projection_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.15
    use_set_context: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class StructuredSetActionHead(nn.Module):
    """Shared candidate scoring with task-level public STOP context."""

    def __init__(self, config: StructuredSetActionHeadConfig) -> None:
        super().__init__()
        if config.input_dim <= config.metadata_dim:
            raise ValueError("visual state dimension must be positive")
        self.config = config
        visual_dim = config.input_dim - config.metadata_dim
        self.metadata_projection = nn.Sequential(
            nn.Linear(config.metadata_dim, config.projection_dim),
            nn.LayerNorm(config.projection_dim),
            nn.GELU(),
        )
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, config.projection_dim),
            nn.LayerNorm(config.projection_dim),
            nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(3 * config.projection_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        candidate_input_dim = 3 * config.hidden_dim if config.use_set_context else config.hidden_dim
        self.candidate_value = nn.Sequential(
            nn.Linear(candidate_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        if config.use_set_context:
            self.stop_value: nn.Module | None = nn.Sequential(
                nn.Linear(2 * config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )
            self.stop_bias = None
        else:
            self.stop_value = None
            self.stop_bias = nn.Parameter(torch.zeros(1))

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        if features.ndim != 3 or mask.shape != features.shape[:2]:
            raise ValueError("invalid structured SET_SELECT batch")
        metadata = self.metadata_projection(features[..., : self.config.metadata_dim])
        visual = self.visual_projection(features[..., self.config.metadata_dim :])
        encoded = self.candidate_encoder(torch.cat((metadata, visual, metadata * visual), dim=-1))
        valid = mask.unsqueeze(-1).to(encoded.dtype)
        masked = encoded * valid
        count = valid.sum(dim=1).clamp_min(1.0)
        task_mean = masked.sum(dim=1) / count
        task_max = encoded.masked_fill(~mask.unsqueeze(-1), -1e4).amax(dim=1)
        context = torch.cat((task_mean, task_max), dim=-1)
        if self.config.use_set_context:
            candidate_context = context.unsqueeze(1).expand(-1, encoded.shape[1], -1)
            candidate_input = torch.cat((encoded, candidate_context), dim=-1)
            assert self.stop_value is not None
            stop = self.stop_value(context).squeeze(-1).unsqueeze(1)
        else:
            candidate_input = encoded
            assert self.stop_bias is not None
            stop = self.stop_bias.expand(encoded.shape[0], 1)
        candidate = self.candidate_value(candidate_input).squeeze(-1)
        return (candidate - stop).masked_fill(~mask, -1e4)


def action_loss(
    predicted: Tensor,
    advantages: Tensor,
    mask: Tensor,
    *,
    temperature: float,
    positive_weight: float,
    regression_weight: float,
    listwise_weight: float,
    gate_weight: float,
) -> Tensor:
    weights = torch.where(advantages > 0.0, positive_weight, 1.0) * mask
    regression = (F.smooth_l1_loss(predicted, advantages, reduction="none") * weights).sum() / weights.sum().clamp_min(1.0)
    action_logits = torch.cat((torch.zeros(predicted.shape[0], 1, device=predicted.device), predicted), dim=1)
    target_logits = torch.cat((torch.zeros_like(action_logits[:, :1]), advantages.masked_fill(~mask, -1e4)), dim=1)
    target_distribution = F.softmax(target_logits / temperature, dim=1)
    listwise = -(target_distribution * F.log_softmax(action_logits / temperature, dim=1)).sum(dim=1).mean()
    target_call = advantages.masked_fill(~mask, -1e4).amax(dim=1) > 0.0
    predicted_best = predicted.masked_fill(~mask, -1e4).amax(dim=1)
    gate = F.binary_cross_entropy_with_logits(predicted_best / temperature, target_call.float())
    return regression_weight * regression + listwise_weight * listwise + gate_weight * gate


def train_head(
    tasks: list[dict[str, Any]],
    config: StructuredSetActionHeadConfig,
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    positive_sampling_fraction: float,
    temperature: float,
    positive_weight: float,
    regression_weight: float,
    listwise_weight: float,
    gate_weight: float,
) -> StructuredSetActionHead:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    positive = np.asarray([float(np.max(task["advantages"])) > 0.0 for task in tasks])
    count_positive = int(positive.sum())
    count_negative = len(positive) - count_positive
    if count_positive == 0 or count_negative == 0:
        raise ValueError("SET_SELECT head needs both STOP and ACQUIRE train tasks")
    sampling_weight = positive_sampling_fraction / (1.0 - positive_sampling_fraction) * count_negative / count_positive
    sampler = WeightedRandomSampler(
        torch.as_tensor(np.where(positive, sampling_weight, 1.0), dtype=torch.double),
        num_samples=len(tasks),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(tasks, batch_size=batch_size, sampler=sampler, collate_fn=_collate)
    model = StructuredSetActionHead(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for batch in loader:
            prediction = model(batch["features"].to(device), batch["mask"].to(device))
            loss = action_loss(
                prediction,
                batch["advantages"].to(device),
                batch["mask"].to(device),
                temperature=temperature,
                positive_weight=positive_weight,
                regression_weight=regression_weight,
                listwise_weight=listwise_weight,
                gate_weight=gate_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    return model.eval()


@torch.no_grad()
def predict(model: StructuredSetActionHead, tasks: list[dict[str, Any]], device: torch.device) -> list[np.ndarray]:
    loader = DataLoader(tasks, batch_size=128, shuffle=False, collate_fn=_collate)
    result: list[np.ndarray] = []
    for batch in loader:
        values = model(batch["features"].to(device), batch["mask"].to(device)).cpu().numpy()
        mask = batch["mask"].numpy()
        result.extend([row[row_mask] for row, row_mask in zip(values, mask, strict=True)])
    return result


def choose_margin(tasks: list[dict[str, Any]], scores: list[np.ndarray], maximum_false_call_rate: float) -> tuple[float, dict[str, float]]:
    maxima = np.asarray([float(np.max(score)) for score in scores])
    labels = np.asarray([float(np.max(task["advantages"])) > 0.0 for task in tasks])
    negative = maxima[~labels]
    if len(negative) == 0:
        raise ValueError("STOP margin calibration requires negative tasks")
    thresholds = list(candidate_thresholds([np.asarray([value]) for value in maxima], (0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)))
    thresholds += [float(np.quantile(negative, q, method="higher")) + 1e-6 for q in (0.90, 0.925, 0.95, 0.975, 0.99, 1.0)]
    candidates = []
    for threshold in sorted(set(thresholds)):
        metrics = action_metrics(action_traces(tasks, scores, threshold))
        candidates.append((threshold, metrics))
    safe = [item for item in candidates if item[1]["false_call_rate"] <= maximum_false_call_rate]
    pool = safe or candidates
    return max(pool, key=lambda item: (item[1]["utility_mean"], item[1]["exact_candidate_recall"], -item[1]["false_call_rate"], -item[0]))


def write_raw_scores(path: Path, tasks: list[dict[str, Any]], scores: list[np.ndarray]) -> None:
    """Persist raw action values for an auditable train-only OOF fusion rule."""

    if len(tasks) != len(scores):
        raise ValueError("task/score count differs")
    with path.open("w", encoding="utf-8") as handle:
        for task, values in zip(tasks, scores, strict=True):
            if len(task["candidate_ids"]) != len(values):
                raise ValueError("candidate score dimension differs")
            handle.write(
                json.dumps(
                    {
                        "task_id": str(task["task_id"]),
                        "candidate_ids": [str(value) for value in task["candidate_ids"]],
                        "advantages": [float(value) for value in task["advantages"]],
                        "relative_action_scores": [float(value) for value in values],
                        "test_assets_read": False,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )


def grouped_oof_scores(
    tasks: list[dict[str, Any]],
    config: StructuredSetActionHeadConfig,
    *,
    device: torch.device,
    seed: int,
    folds: int,
    train_kwargs: dict[str, Any],
) -> list[np.ndarray]:
    labels = np.asarray([float(np.max(task["advantages"])) > 0.0 for task in tasks])
    positives = int(labels.sum())
    negatives = len(labels) - positives
    folds = min(folds, positives, negatives)
    if folds < 2:
        raise ValueError("insufficient class support for task-grouped OOF")
    # Class-stratified folds are applied to complete tasks, never individual candidates.
    rng = np.random.default_rng(seed)
    indices = np.arange(len(tasks))
    positive_indices = indices[labels]
    negative_indices = indices[~labels]
    rng.shuffle(positive_indices)
    rng.shuffle(negative_indices)
    fold_indices = [np.concatenate((positive_indices[index::folds], negative_indices[index::folds])) for index in range(folds)]
    scores: list[np.ndarray | None] = [None] * len(tasks)
    for fold, holdout in enumerate(fold_indices):
        holdout_set = {int(index) for index in holdout}
        train = [task for index, task in enumerate(tasks) if index not in holdout_set]
        validation = [tasks[int(index)] for index in holdout]
        model = train_head(train, config, device=device, seed=seed + fold + 1, **train_kwargs)
        for index, value in zip(holdout, predict(model, validation, device), strict=True):
            scores[int(index)] = value
    if any(value is None for value in scores):
        raise RuntimeError("OOF training left a task without scores")
    return [np.asarray(value) for value in scores if value is not None]


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def ablate_features(
    tasks: list[dict[str, Any]], *, metadata_dim: int, mode: str
) -> list[dict[str, Any]]:
    """Apply a named zero-information component ablation after normalization."""

    if mode == "full":
        return tasks
    result = []
    for task in tasks:
        features = np.asarray(task["features"], dtype=np.float32).copy()
        if mode == "visual_only":
            features[:, :metadata_dim] = 0.0
        elif mode == "structured_only":
            features[:, metadata_dim:] = 0.0
        else:
            raise ValueError(f"unsupported feature mode: {mode}")
        result.append({**task, "features": features})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_features", type=Path)
    parser.add_argument("val_features", type=Path)
    parser.add_argument("train_manifest", type=Path)
    parser.add_argument("val_manifest", type=Path)
    parser.add_argument("train_context_audit", type=Path)
    parser.add_argument("val_context_audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument(
        "--feature-mode", choices=("full", "visual_only", "structured_only"), default="full"
    )
    parser.add_argument("--disable-set-context", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--positive-sampling-fraction", type=float, default=0.25)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--listwise-weight", type=float, default=1.0)
    parser.add_argument("--gate-weight", type=float, default=1.0)
    parser.add_argument("--maximum-false-call-rate", type=float, default=0.10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not 0.0 < args.positive_sampling_fraction < 1.0:
        raise ValueError("positive sampling fraction must be in (0, 1)")
    if min(args.epochs, args.batch_size, args.folds) <= 0:
        raise ValueError("epochs, batch size, and folds must be positive")

    train_audit = _audit(args.train_context_audit, "train")
    val_audit = _audit(args.val_context_audit, "val")
    train_base, train_summary = load_task_examples(args.train_features, args.train_manifest, "train")
    val_base, val_summary = load_task_examples(args.val_features, args.val_manifest, "val")
    if train_summary.get("pooling") != val_summary.get("pooling"):
        raise ValueError("train and validation VLM pooling differ")
    train = attach_public_features(train_base, args.train_manifest, "train")
    validation = attach_public_features(val_base, args.val_manifest, "val")
    mean, standard_deviation = feature_statistics(train)
    train = normalize(train, mean, standard_deviation)
    validation = normalize(validation, mean, standard_deviation)
    train = ablate_features(train, metadata_dim=len(PUBLIC_FEATURE_NAMES), mode=args.feature_mode)
    validation = ablate_features(
        validation, metadata_dim=len(PUBLIC_FEATURE_NAMES), mode=args.feature_mode
    )
    config = StructuredSetActionHeadConfig(
        input_dim=int(train[0]["features"].shape[1]),
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_set_context=not args.disable_set_context,
    )
    device = torch.device(args.device)
    train_kwargs = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "positive_sampling_fraction": args.positive_sampling_fraction,
        "temperature": args.temperature,
        "positive_weight": args.positive_weight,
        "regression_weight": args.regression_weight,
        "listwise_weight": args.listwise_weight,
        "gate_weight": args.gate_weight,
    }
    oof_scores = grouped_oof_scores(
        train, config, device=device, seed=args.seed, folds=args.folds, train_kwargs=train_kwargs
    )
    margin, oof_metrics = choose_margin(train, oof_scores, args.maximum_false_call_rate)
    model = train_head(train, config, device=device, seed=args.seed, **train_kwargs)
    validation_scores = predict(model, validation, device)
    traces = action_traces(validation, validation_scores, margin)
    metrics = action_metrics(traces)
    bootstrap = task_bootstrap(traces, repetitions=args.bootstrap_repetitions, seed=args.seed)
    promotion = {
        "nonzero_calls": metrics["call_rate"] > 0.0,
        "bounded_call_rate": metrics["call_rate"] <= 0.50,
        "positive_utility": metrics["utility_mean"] > 0.0,
        "task_utility_ci_above_zero": bootstrap["utility_mean"]["ci95_low"] > 0.0,
        "false_call_rate_at_most_0_10": metrics["false_call_rate"] <= args.maximum_false_call_rate,
        "exact_candidate_recall_above_zero": metrics["exact_candidate_recall"] > 0.0,
    }
    args.output.mkdir(parents=True)
    write_raw_scores(args.output / "oof_raw_scores.jsonl", train, oof_scores)
    write_raw_scores(args.output / "validation_raw_scores.jsonl", validation, validation_scores)
    torch.save(
        {
            "schema_version": "structured-vla-set-action-head-v1",
            "model_config": config.as_dict(),
            "state_dict": model.cpu().state_dict(),
            "feature_mean": mean,
            "feature_std": standard_deviation,
            "safety_margin": margin,
            "public_feature_names": PUBLIC_FEATURE_NAMES,
        },
        args.output / "action_head.pt",
    )
    with (args.output / "validation_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "structured-vla-set-action-head-v1",
        "role": "diagnostic-only-explicit-multicandidate-set-select",
        "controller_interface": "STOP-or-one-public-evidence-id",
        "encoder": "frozen-Qwen3-VL-SELECT-state",
        "train_protocol": "task-grouped-OOF-margin-selection-train-only",
        "feature_mode": args.feature_mode,
        "set_context_enabled": not args.disable_set_context,
        "public_structured_features": list(PUBLIC_FEATURE_NAMES),
        "model_config": config.as_dict(),
        "train_task_count": len(train),
        "validation_task_count": len(validation),
        "oof": {"safety_margin": margin, "metrics": oof_metrics},
        "validation": metrics,
        "validation_task_bootstrap": bootstrap,
        "promotion_gate": {**promotion, "passed": all(promotion.values())},
        "context_audits": {
            "train": train_audit["explicit_multicandidate_reformulation_task_rate"],
            "val": val_audit["explicit_multicandidate_reformulation_task_rate"],
        },
        "sources": {
            "train_features": _sha256(args.train_features / "summary.json"),
            "val_features": _sha256(args.val_features / "summary.json"),
            "train_manifest": _sha256(args.train_manifest),
            "val_manifest": _sha256(args.val_manifest),
            "train_context_audit": _sha256(args.train_context_audit / "summary.json"),
            "val_context_audit": _sha256(args.val_context_audit / "summary.json"),
        },
        "test_assets_read": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Structured VLA SET_SELECT Smoke",
        "",
        "Diagnostic-only. Threshold selection used train-only task-grouped OOF predictions.",
        "",
        "| Split | Utility | Macro-F1 | Call rate | False-call rate | Exact candidate recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| train OOF | {oof_metrics['utility_mean']:.6f} | {oof_metrics['macro_f1']:.6f} | {oof_metrics['call_rate']:.6f} | {oof_metrics['false_call_rate']:.6f} | {oof_metrics['exact_candidate_recall']:.6f} |",
        f"| validation | {metrics['utility_mean']:.6f} | {metrics['macro_f1']:.6f} | {metrics['call_rate']:.6f} | {metrics['false_call_rate']:.6f} | {metrics['exact_candidate_recall']:.6f} |",
        "",
        f"Validation utility 95% task-bootstrap CI: [{bootstrap['utility_mean']['ci95_low']:.6f}, {bootstrap['utility_mean']['ci95_high']:.6f}]",
        f"Promotion gate passed: {all(promotion.values())}",
    ]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
