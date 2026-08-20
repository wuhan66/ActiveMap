"""Reproducible oracle-imitation training for evidence selectors."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from activemap.config import load_yaml
from activemap.features import AblationSpec
from activemap.nn.selector import EvidenceSelector, SelectorConfig
from activemap.training.contracts import validate_selector_data_contract
from activemap.training.data import (
    SelectorDataset,
    collate_selector_batch,
    fit_selector_feature_normalizer,
    load_selector_samples,
)
from activemap.training.monitoring import RunMonitor, save_checkpoint


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def ablation_from_dict(payload: dict[str, Any]) -> AblationSpec:
    return AblationSpec(
        name=str(payload.get("name", "full")),
        condition_on_hypothesis=bool(payload.get("condition_on_hypothesis", True)),
        drop_hypothesis_groups=tuple(payload.get("drop_hypothesis_groups", [])),
        drop_evidence_groups=tuple(payload.get("drop_evidence_groups", [])),
        drop_state_groups=tuple(payload.get("drop_state_groups", [])),
        false_edit_penalty=bool(payload.get("false_edit_penalty", True)),
        allow_stop=bool(payload.get("allow_stop", True)),
    )


def split_fit_calibration_samples(
    samples: list[Any],
    *,
    fraction: float,
    seed: int,
    group_key: str = "source_episode",
) -> tuple[list[Any], list[Any]]:
    """Create a deterministic train-only calibration holdout without group leakage."""

    if not 0.0 <= fraction < 1.0:
        raise ValueError("calibration_fraction must be in [0, 1)")
    if fraction == 0.0:
        return samples, samples
    groups: dict[str, list[Any]] = {}
    for sample in samples:
        group = str(sample.metadata.get(group_key, sample.sample_id))
        groups.setdefault(group, []).append(sample)
    if len(groups) < 2:
        raise ValueError("calibration split requires at least two distinct groups")
    group_names = sorted(groups)
    random.Random(seed).shuffle(group_names)
    calibration_group_count = min(
        max(int(round(len(group_names) * fraction)), 1), len(group_names) - 1
    )
    calibration_groups = set(group_names[:calibration_group_count])
    fit = [
        sample
        for sample in samples
        if str(sample.metadata.get(group_key, sample.sample_id))
        not in calibration_groups
    ]
    calibration = [
        sample
        for sample in samples
        if str(sample.metadata.get(group_key, sample.sample_id)) in calibration_groups
    ]
    return fit, calibration


def _move_batch(
    batch: dict[str, Tensor | list[str]], device: torch.device
) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _normalized_utilities(utilities: Tensor) -> tuple[Tensor, Tensor]:
    """Normalize each action set while keeping padded actions out of the loss."""

    finite = torch.isfinite(utilities)
    minimum = torch.where(finite, utilities, torch.inf).min(dim=-1, keepdim=True).values
    maximum = (
        torch.where(finite, utilities, -torch.inf).max(dim=-1, keepdim=True).values
    )
    scale = torch.clamp(maximum - minimum, min=1e-6)
    normalized = torch.where(
        finite, (utilities - minimum) / scale, torch.zeros_like(utilities)
    )
    return normalized, finite


def selector_loss_components(
    logits: Tensor,
    utilities: Tensor,
    targets: Tensor,
    *,
    regret_weight: float,
    listwise_weight: float,
    utility_temperature: float,
    stop_weight: float,
    include_stop: bool,
    acquire_weight: float = 1.0,
    imitation_weight: float = 1.0,
    utility_regression_weight: float = 0.0,
    utility_scale: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Oracle imitation with utility ranking, normalized regret, and STOP supervision."""

    if utility_temperature <= 0:
        raise ValueError("utility_temperature must be positive")
    if acquire_weight <= 0:
        raise ValueError("acquire_weight must be positive")
    if imitation_weight < 0 or utility_regression_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if utility_scale <= 0:
        raise ValueError("utility_scale must be positive")
    if include_stop:
        stop_index = logits.shape[1] - 1
        sample_weights = torch.where(
            targets == stop_index,
            torch.ones_like(targets, dtype=logits.dtype),
            torch.full_like(targets, acquire_weight, dtype=logits.dtype),
        )
    else:
        sample_weights = torch.ones_like(targets, dtype=logits.dtype)
    sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-12)

    imitation_per_sample = nn.functional.cross_entropy(
        logits, targets, reduction="none"
    )
    imitation = (imitation_per_sample * sample_weights).mean()
    normalized, finite = _normalized_utilities(utilities)

    oracle_logits = (normalized / utility_temperature).masked_fill(~finite, -torch.inf)
    oracle_distribution = torch.softmax(oracle_logits, dim=-1)
    log_probabilities = torch.log_softmax(logits, dim=-1)
    safe_log_probabilities = torch.where(
        finite, log_probabilities, torch.zeros_like(log_probabilities)
    )
    listwise_per_sample = -torch.sum(
        oracle_distribution * safe_log_probabilities, dim=-1
    )
    listwise = (listwise_per_sample * sample_weights).mean()

    probabilities = torch.softmax(logits, dim=-1)
    expected_utility = torch.sum(probabilities * normalized, dim=-1)
    oracle_utility = torch.max(normalized, dim=-1).values
    expected_regret = ((oracle_utility - expected_utility) * sample_weights).mean()

    if include_stop:
        stop_target = (targets == stop_index).to(logits.dtype)
        stop_per_sample = nn.functional.binary_cross_entropy_with_logits(
            logits[:, stop_index], stop_target, reduction="none"
        )
        stop = (stop_per_sample * sample_weights).mean()
    else:
        stop = logits.sum() * 0.0

    regression = nn.functional.smooth_l1_loss(
        logits[finite], utilities[finite] * utility_scale
    )

    total = (
        imitation_weight * imitation
        + listwise_weight * listwise
        + regret_weight * expected_regret
        + stop_weight * stop
        + utility_regression_weight * regression
    )
    return total, {
        "imitation": imitation.detach(),
        "listwise": listwise.detach(),
        "expected_regret": expected_regret.detach(),
        "stop": stop.detach(),
        "utility_regression": regression.detach(),
    }


def selector_loss(
    logits: Tensor, utilities: Tensor, targets: Tensor, weight: float
) -> Tensor:
    """Backward-compatible entry point using the research-default objective."""

    total, _ = selector_loss_components(
        logits,
        utilities,
        targets,
        regret_weight=weight,
        listwise_weight=0.5,
        utility_temperature=0.25,
        stop_weight=0.25,
        include_stop=True,
        imitation_weight=1.0,
        utility_regression_weight=0.0,
        utility_scale=1.0,
    )
    return total


def two_stage_selector_loss_components(
    evidence_logits: Tensor,
    gate_logits: Tensor,
    utilities: Tensor,
    targets: Tensor,
    *,
    regret_weight: float,
    listwise_weight: float,
    utility_temperature: float,
    acquire_weight: float,
    imitation_weight: float,
    utility_regression_weight: float,
    utility_scale: float,
    gate_utility_weight: float,
    utility_regression_target: str = "absolute",
    candidate_value_predictions: Tensor | None = None,
    candidate_value_sign_weight: float = 0.0,
    context_gate_loss_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Train terminal gating separately from ranking among useful candidates."""

    if utility_regression_target not in {"absolute", "stop_delta"}:
        raise ValueError("utility_regression_target must be absolute or stop_delta")
    if candidate_value_sign_weight < 0 or context_gate_loss_weight < 0:
        raise ValueError(
            "candidate value and context gate loss weights must be non-negative"
        )

    stop_index = utilities.shape[1] - 1
    candidate_utilities = utilities[:, :-1]
    finite = torch.isfinite(candidate_utilities)
    acquire_targets = targets != stop_index
    gate_targets = acquire_targets.to(evidence_logits.dtype)
    gate_weights = torch.where(
        acquire_targets,
        torch.full_like(gate_targets, acquire_weight),
        torch.ones_like(gate_targets),
    )
    gate_weights = gate_weights / gate_weights.mean().clamp_min(1e-12)
    gate_bce = (
        nn.functional.binary_cross_entropy_with_logits(
            gate_logits, gate_targets, reduction="none"
        )
        * gate_weights
    ).mean()
    best_candidate_utility = (
        candidate_utilities.masked_fill(~finite, -torch.inf).max(dim=-1).values
    )
    gate_margin_target = (best_candidate_utility - utilities[:, -1]) * utility_scale
    gate_utility = (
        nn.functional.smooth_l1_loss(gate_logits, gate_margin_target, reduction="none")
        * gate_weights
    ).mean()

    if acquire_targets.any():
        acquire_logits = evidence_logits[acquire_targets]
        acquire_utilities = candidate_utilities[acquire_targets]
        acquire_finite = finite[acquire_targets]
        acquire_indices = targets[acquire_targets]
        rank_imitation = nn.functional.cross_entropy(acquire_logits, acquire_indices)
        normalized, _ = _normalized_utilities(acquire_utilities)
        oracle_logits = (normalized / utility_temperature).masked_fill(
            ~acquire_finite, -torch.inf
        )
        oracle_distribution = torch.softmax(oracle_logits, dim=-1)
        log_probabilities = torch.log_softmax(acquire_logits, dim=-1)
        listwise = -torch.sum(
            oracle_distribution
            * torch.where(
                acquire_finite, log_probabilities, torch.zeros_like(log_probabilities)
            ),
            dim=-1,
        ).mean()
        probabilities = torch.softmax(acquire_logits, dim=-1)
        expected_utility = torch.sum(probabilities * normalized, dim=-1)
        expected_regret = (normalized.max(dim=-1).values - expected_utility).mean()
    else:
        zero = evidence_logits[finite].sum() * 0.0
        rank_imitation = zero
        listwise = zero
        expected_regret = zero
    regression_targets = (
        candidate_utilities - utilities[:, -1, None]
        if utility_regression_target == "stop_delta"
        else candidate_utilities
    )
    value_predictions = (
        candidate_value_predictions
        if candidate_value_predictions is not None
        else evidence_logits
    )
    utility_regression = nn.functional.smooth_l1_loss(
        value_predictions[finite], regression_targets[finite] * utility_scale
    )
    sign_targets = (candidate_utilities > utilities[:, -1, None]).to(
        value_predictions.dtype
    )
    candidate_value_sign = nn.functional.binary_cross_entropy_with_logits(
        value_predictions[finite], sign_targets[finite]
    )
    total = (
        context_gate_loss_weight * (gate_bce + gate_utility_weight * gate_utility)
        + imitation_weight * rank_imitation
        + listwise_weight * listwise
        + regret_weight * expected_regret
        + utility_regression_weight * utility_regression
        + candidate_value_sign_weight * candidate_value_sign
    )
    return total, {
        "gate_bce": gate_bce.detach(),
        "gate_utility": gate_utility.detach(),
        "rank_imitation": rank_imitation.detach(),
        "listwise": listwise.detach(),
        "expected_regret": expected_regret.detach(),
        "utility_regression": utility_regression.detach(),
        "candidate_value_sign": candidate_value_sign.detach(),
    }


def run_epoch(
    model: EvidenceSelector,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    regret_loss_weight: float,
    listwise_loss_weight: float,
    utility_temperature: float,
    stop_loss_weight: float,
    acquire_loss_weight: float,
    imitation_loss_weight: float,
    utility_regression_weight: float,
    utility_scale: float,
    gate_utility_weight: float,
    utility_regression_target: str,
    candidate_value_sign_weight: float,
    context_gate_loss_weight: float,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
    progress_desc: str | None = None,
    stop_margin: float = 0.0,
) -> dict[str, float]:
    model.train(optimizer is not None)
    metric_sums = {
        "loss": 0.0,
        "accuracy": 0.0,
        "regret": 0.0,
        "normalized_regret": 0.0,
        "chosen_utility": 0.0,
        "oracle_utility": 0.0,
    }
    sample_count = 0
    stop_true_positives = 0
    stop_false_positives = 0
    stop_false_negatives = 0
    exact_acquire_hits = 0
    acquire_calls = 0
    harmful_calls = 0
    component_values: dict[str, list[float]] = {}
    batches = tqdm(loader, desc=progress_desc, unit="batch", leave=False)
    for raw_batch in batches:
        batch = _move_batch(raw_batch, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        (
            rank_logits,
            candidate_value_predictions,
            gate_or_stop_logits,
        ) = model.forward_training_components(
            batch["evidence"], batch["hypothesis"], batch["state"], batch["mask"]
        )
        evidence_logits = model.candidate_decision_scores(
            rank_logits, candidate_value_predictions
        )
        terminal_gate_logits = model.terminal_gate_score(
            candidate_value_predictions, gate_or_stop_logits
        )
        if model.config.allow_stop:
            if model.config.decision_mode == "two_stage":
                stop_logits = evidence_logits.max(dim=-1).values - terminal_gate_logits
            else:
                stop_logits = gate_or_stop_logits
            logits = torch.cat([evidence_logits, stop_logits[:, None]], dim=1)
        else:
            logits = evidence_logits
        utilities = batch["utilities"]
        if not model.config.allow_stop:
            utilities = utilities[:, :-1]
        targets = torch.argmax(utilities, dim=-1)
        if model.config.decision_mode == "two_stage" and model.config.allow_stop:
            loss, components = two_stage_selector_loss_components(
                rank_logits,
                gate_or_stop_logits,
                utilities,
                targets,
                regret_weight=regret_loss_weight,
                listwise_weight=listwise_loss_weight,
                utility_temperature=utility_temperature,
                acquire_weight=acquire_loss_weight,
                imitation_weight=imitation_loss_weight,
                utility_regression_weight=utility_regression_weight,
                utility_scale=utility_scale,
                gate_utility_weight=gate_utility_weight,
                utility_regression_target=utility_regression_target,
                candidate_value_predictions=candidate_value_predictions,
                candidate_value_sign_weight=candidate_value_sign_weight,
                context_gate_loss_weight=context_gate_loss_weight,
            )
        else:
            loss, components = selector_loss_components(
                logits,
                utilities,
                targets,
                regret_weight=regret_loss_weight,
                listwise_weight=listwise_loss_weight,
                utility_temperature=utility_temperature,
                stop_weight=stop_loss_weight,
                include_stop=model.config.allow_stop,
                acquire_weight=acquire_loss_weight,
                imitation_weight=imitation_loss_weight,
                utility_regression_weight=utility_regression_weight,
                utility_scale=utility_scale,
            )
        if optimizer is not None:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        decision_logits = logits
        if model.config.allow_stop and stop_margin != 0.0:
            decision_logits = logits.clone()
            decision_logits[:, -1] += stop_margin
        predictions = torch.argmax(decision_logits, dim=-1)
        chosen_utility = torch.gather(utilities, 1, predictions[:, None]).squeeze(1)
        oracle_utility = torch.max(utilities, dim=-1).values
        normalized_utilities, _ = _normalized_utilities(utilities)
        chosen_normalized = torch.gather(
            normalized_utilities, 1, predictions[:, None]
        ).squeeze(1)
        oracle_normalized = torch.max(normalized_utilities, dim=-1).values
        batch_size = int(targets.shape[0])
        sample_count += batch_size
        metric_sums["loss"] += float(loss.detach().cpu()) * batch_size
        metric_sums["accuracy"] += float((predictions == targets).sum().detach().cpu())
        metric_sums["regret"] += float(
            (oracle_utility - chosen_utility).sum().detach().cpu()
        )
        metric_sums["normalized_regret"] += float(
            (oracle_normalized - chosen_normalized).sum().detach().cpu()
        )
        metric_sums["chosen_utility"] += float(chosen_utility.sum().detach().cpu())
        metric_sums["oracle_utility"] += float(oracle_utility.sum().detach().cpu())
        for name, value in components.items():
            component_values.setdefault(name, []).append(
                float(value.cpu()) * batch_size
            )
        if model.config.allow_stop:
            stop_index = logits.shape[1] - 1
            predicted_stop = predictions == stop_index
            target_stop = targets == stop_index
            stop_true_positives += int(
                (predicted_stop & target_stop).sum().detach().cpu()
            )
            stop_false_positives += int(
                (predicted_stop & ~target_stop).sum().detach().cpu()
            )
            stop_false_negatives += int(
                (~predicted_stop & target_stop).sum().detach().cpu()
            )
            exact_acquire_hits += int(
                ((predictions == targets) & ~target_stop).sum().detach().cpu()
            )
            acquire_calls += int((~predicted_stop).sum().detach().cpu())
            harmful_calls += int(
                ((~predicted_stop) & (chosen_utility < utilities[:, -1]))
                .sum()
                .detach()
                .cpu()
            )
        batches.set_postfix(
            loss=f"{float(loss.detach().cpu()):.4f}",
            regret=f"{float((oracle_normalized - chosen_normalized).mean().detach().cpu()):.3f}",
        )
    if sample_count == 0:
        raise ValueError("selector loader produced no samples")
    metrics = {
        "loss": metric_sums["loss"] / sample_count,
        "accuracy": metric_sums["accuracy"] / sample_count,
        "regret": metric_sums["regret"] / sample_count,
        "normalized_regret": metric_sums["normalized_regret"] / sample_count,
        "mean_chosen_utility": metric_sums["chosen_utility"] / sample_count,
        "mean_oracle_utility": metric_sums["oracle_utility"] / sample_count,
    }
    metrics.update(
        {
            f"loss_{name}": sum(values) / sample_count
            for name, values in component_values.items()
        }
    )
    if model.config.allow_stop:
        stop_true_negatives = sample_count - (
            stop_true_positives + stop_false_positives + stop_false_negatives
        )
        metrics["stop_accuracy"] = (
            stop_true_positives + stop_true_negatives
        ) / sample_count
        precision = stop_true_positives / max(
            stop_true_positives + stop_false_positives, 1
        )
        recall = stop_true_positives / max(
            stop_true_positives + stop_false_negatives, 1
        )
        metrics["stop_precision"] = precision
        metrics["stop_recall"] = recall
        metrics["stop_f1"] = 2.0 * precision * recall / max(precision + recall, 1e-12)
        acquire_precision = stop_true_negatives / max(
            stop_true_negatives + stop_false_negatives, 1
        )
        acquire_recall = stop_true_negatives / max(
            stop_true_negatives + stop_false_positives, 1
        )
        metrics["acquire_precision"] = acquire_precision
        metrics["acquire_recall"] = acquire_recall
        metrics["acquire_f1"] = (
            2.0
            * acquire_precision
            * acquire_recall
            / max(acquire_precision + acquire_recall, 1e-12)
        )
        metrics["balanced_terminal_accuracy"] = 0.5 * (recall + acquire_recall)
        metrics["target_acquire_rate"] = (
            stop_true_negatives + stop_false_positives
        ) / sample_count
        metrics["predicted_acquire_rate"] = (
            stop_true_negatives + stop_false_negatives
        ) / sample_count
        target_acquire_count = stop_true_negatives + stop_false_positives
        metrics["exact_acquire_recall"] = exact_acquire_hits / max(
            target_acquire_count, 1
        )
        metrics["false_call_rate"] = stop_false_negatives / sample_count
        metrics["harmful_call_fraction"] = harmful_calls / max(acquire_calls, 1)
    return metrics


@torch.no_grad()
def select_stop_margin(
    margin_values: np.ndarray,
    deltas: np.ndarray,
    stop_values: np.ndarray,
    target_acquire: np.ndarray,
    *,
    max_false_call_rate: float | None = None,
    max_harmful_call_fraction: float | None = None,
    min_acquire_recall: float = 0.0,
) -> dict[str, float]:
    """Maximize utility over threshold prefixes subject to train-only safety limits."""

    margin_values = np.asarray(margin_values, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    stop_values = np.asarray(stop_values, dtype=np.float64)
    target_acquire = np.asarray(target_acquire, dtype=bool)
    if not (
        margin_values.ndim
        == deltas.ndim
        == stop_values.ndim
        == target_acquire.ndim
        == 1
        and len(margin_values) == len(deltas) == len(stop_values) == len(target_acquire)
        and len(margin_values) > 0
    ):
        raise ValueError(
            "stop-margin calibration arrays must be aligned non-empty vectors"
        )
    order = np.argsort(-margin_values, kind="stable")
    sorted_margins = margin_values[order]
    cumulative_delta = np.cumsum(deltas[order])
    cumulative_false_calls = np.cumsum(~target_acquire[order])
    cumulative_harmful_calls = np.cumsum(deltas[order] < 0.0)
    cumulative_true_calls = np.cumsum(target_acquire[order])
    target_acquire_count = int(target_acquire.sum())
    candidate_counts = [0]
    candidate_totals = [float(stop_values.sum())]
    for index in range(len(order)):
        is_group_end = (
            index == len(order) - 1 or sorted_margins[index + 1] < sorted_margins[index]
        )
        if is_group_end:
            candidate_counts.append(index + 1)
            candidate_totals.append(float(stop_values.sum() + cumulative_delta[index]))
    candidates = []
    all_candidates = []
    for position, acquire_count in enumerate(candidate_counts):
        last = acquire_count - 1
        false_call_rate = (
            float(cumulative_false_calls[last]) / len(order) if acquire_count else 0.0
        )
        harmful_call_fraction = (
            float(cumulative_harmful_calls[last]) / acquire_count
            if acquire_count
            else 0.0
        )
        acquire_recall = (
            float(cumulative_true_calls[last]) / max(target_acquire_count, 1)
            if acquire_count
            else 0.0
        )
        feasible = (
            (max_false_call_rate is None or false_call_rate <= max_false_call_rate)
            and (
                max_harmful_call_fraction is None
                or harmful_call_fraction <= max_harmful_call_fraction
            )
            and acquire_recall >= min_acquire_recall
        )
        false_violation = (
            max(false_call_rate - max_false_call_rate, 0.0)
            if max_false_call_rate is not None
            else 0.0
        )
        harmful_violation = (
            max(harmful_call_fraction - max_harmful_call_fraction, 0.0)
            if max_harmful_call_fraction is not None
            else 0.0
        )
        recall_violation = max(min_acquire_recall - acquire_recall, 0.0)
        row = (
            position,
            false_call_rate,
            harmful_call_fraction,
            acquire_recall,
            false_violation,
            harmful_violation,
            recall_violation,
        )
        all_candidates.append(row)
        if feasible:
            candidates.append(row)
    constraints_satisfied = bool(candidates)
    if constraints_satisfied:
        selected = max(
            candidates,
            key=lambda row: (candidate_totals[row[0]], -candidate_counts[row[0]]),
        )
    else:
        selected = min(
            all_candidates,
            key=lambda row: (
                row[4] + row[5],
                row[6],
                -candidate_totals[row[0]],
                candidate_counts[row[0]],
            ),
        )
    best_position, false_call_rate, harmful_call_fraction, acquire_recall = selected[:4]
    acquire_count = candidate_counts[best_position]
    if acquire_count == 0:
        margin = float(sorted_margins[0] + 1.0)
    elif acquire_count == len(order):
        margin = float(sorted_margins[-1] - 1.0)
    else:
        margin = float(
            0.5 * (sorted_margins[acquire_count - 1] + sorted_margins[acquire_count])
        )
    return {
        "stop_margin": margin,
        "mean_utility": candidate_totals[best_position] / len(order),
        "acquire_rate": acquire_count / len(order),
        "sample_count": float(len(order)),
        "false_call_rate": false_call_rate,
        "harmful_call_fraction": harmful_call_fraction,
        "acquire_recall": acquire_recall,
        "constraints_satisfied": float(constraints_satisfied),
    }


@torch.no_grad()
def calibrate_stop_margin(
    model: EvidenceSelector,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    max_false_call_rate: float | None = None,
    max_harmful_call_fraction: float | None = None,
    min_acquire_recall: float = 0.0,
) -> dict[str, float]:
    """Choose a safety-constrained STOP margin on held-out training groups."""

    if not model.config.allow_stop:
        return {
            "stop_margin": 0.0,
            "mean_utility": 0.0,
            "acquire_rate": 1.0,
            "constraints_satisfied": 1.0,
        }
    model.eval()
    margins: list[np.ndarray] = []
    utility_deltas: list[np.ndarray] = []
    stop_utilities: list[np.ndarray] = []
    target_acquires: list[np.ndarray] = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        logits = model(
            batch["evidence"],
            batch["hypothesis"],
            batch["state"],
            batch["mask"],
        )
        evidence_logits = logits[:, :-1]
        best_evidence = torch.argmax(evidence_logits, dim=-1)
        best_scores = torch.gather(evidence_logits, 1, best_evidence[:, None]).squeeze(
            1
        )
        utilities = batch["utilities"]
        chosen = torch.gather(utilities[:, :-1], 1, best_evidence[:, None]).squeeze(1)
        stop = utilities[:, -1]
        margins.append((best_scores - logits[:, -1]).cpu().numpy())
        utility_deltas.append((chosen - stop).cpu().numpy())
        stop_utilities.append(stop.cpu().numpy())
        target_acquires.append(
            (utilities[:, :-1].max(dim=-1).values > stop).cpu().numpy()
        )

    return select_stop_margin(
        np.concatenate(margins),
        np.concatenate(utility_deltas),
        np.concatenate(stop_utilities),
        np.concatenate(target_acquires),
        max_false_call_rate=max_false_call_rate,
        max_harmful_call_fraction=max_harmful_call_fraction,
        min_acquire_recall=min_acquire_recall,
    )


def train_selector(
    config_path: Path, *, output_override: Path | None = None
) -> dict[str, Any]:
    config = load_yaml(config_path)
    seed = int(config.get("seed", 20260710))
    set_global_seed(seed)
    data_path = Path(config["data"]["samples"])
    if not data_path.is_absolute():
        data_path = (config_path.parent / data_path).resolve()
    output_dir = output_override or Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation = ablation_from_dict(config.get("ablation", {}))
    model_payload = config.get("model", {})
    model_config = SelectorConfig(
        hidden_dim=int(model_payload.get("hidden_dim", 128)),
        dropout=float(model_payload.get("dropout", 0.10)),
        condition_on_hypothesis=ablation.condition_on_hypothesis,
        allow_stop=ablation.allow_stop,
        decision_mode=str(model_payload.get("decision_mode", "joint")),
        candidate_value_head=bool(model_payload.get("candidate_value_head", False)),
        candidate_decision_mode=str(
            model_payload.get("candidate_decision_mode", "rank")
        ),
        terminal_gate_mode=str(model_payload.get("terminal_gate_mode", "context")),
    )
    model = EvidenceSelector(model_config)
    training_payload = config.get("training", {})
    device = resolve_device(str(training_payload.get("device", "auto")))
    model.to(device)

    train_samples = load_selector_samples(data_path, split="train")
    val_samples = load_selector_samples(data_path, split="val")
    data_contract_payload = config.get("data_contract")
    if data_contract_payload is not None and not isinstance(data_contract_payload, dict):
        raise ValueError("data_contract must be a mapping")
    data_contract = validate_selector_data_contract(
        train_samples + val_samples, data_contract_payload
    )
    batch_size = int(training_payload.get("batch_size", 32))
    calibration_fraction = float(training_payload.get("calibration_fraction", 0.0))
    calibration_group_key = str(
        training_payload.get("calibration_group_key", "source_episode")
    )
    fit_samples, calibration_samples = split_fit_calibration_samples(
        train_samples,
        fraction=calibration_fraction,
        seed=seed,
        group_key=calibration_group_key,
    )
    normalize_features = bool(training_payload.get("normalize_features", False))
    feature_normalizer = (
        fit_selector_feature_normalizer(fit_samples) if normalize_features else None
    )
    target_sampling_power = float(training_payload.get("target_sampling_power", 0.0))
    if not 0.0 <= target_sampling_power <= 1.0:
        raise ValueError("target_sampling_power must be between zero and one")
    target_is_stop = [
        sample.target_index(allow_stop=ablation.allow_stop) == len(sample.evidence_ids)
        for sample in fit_samples
    ]
    target_counts = {
        "ACQUIRE": sum(not value for value in target_is_stop),
        "STOP": sum(target_is_stop),
    }
    train_sampler = None
    if target_sampling_power > 0.0 and all(target_counts.values()):
        weights = [
            target_counts["STOP" if is_stop else "ACQUIRE"] ** -target_sampling_power
            for is_stop in target_is_stop
        ]
        train_sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    train_loader = DataLoader(
        SelectorDataset(fit_samples, ablation, feature_normalizer),
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(training_payload.get("num_workers", 0)),
        collate_fn=collate_selector_batch,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        SelectorDataset(val_samples, ablation, feature_normalizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(training_payload.get("num_workers", 0)),
        collate_fn=collate_selector_batch,
    )
    calibration_loader = DataLoader(
        SelectorDataset(calibration_samples, ablation, feature_normalizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(training_payload.get("num_workers", 0)),
        collate_fn=collate_selector_batch,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_payload.get("learning_rate", 3e-4)),
        weight_decay=float(training_payload.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_payload.get("scheduler_factor", 0.5)),
        patience=int(training_payload.get("scheduler_patience", 3)),
        min_lr=float(training_payload.get("min_learning_rate", 1e-6)),
    )
    epochs = int(training_payload.get("epochs", 20))
    patience = int(training_payload.get("patience", 5))
    regret_loss_weight = float(
        training_payload.get(
            "regret_loss_weight", training_payload.get("utility_loss_weight", 0.25)
        )
    )
    listwise_loss_weight = float(training_payload.get("listwise_loss_weight", 0.5))
    utility_temperature = float(training_payload.get("utility_temperature", 0.25))
    stop_loss_weight = float(training_payload.get("stop_loss_weight", 0.25))
    acquire_loss_weight = float(training_payload.get("acquire_loss_weight", 1.0))
    imitation_loss_weight = float(training_payload.get("imitation_loss_weight", 1.0))
    utility_regression_weight = float(
        training_payload.get("utility_regression_weight", 0.0)
    )
    utility_scale = float(training_payload.get("utility_scale", 1.0))
    gate_utility_weight = float(training_payload.get("gate_utility_weight", 0.5))
    candidate_value_sign_weight = float(
        training_payload.get("candidate_value_sign_weight", 0.0)
    )
    context_gate_loss_weight = float(
        training_payload.get("context_gate_loss_weight", 1.0)
    )
    if candidate_value_sign_weight < 0 or context_gate_loss_weight < 0:
        raise ValueError(
            "candidate value and context gate loss weights must be non-negative"
        )
    utility_regression_target = str(
        training_payload.get("utility_regression_target", "absolute")
    )
    if utility_regression_target not in {"absolute", "stop_delta"}:
        raise ValueError("utility_regression_target must be absolute or stop_delta")
    use_stop_margin_calibration = bool(
        training_payload.get("calibrate_stop_margin", False)
    )
    calibration_max_false_call_rate = training_payload.get(
        "calibration_max_false_call_rate"
    )
    calibration_max_harmful_call_fraction = training_payload.get(
        "calibration_max_harmful_call_fraction"
    )
    calibration_min_acquire_recall = float(
        training_payload.get("calibration_min_acquire_recall", 0.0)
    )
    calibration_max_false_call_rate = (
        float(calibration_max_false_call_rate)
        if calibration_max_false_call_rate is not None
        else None
    )
    calibration_max_harmful_call_fraction = (
        float(calibration_max_harmful_call_fraction)
        if calibration_max_harmful_call_fraction is not None
        else None
    )
    for name, value in {
        "calibration_max_false_call_rate": calibration_max_false_call_rate,
        "calibration_max_harmful_call_fraction": calibration_max_harmful_call_fraction,
        "calibration_min_acquire_recall": calibration_min_acquire_recall,
    }.items():
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    checkpoint_min_acquire_recall = float(
        training_payload.get("checkpoint_min_acquire_recall", 0.0)
    )
    checkpoint_min_mean_utility = float(
        training_payload.get("checkpoint_min_mean_utility", float("-inf"))
    )
    checkpoint_metric = str(
        training_payload.get("checkpoint_metric", "normalized_regret")
    )
    if checkpoint_metric not in {"normalized_regret", "regret", "mean_chosen_utility"}:
        raise ValueError(f"unsupported checkpoint_metric: {checkpoint_metric}")
    grad_clip = float(training_payload.get("grad_clip", 1.0))
    history: list[dict[str, Any]] = []
    best_regret = float("inf")
    best_normalized_regret = float("inf")
    best_checkpoint_score = float("inf")
    best_checkpoint_eligible = False
    stale_epochs = 0
    monitor = RunMonitor(output_dir, config.get("monitoring", {}))
    resume = bool(training_payload.get("resume", False))
    start_epoch = 1
    last_checkpoint = output_dir / "last.pt"
    if resume and last_checkpoint.is_file():
        checkpoint = torch.load(
            last_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_regret = float(checkpoint["best_regret"])
        best_normalized_regret = float(checkpoint["best_normalized_regret"])
        best_checkpoint_score = float(
            checkpoint.get("best_checkpoint_score", best_normalized_regret)
        )
        best_checkpoint_eligible = bool(
            checkpoint.get("best_checkpoint_eligible", False)
        )
        stale_epochs = int(checkpoint["stale_epochs"])
        history = list(checkpoint.get("history", []))
    elif not resume and monitor.history_path.exists():
        monitor.history_path.unlink()
    monitor.write_state(
        "running",
        epoch=start_epoch - 1,
        total_epochs=epochs,
        resumed=resume and start_epoch > 1,
    )
    stopped = False

    for epoch in range(start_epoch, epochs + 1):
        if monitor.wait_if_paused(epoch=epoch) or monitor.should_stop():
            stopped = True
            monitor.write_state("stopped", epoch=epoch - 1, total_epochs=epochs)
            break
        monitor.write_state("running", epoch=epoch, total_epochs=epochs, phase="train")
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            regret_loss_weight=regret_loss_weight,
            listwise_loss_weight=listwise_loss_weight,
            utility_temperature=utility_temperature,
            stop_loss_weight=stop_loss_weight,
            acquire_loss_weight=acquire_loss_weight,
            imitation_loss_weight=imitation_loss_weight,
            utility_regression_weight=utility_regression_weight,
            utility_scale=utility_scale,
            gate_utility_weight=gate_utility_weight,
            utility_regression_target=utility_regression_target,
            candidate_value_sign_weight=candidate_value_sign_weight,
            context_gate_loss_weight=context_gate_loss_weight,
            optimizer=optimizer,
            grad_clip=grad_clip,
            progress_desc=f"Epoch {epoch:03d}/{epochs:03d} train",
        )
        with torch.no_grad():
            calibration = (
                calibrate_stop_margin(
                    model,
                    calibration_loader,
                    device=device,
                    max_false_call_rate=calibration_max_false_call_rate,
                    max_harmful_call_fraction=calibration_max_harmful_call_fraction,
                    min_acquire_recall=calibration_min_acquire_recall,
                )
                if use_stop_margin_calibration
                else {"stop_margin": 0.0, "mean_utility": 0.0, "acquire_rate": 0.0}
            )
            monitor.write_state(
                "running", epoch=epoch, total_epochs=epochs, phase="validation"
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                device=device,
                regret_loss_weight=regret_loss_weight,
                listwise_loss_weight=listwise_loss_weight,
                utility_temperature=utility_temperature,
                stop_loss_weight=stop_loss_weight,
                acquire_loss_weight=acquire_loss_weight,
                imitation_loss_weight=imitation_loss_weight,
                utility_regression_weight=utility_regression_weight,
                utility_scale=utility_scale,
                gate_utility_weight=gate_utility_weight,
                utility_regression_target=utility_regression_target,
                candidate_value_sign_weight=candidate_value_sign_weight,
                context_gate_loss_weight=context_gate_loss_weight,
                optimizer=None,
                grad_clip=grad_clip,
                progress_desc=f"Epoch {epoch:03d}/{epochs:03d} validation",
                stop_margin=float(calibration["stop_margin"]),
            )
            val_metrics["stop_margin"] = float(calibration["stop_margin"])
            val_metrics["calibration_train_mean_utility"] = float(
                calibration["mean_utility"]
            )
            val_metrics["calibration_train_acquire_rate"] = float(
                calibration["acquire_rate"]
            )
            for name in (
                "false_call_rate",
                "harmful_call_fraction",
                "acquire_recall",
                "constraints_satisfied",
            ):
                if name in calibration:
                    val_metrics[f"calibration_train_{name}"] = float(calibration[name])
        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_metrics["normalized_regret"])
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "next_learning_rate": next_learning_rate,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        monitor.record_epoch(record)
        checkpoint_eligible = (
            val_metrics.get("acquire_recall", 1.0) >= checkpoint_min_acquire_recall
            and val_metrics["mean_chosen_utility"] >= checkpoint_min_mean_utility
            and bool(calibration.get("constraints_satisfied", 1.0))
        )
        checkpoint_score = (
            -val_metrics["mean_chosen_utility"]
            if checkpoint_metric == "mean_chosen_utility"
            else val_metrics[checkpoint_metric]
        )
        should_promote = (
            checkpoint_eligible
            and (
                not best_checkpoint_eligible or checkpoint_score < best_checkpoint_score
            )
        ) or (
            not checkpoint_eligible
            and not best_checkpoint_eligible
            and checkpoint_score < best_checkpoint_score
        )
        if should_promote:
            best_regret = val_metrics["regret"]
            best_normalized_regret = val_metrics["normalized_regret"]
            best_checkpoint_score = checkpoint_score
            best_checkpoint_eligible = checkpoint_eligible
            stale_epochs = 0
            checkpoint = {
                "state_dict": model.state_dict(),
                "model_config": model_config.as_dict(),
                "ablation": asdict(ablation),
                "seed": seed,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "checkpoint_eligible": checkpoint_eligible,
                "checkpoint_min_acquire_recall": checkpoint_min_acquire_recall,
                "checkpoint_min_mean_utility": checkpoint_min_mean_utility,
                "checkpoint_metric": checkpoint_metric,
                "checkpoint_score": checkpoint_score,
                "stop_margin": float(calibration["stop_margin"]),
                "data_path": str(data_path),
                "data_contract": data_contract,
                "feature_normalizer": (
                    feature_normalizer.as_dict() if feature_normalizer else None
                ),
            }
            save_checkpoint(output_dir / "best.pt", checkpoint)
        else:
            stale_epochs += 1
        save_checkpoint(
            last_checkpoint,
            {
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "model_config": model_config.as_dict(),
                "ablation": asdict(ablation),
                "seed": seed,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "best_regret": best_regret,
                "best_normalized_regret": best_normalized_regret,
                "best_checkpoint_eligible": best_checkpoint_eligible,
                "best_checkpoint_score": best_checkpoint_score,
                "stop_margin": float(calibration["stop_margin"]),
                "stale_epochs": stale_epochs,
                "history": history,
                "data_path": str(data_path),
                "data_contract": data_contract,
                "feature_normalizer": (
                    feature_normalizer.as_dict() if feature_normalizer else None
                ),
            },
        )
        monitor.write_state(
            "running",
            epoch=epoch,
            total_epochs=epochs,
            phase="idle",
            latest_metrics=val_metrics,
        )
        if monitor.should_stop():
            stopped = True
            monitor.write_state("stopped", epoch=epoch, total_epochs=epochs)
            break
        if stale_epochs >= patience:
            break

    summary = {
        "config": str(config_path.resolve()),
        "output_dir": str(output_dir),
        "device": str(device),
        "parameter_count": model.parameter_count(),
        "best_val_regret": best_regret,
        "best_val_normalized_regret": best_normalized_regret,
        "epochs_completed": len(history),
        "stopped_by_user": stopped,
        "target_counts": target_counts,
        "fit_sample_count": len(fit_samples),
        "calibration_sample_count": len(calibration_samples),
        "calibration_fraction": calibration_fraction,
        "calibration_group_key": calibration_group_key,
        "normalize_features": normalize_features,
        "feature_normalizer": (
            feature_normalizer.as_dict() if feature_normalizer else None
        ),
        "target_sampling_power": target_sampling_power,
        "acquire_loss_weight": acquire_loss_weight,
        "imitation_loss_weight": imitation_loss_weight,
        "utility_regression_weight": utility_regression_weight,
        "utility_scale": utility_scale,
        "gate_utility_weight": gate_utility_weight,
        "candidate_value_sign_weight": candidate_value_sign_weight,
        "context_gate_loss_weight": context_gate_loss_weight,
        "utility_regression_target": utility_regression_target,
        "calibrate_stop_margin": use_stop_margin_calibration,
        "calibration_constraints": {
            "max_false_call_rate": calibration_max_false_call_rate,
            "max_harmful_call_fraction": calibration_max_harmful_call_fraction,
            "min_acquire_recall": calibration_min_acquire_recall,
        },
        "best_checkpoint_eligible": best_checkpoint_eligible,
        "best_checkpoint_score": best_checkpoint_score,
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_min_acquire_recall": checkpoint_min_acquire_recall,
        "checkpoint_min_mean_utility": checkpoint_min_mean_utility,
        "data_contract": data_contract,
        "ablation": asdict(ablation),
        "history": history,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not stopped:
        monitor.write_state(
            "completed",
            epoch=history[-1]["epoch"] if history else start_epoch - 1,
            total_epochs=epochs,
            early_stopped=stale_epochs >= patience,
        )
    monitor.close()
    return summary
