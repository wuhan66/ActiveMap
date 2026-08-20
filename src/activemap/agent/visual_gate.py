"""Frozen visual-state features and train-only calibration for sparse tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass
class NumpyMLPGate:
    """Portable one-hidden-layer gate exported from a frozen Torch training run."""

    mean: np.ndarray
    scale: np.ndarray
    weight1: np.ndarray
    bias1: np.ndarray
    weight2: np.ndarray
    bias2: float

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        normalized = (values - self.mean) / self.scale
        hidden = np.maximum(normalized @ self.weight1 + self.bias1, 0.0)
        logits = hidden @ self.weight2 + self.bias2
        positive = np.empty_like(logits, dtype=np.float64)
        nonnegative = logits >= 0.0
        positive[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
        exponential = np.exp(logits[~nonnegative])
        positive[~nonnegative] = exponential / (1.0 + exponential)
        return np.column_stack([1.0 - positive, positive])


def unique_pre_tool_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one PRE_TOOL state per example and reject inconsistent duplicates."""

    result: list[dict[str, Any]] = []
    seen: dict[str, bool] = {}
    for row in rows:
        if row.get("stage") != "PRE_TOOL":
            continue
        example_id = str(row.get("example_id", ""))
        if not example_id:
            raise ValueError("PRE_TOOL row is missing example_id")
        label = row.get("oracle_use_tool")
        if not isinstance(label, bool):
            raise ValueError(f"PRE_TOOL row has invalid oracle_use_tool: {example_id}")
        if example_id in seen:
            if seen[example_id] != label:
                raise ValueError(f"duplicate PRE_TOOL labels disagree: {example_id}")
            continue
        seen[example_id] = label
        result.append(row)
    if not result:
        raise ValueError("dataset contains no PRE_TOOL rows")
    return result


def pool_last_prompt_state(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the final attended prompt token from a right-padded hidden-state batch."""

    if hidden.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("hidden states and attention mask must have ranks 3 and 2")
    if hidden.shape[:2] != attention_mask.shape:
        raise ValueError("hidden-state sequence does not match attention mask")
    lengths = attention_mask.to(dtype=torch.long).sum(dim=1)
    if torch.any(lengths <= 0):
        raise ValueError("attention mask contains an empty prompt")
    indices = (lengths - 1).clamp_max(hidden.shape[1] - 1)
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch, indices]


def pool_prompt_state(
    hidden: Tensor, attention_mask: Tensor, *, mode: str = "last_mean"
) -> Tensor:
    """Pool a prompt while retaining both decision context and distributed visual evidence."""

    last = pool_last_prompt_state(hidden, attention_mask)
    if mode == "last":
        return last
    if mode != "last_mean":
        raise ValueError(f"unsupported visual gate pooling mode: {mode}")
    weights = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    mean = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    return torch.cat([last, mean], dim=-1)


def binary_call_metrics(
    target: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, float]:
    """Return sparse-call metrics without requiring a learned-model dependency."""

    labels = np.asarray(target, dtype=np.int64)
    scores = np.asarray(probability, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("target and probability must be aligned vectors")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("target must be binary")
    if not np.all(np.isfinite(scores)):
        raise ValueError("probability must be finite")
    prediction = scores >= threshold
    positive = labels == 1
    true_positive = int(np.sum(prediction & positive))
    false_positive = int(np.sum(prediction & ~positive))
    false_negative = int(np.sum(~prediction & positive))
    negative_count = int(np.sum(~positive))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    beta_squared = 0.25
    return {
        "threshold": float(threshold),
        "call_rate": float(np.mean(prediction)),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "f0_5": (1.0 + beta_squared)
        * precision
        * recall
        / max(beta_squared * precision + recall, 1e-12),
        "target_call_rate": float(np.mean(positive)),
        "predicted_calls": int(np.sum(prediction)),
        "target_calls": int(np.sum(positive)),
        "true_calls": true_positive,
        "false_calls": false_positive,
        "false_call_rate": false_positive / max(negative_count, 1),
    }
