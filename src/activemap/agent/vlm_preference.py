"""Multimodal preference datasets and DPO utilities for active evidence selection."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from activemap.agent.vlm_sft import VisualActionSFTCollator, encode_vlm_action_example

FORBIDDEN_PROMPT_KEYS = {
    "chosen_utility",
    "rejected_utility",
    "utility_margin",
    "target_selection",
    "target_evidence_id",
    "oracle_utilities",
    "gt_edit",
}


def load_vlm_preference_rows(
    path: Path, limit: int | None = None
) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != "active-catalog-vlm-preference-v1":
                raise ValueError(f"unexpected preference schema at {path}:{line_number}")
            if row.get("split") not in {"train", "val"}:
                raise ValueError("VLM preferences only permit train or validation")
            if row.get("test_assets_read") is not False:
                raise ValueError("VLM preference does not prove test isolation")
            if row.get("model_visible_utility") is not False:
                raise ValueError("utility metadata must remain model-hidden")
            if float(row["chosen_utility"]) <= float(row["rejected_utility"]):
                raise ValueError("preference utility margin must be positive")
            prompt = row.get("prompt")
            if not isinstance(prompt, list) or [item.get("role") for item in prompt] != [
                "system",
                "user",
            ]:
                raise ValueError("preference prompt must contain system and user messages")
            for field in ("chosen", "rejected"):
                message = row.get(field)
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    raise ValueError(f"preference {field} must be an assistant message")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no VLM preferences in {path}")
    return rows


def nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(nested_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(nested_keys(item) for item in value))
    if isinstance(value, str):
        try:
            return nested_keys(json.loads(value))
        except (json.JSONDecodeError, TypeError):
            return set()
    return set()


def audit_preference_splits(
    train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    summaries = {}
    tasks_by_split = {}
    for split, rows in (("train", train_rows), ("val", eval_rows)):
        if any(row.get("split") != split for row in rows):
            raise ValueError(f"preference {split} split mismatch")
        identities = [
            (str(row["example_id"]), str(row["rejected_action_key"])) for row in rows
        ]
        if len(set(identities)) != len(identities):
            raise ValueError(f"duplicate preference pairs in {split}")
        leaked = [
            row["example_id"]
            for row in rows
            if FORBIDDEN_PROMPT_KEYS.intersection(nested_keys(row["prompt"]))
        ]
        if leaked:
            raise ValueError(f"model-visible target metadata in {split}: {leaked[0]}")
        tasks = {str(row["task_id"]) for row in rows}
        tasks_by_split[split] = tasks
        families: dict[str, int] = {}
        for row in rows:
            family = str(row["preference_family"])
            families[family] = families.get(family, 0) + 1
        summaries[split] = {
            "pairs": len(rows),
            "states": len({str(row["example_id"]) for row in rows}),
            "tasks": len(tasks),
            "families": dict(sorted(families.items())),
            "model_visible_target_metadata": False,
        }
    overlap = tasks_by_split["train"] & tasks_by_split["val"]
    if overlap:
        raise ValueError(f"preference train/validation task overlap: {len(overlap)}")
    return {
        **summaries,
        "task_overlap": 0,
        "test_assets_read": False,
    }


def preference_family_weights(
    rows: list[dict[str, Any]], *, mode: str = "none", power: float = 1.0
) -> tuple[list[float] | None, dict[str, float]]:
    if mode == "none":
        return None, {}
    if mode != "inverse_frequency":
        raise ValueError(f"unsupported preference family weighting: {mode}")
    if not 0.0 <= power <= 1.0:
        raise ValueError("preference family balance power must be between zero and one")
    counts = Counter(str(row["preference_family"]) for row in rows)
    if not counts:
        raise ValueError("cannot weight empty preference rows")
    total = len(rows)
    raw = {family: count ** (-power) for family, count in counts.items()}
    normalization = total / sum(counts[family] * value for family, value in raw.items())
    by_family = {
        family: normalization * raw[family] for family in sorted(counts)
    }
    return [by_family[str(row["preference_family"])] for row in rows], by_family


def preference_safety_weights(
    rows: list[dict[str, Any]],
    *,
    safe_acquire_boost: float = 0.0,
    unsafe_stop_scale: float = 1.0,
) -> tuple[list[float] | None, dict[str, Any]]:
    """Weight safety-resolving acquisition and suppress unsafe STOP preferences."""
    if safe_acquire_boost < 0:
        raise ValueError("safe acquire boost must be nonnegative")
    if not 0 < unsafe_stop_scale <= 1:
        raise ValueError("unsafe STOP scale must be in (0, 1]")
    if safe_acquire_boost == 0 and unsafe_stop_scale == 1:
        return None, {}
    if not rows:
        raise ValueError("cannot safety-weight empty preference rows")

    raw_weights = []
    boosted = 0
    downweighted = 0
    for row in rows:
        try:
            chosen_false_edit = float(row["chosen_false_edit_rate"])
            rejected_false_edit = float(row["rejected_false_edit_rate"])
        except KeyError as error:
            raise ValueError(
                "safety weighting requires chosen/rejected false-edit rates"
            ) from error
        safety_gap = max(0.0, rejected_false_edit - chosen_false_edit)
        weight = 1.0 + safe_acquire_boost * safety_gap
        if safety_gap > 0:
            boosted += 1
        if str(row.get("chosen_action_key")) == "STOP" and chosen_false_edit > 0:
            weight *= unsafe_stop_scale
            downweighted += 1
        raw_weights.append(weight)

    normalization = len(raw_weights) / sum(raw_weights)
    weights = [weight * normalization for weight in raw_weights]
    return weights, {
        "safe_acquire_boost": safe_acquire_boost,
        "unsafe_stop_scale": unsafe_stop_scale,
        "boosted_pairs": boosted,
        "downweighted_unsafe_stop_pairs": downweighted,
        "minimum_weight": min(weights),
        "maximum_weight": max(weights),
        "mean_weight": sum(weights) / len(weights),
    }


def combine_preference_weights(
    *components: list[float] | None,
) -> list[float] | None:
    active = [weights for weights in components if weights is not None]
    if not active:
        return None
    length = len(active[0])
    if any(len(weights) != length for weights in active):
        raise ValueError("preference weight components must have identical lengths")
    combined = [1.0] * length
    for weights in active:
        for index, weight in enumerate(weights):
            if weight <= 0:
                raise ValueError("preference weight components must be positive")
            combined[index] *= weight
    normalization = length / sum(combined)
    return [weight * normalization for weight in combined]


class VisualPreferenceDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        processor: Any,
        *,
        max_length: int,
        loss_weights: list[float] | None = None,
    ) -> None:
        if loss_weights is not None:
            if len(loss_weights) != len(rows):
                raise ValueError("preference loss weights must match rows")
            if any(weight <= 0 for weight in loss_weights):
                raise ValueError("preference loss weights must be positive")
        self.rows = rows
        self.processor = processor
        self.max_length = max_length
        self.loss_weights = loss_weights

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, row: dict[str, Any], field: str) -> dict[str, Tensor]:
        example = {
            "messages": [*row["prompt"], row[field]],
            "split": row["split"],
        }
        return encode_vlm_action_example(
            example, self.processor, max_length=self.max_length
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        result = {
            "chosen": self._encode(row, "chosen"),
            "rejected": self._encode(row, "rejected"),
        }
        if self.loss_weights is not None:
            result["loss_weight"] = torch.tensor(
                self.loss_weights[index], dtype=torch.float32
            )
        return result


@dataclass
class VisualPreferenceCollator:
    pad_token_id: int

    def __call__(
        self, features: list[dict[str, Any]]
    ) -> dict[str, Tensor]:
        if not features:
            raise ValueError("cannot collate an empty preference batch")
        collator = VisualActionSFTCollator(self.pad_token_id)
        combined = collator(
            [feature["chosen"] for feature in features]
            + [feature["rejected"] for feature in features]
        )
        result = {f"pair__{key}": value for key, value in combined.items()}
        weighted = ["loss_weight" in feature for feature in features]
        if any(weighted):
            if not all(weighted):
                raise ValueError("cannot mix weighted and unweighted preferences")
            result["pair_loss_weight"] = torch.stack(
                [feature["loss_weight"] for feature in features]
            )
        return result


def split_preference_batch(batch: dict[str, Tensor], branch: str) -> dict[str, Tensor]:
    prefix = f"{branch}__"
    result = {key[len(prefix) :]: value for key, value in batch.items() if key.startswith(prefix)}
    if "input_ids" not in result or "labels" not in result:
        raise ValueError(f"preference batch lacks {branch} input IDs or labels")
    return result


def combined_preference_batch(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    prefix = "pair__"
    result = {key[len(prefix) :]: value for key, value in batch.items() if key.startswith(prefix)}
    if "input_ids" not in result or "labels" not in result:
        raise ValueError("combined preference batch lacks input IDs or labels")
    if result["labels"].shape[0] % 2:
        raise ValueError("combined preference batch must contain chosen/rejected pairs")
    return result


def sequence_log_probabilities(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels have incompatible shapes")
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(shifted_logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("preference branch has no supervised assistant tokens")
    return (token_logps * mask).sum(dim=1)


def dpo_loss(
    policy_chosen: Tensor,
    policy_rejected: Tensor,
    reference_chosen: Tensor,
    reference_rejected: Tensor,
    *,
    beta: float,
) -> Tensor:
    if beta <= 0:
        raise ValueError("DPO beta must be positive")
    policy_ratio = policy_chosen - policy_rejected
    reference_ratio = reference_chosen - reference_rejected
    return -F.logsigmoid(beta * (policy_ratio - reference_ratio))


def preference_weighted_loss(
    losses: Tensor, weights: Tensor | None = None
) -> Tensor:
    if weights is None:
        return losses.mean()
    if losses.shape != weights.shape:
        raise ValueError("preference losses and weights must have identical shapes")
    if bool((weights <= 0).any()):
        raise ValueError("preference loss weights must be positive")
    # Family weights are normalized to mean one over the full train set. Do not
    # renormalize within a microbatch: batch size one would cancel weighting.
    return (losses * weights.to(losses.device)).mean()
