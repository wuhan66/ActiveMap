"""Multimodal contextual policy optimization over executable catalog actions."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from activemap.agent.vlm_preference import FORBIDDEN_PROMPT_KEYS, nested_keys
from activemap.agent.vlm_sft import VisualActionSFTCollator, encode_vlm_action_example


def load_rl_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != "active-catalog-vlm-rl-state-v1":
                raise ValueError(f"unexpected RL schema at {path}:{line_number}")
            if row.get("split") not in {"train", "val"}:
                raise ValueError("RL rows only permit train or validation")
            if row.get("test_assets_read") is not False or row.get("model_visible_utility") is not False:
                raise ValueError("RL row violates hidden-utility/test-isolation contract")
            actions = row.get("actions")
            if not isinstance(actions, list) or len(actions) < 2:
                raise ValueError("RL row requires at least two executable actions")
            keys = [str(action["key"]) for action in actions]
            if len(keys) != len(set(keys)) or "STOP" not in keys:
                raise ValueError("RL action set is duplicate or lacks STOP")
            target = next(
                (action for action in actions if action["key"] == row.get("target_action_key")),
                None,
            )
            if target is None or float(target["utility"]) + 1e-8 < max(
                float(action["utility"]) for action in actions
            ):
                raise ValueError("RL target is not utility-optimal")
            if FORBIDDEN_PROMPT_KEYS.intersection(nested_keys(row.get("prompt"))):
                raise ValueError("RL prompt leaks hidden target metadata")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no RL rows in {path}")
    return rows


def audit_rl_splits(train: list[dict[str, Any]], val: list[dict[str, Any]]) -> dict[str, Any]:
    if any(row["split"] != "train" for row in train) or any(row["split"] != "val" for row in val):
        raise ValueError("RL train/validation split mismatch")
    train_tasks = {str(row["task_id"]) for row in train}
    val_tasks = {str(row["task_id"]) for row in val}
    overlap = train_tasks & val_tasks
    if overlap:
        raise ValueError(f"RL train/validation task overlap: {len(overlap)}")
    return {
        "train_states": len(train),
        "val_states": len(val),
        "train_tasks": len(train_tasks),
        "val_tasks": len(val_tasks),
        "task_overlap": 0,
        "test_assets_read": False,
    }


def sample_rl_training_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int | None,
    acquire_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministically rebalance target actions without changing validation."""
    if not 0.0 <= acquire_fraction <= 1.0:
        raise ValueError("acquire fraction must be in [0, 1]")
    if limit is not None and limit <= 0:
        raise ValueError("training sample limit must be positive")
    size = min(limit, len(rows)) if limit is not None else len(rows)
    if acquire_fraction == 0.0:
        return rows[:size]

    acquire = [row for row in rows if str(row["target_action_key"]).startswith("ACQUIRE:")]
    stop = [row for row in rows if row["target_action_key"] == "STOP"]
    if not acquire or not stop:
        raise ValueError("balanced RL sampling requires both ACQUIRE and STOP targets")
    rng = random.Random(seed)

    def draw(pool: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        if count <= len(pool):
            return rng.sample(pool, count)
        return [*rng.sample(pool, len(pool)), *rng.choices(pool, k=count - len(pool))]

    acquire_count = min(size, max(1, round(size * acquire_fraction)))
    selected = draw(acquire, acquire_count) + draw(stop, size - acquire_count)
    rng.shuffle(selected)
    return selected


def select_hard_actions(row: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    actions = list(row["actions"])
    if limit < 2:
        raise ValueError("action limit must be at least two")
    if len(actions) <= limit:
        return actions
    target = str(row["target_action_key"])
    selected_keys = {"STOP", target}
    ranked = sorted(actions, key=lambda action: (-float(action["utility"]), str(action["key"])))
    for action in ranked:
        if len(selected_keys) >= limit:
            break
        selected_keys.add(str(action["key"]))
    return [action for action in actions if str(action["key"]) in selected_keys]


class VisualRLStateDataset(Dataset[dict[str, Any]]):
    def __init__(
        self, rows: list[dict[str, Any]], processor: Any, *, max_length: int, action_limit: int
    ) -> None:
        self.rows = rows
        self.processor = processor
        self.max_length = max_length
        self.action_limit = action_limit

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        actions = select_hard_actions(row, self.action_limit)
        encoded = [
            encode_vlm_action_example(
                {"messages": [*row["prompt"], action["message"]], "split": row["split"]},
                self.processor,
                max_length=self.max_length,
            )
            for action in actions
        ]
        return {
            "encoded_actions": encoded,
            "utilities": torch.tensor([float(action["utility"]) for action in actions]),
            "stop_index": next(
                index for index, action in enumerate(actions) if action["key"] == "STOP"
            ),
            "target_index": next(
                index
                for index, action in enumerate(actions)
                if action["key"] == row["target_action_key"]
            ),
            "target_is_acquire": str(row["target_action_key"]).startswith("ACQUIRE:"),
        }


@dataclass
class VisualRLStateCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Tensor]:
        if len(features) != 1:
            raise ValueError("multimodal RL currently requires batch size one")
        feature = features[0]
        batch = VisualActionSFTCollator(self.pad_token_id)(feature["encoded_actions"])
        batch["rl__utilities"] = feature["utilities"]
        batch["rl__stop_index"] = torch.tensor(feature["stop_index"], dtype=torch.long)
        batch["rl__target_index"] = torch.tensor(feature["target_index"], dtype=torch.long)
        batch["rl__target_is_acquire"] = torch.tensor(
            feature["target_is_acquire"], dtype=torch.bool
        )
        return batch


def sequence_log_probabilities(
    logits: Tensor, labels: Tensor, *, normalize_by_length: bool = False
) -> Tensor:
    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    if bool((mask.sum(dim=1) == 0).any()):
        raise ValueError("RL action has no supervised tokens")

    # Action labels occupy only a short suffix. Restricting the vocabulary
    # normalization to those positions avoids materializing full-sequence
    # FP32 log-probabilities while preserving the exact objective.
    batch_indices, token_indices = mask.nonzero(as_tuple=True)
    active_logits = shifted_logits[batch_indices, token_indices].float()
    active_labels = shifted_labels[batch_indices, token_indices]
    active_logps = active_logits.gather(-1, active_labels.unsqueeze(-1)).squeeze(-1)
    active_logps = active_logps - torch.logsumexp(active_logits, dim=-1)
    sequence_logps = active_logps.new_zeros(logits.shape[0])
    sequence_logps = sequence_logps.scatter_add(0, batch_indices, active_logps)
    if normalize_by_length:
        sequence_logps = sequence_logps / mask.sum(dim=1).to(sequence_logps.dtype)
    return sequence_logps


def contextual_policy_loss(
    policy_logps: Tensor,
    reference_logps: Tensor,
    utilities: Tensor,
    *,
    kl_beta: float,
    entropy_weight: float,
    temperature: float,
    pairwise_weight: float = 0.0,
    pairwise_margin: float = 0.0,
    stop_index: int | None = None,
    target_index: int | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    if policy_logps.shape != reference_logps.shape or policy_logps.shape != utilities.shape:
        raise ValueError("policy, reference, and utility vectors must align")
    if (
        kl_beta < 0
        or entropy_weight < 0
        or temperature <= 0
        or pairwise_weight < 0
        or pairwise_margin < 0
    ):
        raise ValueError("invalid contextual policy optimization settings")
    log_policy = F.log_softmax(policy_logps / temperature, dim=0)
    log_reference = F.log_softmax(reference_logps / temperature, dim=0)
    probabilities = log_policy.exp()
    expected_utility = (probabilities * utilities.float()).sum()
    kl = (probabilities * (log_policy - log_reference)).sum()
    entropy = -(probabilities * log_policy).sum()
    pairwise = policy_logps.new_zeros(())
    if pairwise_weight > 0 and stop_index is not None and target_index is not None:
        if not 0 <= stop_index < len(policy_logps) or not 0 <= target_index < len(policy_logps):
            raise ValueError("pairwise action index is out of range")
        if target_index != stop_index:
            score_gap = policy_logps[target_index] - policy_logps[stop_index]
            pairwise = F.softplus(pairwise_margin - score_gap)
    loss = (
        -expected_utility
        + kl_beta * kl
        - entropy_weight * entropy
        + pairwise_weight * pairwise
    )
    return loss, {
        "expected_utility": expected_utility.detach(),
        "policy_kl": kl.detach(),
        "policy_entropy": entropy.detach(),
        "best_action_probability": probabilities[utilities.argmax()].detach(),
        "pairwise_loss": pairwise.detach(),
    }
