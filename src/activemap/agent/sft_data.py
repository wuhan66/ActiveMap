"""Assistant-only tokenization and batching for structured action SFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


def render_action_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Render system and user messages with a generation marker."""

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def encode_action_example(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
    prompt_prefix_tokens: int = 256,
) -> tuple[dict[str, list[int]], bool]:
    """Encode one chat while masking every non-assistant token from the loss."""

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("SFT row must contain system, user, and assistant messages")
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("SFT message roles must be system, user, assistant")
    prompt = render_action_prompt(tokenizer, messages[:2])
    response = str(messages[2].get("content", ""))
    eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response + eos, add_special_tokens=False)["input_ids"]
    if not response_ids:
        raise ValueError("assistant action produced no tokens")
    available_prompt = max_length - len(response_ids)
    if available_prompt <= 0:
        raise ValueError("max_length is too small for the assistant action")
    truncated = len(prompt_ids) > available_prompt
    if truncated:
        prefix = min(prompt_prefix_tokens, available_prompt)
        suffix = available_prompt - prefix
        prompt_ids = prompt_ids[:prefix] + (prompt_ids[-suffix:] if suffix else [])
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }, truncated


class ActionSFTDataset(Dataset[dict[str, Tensor]]):
    """Pre-tokenized structured-action examples with truncation accounting."""

    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, *, max_length: int) -> None:
        self.items: list[dict[str, Tensor]] = []
        self.truncated_count = 0
        self.max_observed_length = 0
        for row in rows:
            encoded, truncated = encode_action_example(
                row, tokenizer, max_length=max_length
            )
            self.truncated_count += int(truncated)
            self.max_observed_length = max(self.max_observed_length, len(encoded["input_ids"]))
            self.items.append(
                {key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()}
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self.items[index]


@dataclass
class ActionSFTCollator:
    """Right-pad model inputs while preserving the -100 label mask."""

    pad_token_id: int

    def __call__(self, features: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        max_length = max(int(feature["input_ids"].numel()) for feature in features)
        batch: dict[str, list[Tensor]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for feature in features:
            padding = max_length - int(feature["input_ids"].numel())
            batch["input_ids"].append(
                torch.nn.functional.pad(feature["input_ids"], (0, padding), value=self.pad_token_id)
            )
            batch["attention_mask"].append(
                torch.nn.functional.pad(feature["attention_mask"], (0, padding), value=0)
            )
            batch["labels"].append(
                torch.nn.functional.pad(feature["labels"], (0, padding), value=-100)
            )
        return {key: torch.stack(values) for key, values in batch.items()}
