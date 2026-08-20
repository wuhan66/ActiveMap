"""Multimodal assistant-only SFT data handling for the visual map controller."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


def load_vlm_sft_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load and validate one visual SFT split without opening test assets."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                raise ValueError("visual SFT row must contain system, user, assistant messages")
            if [message.get("role") for message in messages] != [
                "system",
                "user",
                "assistant",
            ]:
                raise ValueError("visual SFT roles must be system, user, assistant")
            if row.get("split") not in {"train", "val"}:
                raise ValueError("visual SFT only permits train or validation records")
            image = _image_reference(messages)
            if not image.is_absolute():
                resolved = (path.parent / image).resolve()
                for part in messages[1]["content"]:
                    if part.get("type") == "image":
                        part["image"] = str(resolved)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no visual SFT records in {path}")
    return rows


def _image_reference(messages: list[dict[str, Any]]) -> Path:
    content = messages[1].get("content")
    if not isinstance(content, list):
        raise ValueError("visual SFT user content must be a multimodal list")
    images = [part.get("image") for part in content if part.get("type") == "image"]
    if len(images) != 1 or not isinstance(images[0], str):
        raise ValueError("visual SFT user message must contain exactly one image path")
    return Path(images[0])


def _image_path(messages: list[dict[str, Any]]) -> Path:
    path = _image_reference(messages)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def materialize_vlm_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = copy.deepcopy(messages)
    path = _image_path(result)
    image = Image.open(path).convert("RGB")
    for part in result[1]["content"]:
        if part.get("type") == "image":
            part["image"] = image
    return result


def _chat_encode(processor: Any, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **kwargs,
    )


def encode_vlm_action_example(
    row: dict[str, Any], processor: Any, *, max_length: int
) -> dict[str, Tensor]:
    """Encode a multimodal chat and mask all tokens before the assistant action."""

    messages = materialize_vlm_messages(row["messages"])
    full = _chat_encode(processor, messages, add_generation_prompt=False)
    prefix = _chat_encode(processor, messages[:2], add_generation_prompt=True)
    input_ids = full["input_ids"].squeeze(0)
    prefix_ids = prefix["input_ids"].squeeze(0)
    if input_ids.ndim != 1 or prefix_ids.ndim != 1:
        raise ValueError("processor returned invalid input_ids dimensions")
    if input_ids.numel() > max_length:
        raise ValueError(
            f"encoded sequence has {input_ids.numel()} tokens, exceeding max_length={max_length}"
        )
    common = 0
    common_limit = min(int(input_ids.numel()), int(prefix_ids.numel()))
    while common < common_limit and input_ids[common].item() == prefix_ids[common].item():
        common += 1
    if common == 0 or common >= input_ids.numel():
        raise ValueError("could not isolate assistant tokens from the chat template")
    labels = input_ids.clone()
    labels[:common] = -100
    encoded: dict[str, Tensor] = {"labels": labels}
    for key, value in full.items():
        if isinstance(value, Tensor):
            encoded[key] = value.squeeze(0)
    return encoded


def encode_vlm_prompt(
    row: dict[str, Any], processor: Any, *, max_length: int
) -> dict[str, Tensor]:
    """Encode only the observable prompt for frozen visual-state feature extraction."""

    messages = materialize_vlm_messages(row["messages"][:2])
    encoded = _chat_encode(processor, messages, add_generation_prompt=True)
    input_ids = encoded["input_ids"].squeeze(0)
    if input_ids.ndim != 1:
        raise ValueError("processor returned invalid prompt input_ids dimensions")
    if input_ids.numel() > max_length:
        raise ValueError(
            f"encoded prompt has {input_ids.numel()} tokens, exceeding max_length={max_length}"
        )
    result: dict[str, Tensor] = {}
    for key, value in encoded.items():
        if isinstance(value, Tensor):
            result[key] = value.squeeze(0)
    return result


class VisualActionSFTDataset(Dataset[dict[str, Tensor]]):
    """Lazily process image-text action records to keep host memory bounded."""

    def __init__(self, rows: list[dict[str, Any]], processor: Any, *, max_length: int) -> None:
        self.rows = rows
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return encode_vlm_action_example(
            self.rows[index], self.processor, max_length=self.max_length
        )


class VisualPromptDataset(Dataset[dict[str, Tensor]]):
    """Lazily encode observable visual prompts without assistant targets."""

    def __init__(self, rows: list[dict[str, Any]], processor: Any, *, max_length: int) -> None:
        self.rows = rows
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return encode_vlm_prompt(self.rows[index], self.processor, max_length=self.max_length)


@dataclass
class VisualActionSFTCollator:
    """Pad token tensors and stack fixed-size visual tensors."""

    pad_token_id: int

    def __call__(self, features: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        if not features:
            raise ValueError("cannot collate an empty visual SFT batch")
        sequence_keys = {
            "input_ids",
            "attention_mask",
            "labels",
            "token_type_ids",
            "mm_token_type_ids",
        }
        maximum = max(int(feature["input_ids"].numel()) for feature in features)
        result: dict[str, Tensor] = {}
        keys = set.intersection(*(set(feature) for feature in features))
        for key in sorted(keys):
            values = []
            for feature in features:
                value = feature[key]
                if key in sequence_keys:
                    padding = maximum - int(value.numel())
                    pad_value = -100 if key == "labels" else 0
                    if key == "input_ids":
                        pad_value = self.pad_token_id
                    value = torch.nn.functional.pad(value, (0, padding), value=pad_value)
                values.append(value)
            if key in {"pixel_values", "pixel_values_videos"} and values[0].ndim == 2:
                result[key] = torch.cat(values, dim=0)
            else:
                result[key] = torch.stack(values)
        return result
