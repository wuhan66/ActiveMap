import json

import torch
from PIL import Image

from activemap.agent.vlm_sft import (
    VisualActionSFTCollator,
    encode_vlm_action_example,
    encode_vlm_prompt,
    load_vlm_sft_rows,
)


class FakeProcessor:
    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        prefix = [1, 2, 3, 4]
        ids = prefix if add_generation_prompt else prefix + [8, 9]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            "pixel_values": torch.ones((1, 3, 4, 4)),
        }


def _row(image_path, split="train"):
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": "state"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "{}"}]},
        ],
        "split": split,
    }


def test_visual_encoder_masks_prompt_and_keeps_pixels(tmp_path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (4, 4)).save(image_path)
    encoded = encode_vlm_action_example(_row(image_path), FakeProcessor(), max_length=16)
    assert encoded["labels"].tolist() == [-100, -100, -100, -100, 8, 9]
    assert encoded["pixel_values"].shape == (3, 4, 4)


def test_visual_prompt_encoder_excludes_assistant_tokens(tmp_path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (4, 4)).save(image_path)
    encoded = encode_vlm_prompt(_row(image_path), FakeProcessor(), max_length=16)
    assert encoded["input_ids"].tolist() == [1, 2, 3, 4]
    assert "labels" not in encoded


def test_visual_collator_pads_sequences_and_stacks_images():
    feature_a = {
        "input_ids": torch.tensor([1, 2]),
        "attention_mask": torch.tensor([1, 1]),
        "labels": torch.tensor([-100, 2]),
        "mm_token_type_ids": torch.tensor([1, 1]),
        "pixel_values": torch.ones((3, 2, 2)),
    }
    feature_b = {
        "input_ids": torch.tensor([3]),
        "attention_mask": torch.tensor([1]),
        "labels": torch.tensor([3]),
        "mm_token_type_ids": torch.tensor([1]),
        "pixel_values": torch.zeros((3, 2, 2)),
    }
    batch = VisualActionSFTCollator(pad_token_id=7)([feature_a, feature_b])
    assert batch["input_ids"].tolist() == [[1, 2], [3, 7]]
    assert batch["labels"].tolist() == [[-100, 2], [3, -100]]
    assert batch["mm_token_type_ids"].tolist() == [[1, 1], [1, 0]]
    assert batch["pixel_values"].shape == (2, 3, 2, 2)


def test_visual_collator_concatenates_qwen_flattened_image_patches():
    feature_a = {
        "input_ids": torch.tensor([1, 2]),
        "attention_mask": torch.tensor([1, 1]),
        "labels": torch.tensor([-100, 2]),
        "mm_token_type_ids": torch.tensor([0, 1]),
        "image_grid_thw": torch.tensor([1, 2, 2]),
        "pixel_values": torch.ones((4, 6)),
    }
    feature_b = {
        "input_ids": torch.tensor([3]),
        "attention_mask": torch.tensor([1]),
        "labels": torch.tensor([3]),
        "mm_token_type_ids": torch.tensor([1]),
        "image_grid_thw": torch.tensor([1, 1, 2]),
        "pixel_values": torch.zeros((2, 6)),
    }
    batch = VisualActionSFTCollator(pad_token_id=7)([feature_a, feature_b])
    assert batch["pixel_values"].shape == (6, 6)
    assert batch["image_grid_thw"].shape == (2, 3)
    assert batch["mm_token_type_ids"].shape == (2, 2)


def test_loader_rejects_test_split(tmp_path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (4, 4)).save(image_path)
    dataset_path = tmp_path / "sft.jsonl"
    dataset_path.write_text(json.dumps(_row(image_path, split="test")) + "\n")
    try:
        load_vlm_sft_rows(dataset_path)
    except ValueError as error:
        assert "train or validation" in str(error)
    else:
        raise AssertionError("test split must stay frozen")


def test_loader_resolves_portable_relative_image_path(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "image.jpg"
    Image.new("RGB", (4, 4)).save(image_path)
    row = _row(image_path)
    row["messages"][1]["content"][0]["image"] = "images/image.jpg"
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text(json.dumps(row) + "\n")
    loaded = load_vlm_sft_rows(dataset_path)
    assert loaded[0]["messages"][1]["content"][0]["image"] == str(image_path.resolve())
    encoded = encode_vlm_action_example(loaded[0], FakeProcessor(), max_length=16)
    assert int((encoded["labels"] != -100).sum()) == 2
