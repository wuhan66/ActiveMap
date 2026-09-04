import pytest

torch = pytest.importorskip("torch")

from activemap.agent.sft_data import (  # noqa: E402
    ActionSFTCollator,
    encode_action_example,
)


class FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, **kwargs):
        return "|".join(message["content"] for message in messages) + "|assistant:"

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(character) for character in text]}


def test_action_encoding_masks_prompt_but_not_response() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "state"},
            {"role": "assistant", "content": '{"action":"REJECT"}'},
        ]
    }
    encoded, truncated = encode_action_example(row, FakeTokenizer(), max_length=128)
    supervised = [value for value in encoded["labels"] if value != -100]
    assert truncated is False
    assert supervised == FakeTokenizer()(row["messages"][2]["content"] + "<eos>")["input_ids"]
    assert all(value == -100 for value in encoded["labels"][:-len(supervised)])


def test_action_collator_uses_distinct_input_and_label_padding() -> None:
    collator = ActionSFTCollator(pad_token_id=9)
    batch = collator(
        [
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([-100, 2]),
            },
            {
                "input_ids": torch.tensor([3]),
                "attention_mask": torch.tensor([1]),
                "labels": torch.tensor([3]),
            },
        ]
    )
    assert batch["input_ids"].tolist() == [[1, 2], [3, 9]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["labels"].tolist() == [[-100, 2], [3, -100]]
