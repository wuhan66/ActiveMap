import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.launch_semantic_vlm_sft_seeds import (
    assignments,
    audit_sft_jsonl,
    command,
    read_last_jsonl,
    validate_data_contract,
)


def test_assignments_queue_three_seeds_on_two_gpus():
    assert assignments([16, 19, 22], [5, 7]) == [(16, 5), (19, 7), (22, 5)]


def test_command_forwards_frozen_training_protocol():
    args = Namespace(
        python="python",
        model=Path("model"),
        train_jsonl=Path("train.jsonl"),
        val_jsonl=Path("val.jsonl"),
        epochs=3.0,
        learning_rate=1e-4,
        batch_size=1,
        gradient_accumulation=16,
        max_length=2048,
        lora_rank=16,
        lora_alpha=32,
        logging_steps=5,
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        early_stopping_patience=3,
        max_train_samples=None,
        max_eval_samples=None,
        init_adapter=Path("adapter"),
        acquire_sampling_target=0.25,
    )

    result = command(args, 20260716, Path("out"))

    expected = {
        "--batch-size": "1",
        "--gradient-accumulation": "16",
        "--max-length": "2048",
        "--lora-rank": "16",
        "--lora-alpha": "32",
        "--eval-steps": "50",
        "--early-stopping-patience": "3",
    }
    for flag, value in expected.items():
        index = result.index(flag)
        assert result[index + 1] == value
    assert result[result.index("--init-adapter") + 1] == "adapter"
    assert result[result.index("--acquire-sampling-target") + 1] == "0.25"

    args.max_train_samples = 2
    args.max_eval_samples = 2
    smoke_result = command(args, 20260716, Path("smoke"))
    assert smoke_result[smoke_result.index("--max-train-samples") + 1] == "2"
    assert smoke_result[smoke_result.index("--max-eval-samples") + 1] == "2"


def test_sft_data_contract_counts_structured_assistant_actions(tmp_path):
    path = tmp_path / "sft.jsonl"
    rows = [
        {
            "task_id": f"task-{index}",
            "stage": "PRE_TOOL",
            "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": text}]}
            ]
        }
        for index, text in enumerate(
            (
                json.dumps({"action": "USE_TOOL"}),
                json.dumps({"action": "COMMIT"}),
            )
        )
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    audit = audit_sft_jsonl(path)

    assert audit["records"] == 2
    assert audit["action_counts"] == {"COMMIT": 1, "USE_TOOL": 1}
    assert audit["stage_counts"] == {"PRE_TOOL": 2}
    assert audit["task_count"] == 2
    args = Namespace(
        expected_train_records=2,
        expected_val_records=2,
        expected_train_use_tool=1,
        expected_val_use_tool=1,
        expected_train_action=[],
        expected_val_action=[],
    )
    validate_data_contract(args, {"train": audit, "val": audit})
    args.expected_train_use_tool = 3
    with pytest.raises(ValueError, match="train.USE_TOOL"):
        validate_data_contract(args, {"train": audit, "val": audit})


def test_sft_data_contract_audits_sequential_select_actions(tmp_path):
    path = tmp_path / "selector.jsonl"
    rows = [
        {
            "task_id": f"task-{index}",
            "stage": "SELECT",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {"stage": "SELECT", "selection": selection}
                            ),
                        }
                    ],
                }
            ],
        }
        for index, selection in enumerate(("ACQUIRE", "STOP", "STOP"))
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    audit = audit_sft_jsonl(path)
    args = Namespace(
        expected_train_records=3,
        expected_val_records=3,
        expected_train_use_tool=None,
        expected_val_use_tool=None,
        expected_train_action=["ACQUIRE=1", "STOP=2"],
        expected_val_action=["ACQUIRE=1", "STOP=2"],
    )

    validate_data_contract(args, {"train": audit, "val": audit})

    assert audit["action_counts"] == {"ACQUIRE": 1, "STOP": 2}


def test_read_last_jsonl_returns_latest_training_event(tmp_path):
    path = tmp_path / "history.jsonl"
    assert read_last_jsonl(path) is None
    path.write_text('{"step":1}\n{"step":5}\n', encoding="utf-8")
    assert read_last_jsonl(path) == {"step": 5}
