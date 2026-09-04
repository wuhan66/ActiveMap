from __future__ import annotations

import json

import pytest

from scripts.compose_agent_sft import compose_sft_rows


def _row(
    action: str,
    trajectory: str,
    *,
    split: str,
    stage: int | None = None,
    protocol: str = "post-acquisition-sparse-tool-controller-v1",
) -> dict[str, object]:
    payload: dict[str, object] = {"action": action}
    if action == "USE_TOOL":
        payload["tool_call"] = {
            "call_id": f"call-{trajectory}",
            "tool": "IMAGE_QUALITY",
            "inputs": {"evidence_id": "evidence-1"},
            "parameters": {},
        }
    elif action == "COMMIT":
        payload["edit"] = "ADD"
    row: dict[str, object] = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": json.dumps({"split": split})},
            {"role": "assistant", "content": json.dumps(payload)},
        ],
        "trajectory_id": trajectory,
    }
    if stage is not None:
        row.update(
            {
                "protocol": protocol,
                "oracle_tool_stage": stage,
            }
        )
    return row


def test_training_keeps_positive_trajectory_and_limits_no_tool_sequences() -> None:
    main = [_row("ACQUIRE", "main-1", split="train")]
    tool = [
        _row("USE_TOOL", "positive", split="train", stage=1),
        _row("COMMIT", "positive", split="train", stage=1),
        *[
            _row("REJECT", f"negative-{index}", split="train", stage=0)
            for index in range(10)
        ],
    ]
    rows, summary = compose_sft_rows(
        main,
        tool,
        training=True,
        use_tool_repeat=4,
        no_tool_ratio=2.0,
        seed=7,
    )
    assert summary["selected_tool_negative_sequence_count"] == 2
    assert summary["action_counts"]["USE_TOOL"] == 4
    assert summary["output_count"] == 8
    assert all("composition_source" in row for row in rows)


def test_validation_keeps_natural_distribution_without_repetition() -> None:
    main = [_row("ACQUIRE", "main-1", split="val")]
    tool = [
        _row("USE_TOOL", "positive", split="val", stage=1),
        _row("COMMIT", "positive", split="val", stage=1),
        _row("REJECT", "negative", split="val", stage=0),
    ]
    rows, summary = compose_sft_rows(
        main, tool, training=False, use_tool_repeat=99, no_tool_ratio=0.0
    )
    assert len(rows) == 4
    assert summary["action_counts"]["USE_TOOL"] == 1
    assert summary["selected_tool_negative_sequence_count"] == 1
    assert summary["training_oversampling"] is False


def test_training_can_preserve_natural_tool_sequence_prevalence() -> None:
    main = [_row("ACQUIRE", "main-1", split="train")]
    tool = [
        _row("USE_TOOL", "positive", split="train", stage=1),
        _row("COMMIT", "positive", split="train", stage=1),
        *[
            _row("REJECT", f"negative-{index}", split="train", stage=0)
            for index in range(10)
        ],
    ]

    rows, summary = compose_sft_rows(
        main,
        tool,
        training=True,
        use_tool_repeat=1,
        no_tool_ratio=0.0,
        keep_all_tool_sequences=True,
    )

    assert len(rows) == 13
    assert summary["selected_tool_negative_sequence_count"] == 10
    assert summary["action_counts"]["USE_TOOL"] == 1
    assert summary["keep_all_tool_sequences"] is True


def test_rejects_cross_split_composition() -> None:
    with pytest.raises(ValueError, match="split=train"):
        compose_sft_rows(
            [_row("ACQUIRE", "main", split="val")],
            [_row("REJECT", "tool", split="train", stage=0)],
            training=True,
        )


def test_accepts_selector_reachable_tool_protocol() -> None:
    rows, summary = compose_sft_rows(
        [_row("ACQUIRE", "main", split="train")],
        [
            _row(
                "USE_TOOL",
                "reachable",
                split="train",
                stage=1,
                protocol="post-acquisition-reachable-tool-controller-v2",
            )
        ],
        training=True,
        use_tool_repeat=1,
        no_tool_ratio=0.0,
    )
    assert len(rows) == 2
    assert summary["tool_protocols"] == [
        "post-acquisition-reachable-tool-controller-v2"
    ]


def test_accepts_pointer_action_reachable_tool_protocol() -> None:
    rows, summary = compose_sft_rows(
        [_row("ACQUIRE", "main", split="train")],
        [
            _row(
                "USE_TOOL",
                "pointer",
                split="train",
                stage=1,
                protocol="post-acquisition-reachable-tool-controller-v3-pointer-actions",
            )
        ],
        training=True,
        use_tool_repeat=1,
        no_tool_ratio=0.0,
    )
    assert len(rows) == 2
    assert summary["tool_protocols"] == [
        "post-acquisition-reachable-tool-controller-v3-pointer-actions"
    ]
