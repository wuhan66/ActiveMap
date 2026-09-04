from __future__ import annotations

import json

import pytest

from scripts.compose_agent_preferences import compose_preference_rows


def _action(name: str) -> str:
    payload: dict[str, object] = {"action": name}
    if name == "USE_TOOL":
        payload["tool_call"] = {
            "call_id": "call-1",
            "tool": "IMAGE_QUALITY",
            "inputs": {"evidence_id": "evidence-1"},
            "parameters": {},
        }
    elif name == "COMMIT":
        payload["edit"] = "ADD"
    return json.dumps(payload, separators=(",", ":"))


def _row(
    chosen: str,
    rejected: str,
    trajectory: str,
    *,
    split: str | None = None,
    tool: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "system": "system",
        "prompt": "prompt",
        "chosen": _action(chosen),
        "rejected": _action(rejected),
        "chosen_utility": 1.0,
        "rejected_utility": 0.0,
        "trajectory_id": trajectory,
    }
    if tool:
        row.update(
            {
                "protocol": "post-acquisition-sparse-tool-controller-v1",
                "split": split,
            }
        )
    return row


def test_training_repeats_tool_choices_and_limits_negative_sequences() -> None:
    main = [_row("COMMIT", "REJECT", "main")]
    tool = [
        _row("USE_TOOL", "REJECT", "positive", split="train", tool=True),
        _row("COMMIT", "USE_TOOL", "positive", split="train", tool=True),
        *[
            _row("REJECT", "USE_TOOL", f"negative-{index}", split="train", tool=True)
            for index in range(8)
        ],
    ]
    rows, summary = compose_preference_rows(
        main,
        tool,
        split="train",
        use_tool_repeat=4,
        no_tool_ratio=2.0,
        seed=3,
    )
    assert len(rows) == 8
    assert summary["chosen_use_tool_count"] == 4
    assert summary["selected_tool_negative_sequence_count"] == 2
    assert all(
        row.get("preference_family") == "tool"
        for row in rows
        if row["composition_source"] == "sparse_tool_controller"
    )


def test_validation_preserves_all_tool_preferences_once() -> None:
    rows, summary = compose_preference_rows(
        [_row("COMMIT", "REJECT", "main")],
        [
            _row("USE_TOOL", "REJECT", "positive", split="val", tool=True),
            _row("REJECT", "USE_TOOL", "negative", split="val", tool=True),
        ],
        split="val",
        use_tool_repeat=99,
        no_tool_ratio=0.0,
    )
    assert len(rows) == 3
    assert summary["chosen_use_tool_count"] == 1
    assert summary["training_oversampling"] is False


def test_rejects_tool_split_mismatch() -> None:
    with pytest.raises(ValueError, match="split mismatch"):
        compose_preference_rows(
            [_row("COMMIT", "REJECT", "main")],
            [_row("REJECT", "USE_TOOL", "tool", split="val", tool=True)],
            split="train",
        )
