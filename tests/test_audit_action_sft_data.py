import json
from pathlib import Path

from scripts.audit_action_sft_data import _read


def _row(split: str, action: dict[str, object]) -> dict[str, object]:
    observation = {
        "split": split,
        "available_tools": ["IMAGE_QUALITY"],
        "selected_evidence_ids": ["evidence-1"],
    }
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": json.dumps(observation)},
            {"role": "assistant", "content": json.dumps(action)},
        ],
        "trajectory_id": "task-1__b1p5",
        "step": 0,
        "composition_source": "test",
    }


def test_action_sft_audit_counts_grounded_tool(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            _row(
                "train",
                {
                    "action": "USE_TOOL",
                    "tool_call": {
                        "tool": "IMAGE_QUALITY",
                        "inputs": {"evidence_id": "evidence-1"},
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    report = _read(path, "train")
    assert report["tool_rows"] == 1
    assert report["grounded_tool_rows"] == 1
    assert report["tool_grounding_rate"] == 1.0


def test_action_sft_audit_detects_ungrounded_tool(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            _row(
                "train",
                {
                    "action": "USE_TOOL",
                    "tool_call": {
                        "tool": "IMAGE_QUALITY",
                        "inputs": {"evidence_id": "evidence-other"},
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    report = _read(path, "train")
    assert report["ungrounded_tool_rows"] == 1


def test_action_sft_audit_accepts_grounded_pointer_tool(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            _row(
                "train",
                {
                    "action": "USE_TOOL",
                    "tool_call": {
                        "tool": "IMAGE_QUALITY",
                        "inputs": {"evidence_index": 0},
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    report = _read(path, "train")
    assert report["grounded_tool_rows"] == 1
    assert report["pointer_tool_rows"] == 1
