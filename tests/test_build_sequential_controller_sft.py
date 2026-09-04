import json
from pathlib import Path

from scripts.build_sequential_controller_sft import build_sequential_sft


def _messages(image: Path, stage: str, *, post: bool) -> list[dict]:
    observation = {
        "evidence_id": f"evidence-{stage}",
        "belief": {
            "edit_probabilities": [0.7, 0.1, 0.1, 0.1],
            "confidence": 0.7,
            "geometry_delta": [0.0] * 8,
            "uncertainty": 0.3,
        },
        "available_tools": ["RASTER_SEGMENT"],
        "semantic_tool_cost": 0.75,
        "stage": "POST_TOOL" if post else "PRE_TOOL",
    }
    if post:
        observation["tool_result"] = {"gated_edit": "ADD", "confidence": 0.9}
    return [
        {"role": "system", "content": [{"type": "text", "text": "system"}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image)},
                {"type": "text", "text": json.dumps(observation)},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "{}"}]},
    ]


def test_builder_emits_component_and_joint_curricula(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    rollout = tmp_path / "rollout.jsonl"
    rollout_rows = []
    branches = []
    for index, selected in enumerate((False, True)):
        example_id = f"example-{index}"
        task_id = f"task-{index}"
        for stage, post in (("PRE_TOOL", False), ("POST_TOOL", True)):
            rollout_rows.append(
                {
                    "example_id": example_id,
                    "task_id": task_id,
                    "split": "train",
                    "stage": stage,
                    "oracle_use_tool": selected,
                    "messages": _messages(image, example_id, post=post),
                }
            )
        branches.append(
            {
                "example_id": example_id,
                "task_id": task_id,
                "split": "train",
                "target_operation": "ADD",
                "direct_operation": "KEEP" if selected else "ADD",
                "post_tool_operation": "ADD",
                "belief_operation": "KEEP",
                "policy_relative_use_tool": selected,
                "policy_relative_advantage": 1.0 if selected else -0.75,
                "direct_utility": -0.75 if selected else 1.0,
                "post_tool_utility": 0.25,
                "tool_cost": 0.75,
                "direct_terminal_valid": True,
                "post_terminal_valid": True,
                "static_use_tool": selected,
            }
        )
    rollout.write_text(
        "".join(json.dumps(row) + "\n" for row in rollout_rows), encoding="utf-8"
    )
    branch_root = tmp_path / "branches"
    branch_root.mkdir()
    traces = branch_root / "traces.jsonl"
    traces.write_text(
        "".join(json.dumps(row) + "\n" for row in branches), encoding="utf-8"
    )
    (branch_root / "summary.json").write_text(
        json.dumps(
            {
                "adapter": "/model/final",
                "trace_sha256": "trace-sha",
                "test_assets_read": False,
            }
        ),
        encoding="utf-8",
    )

    summary = build_sequential_sft(
        rollout,
        branch_root,
        tmp_path / "output",
        selector_positive_multiplier=3,
        tool_stage_multiplier=3,
    )

    assert summary["trajectory_count"] == 2
    assert summary["selected_tool_count"] == 1
    assert summary["files"]["joint_sft"]["records"] == 8
    assert summary["files"]["draft_terminal_sft"]["records"] == 4
    assert summary["files"]["selector_sft"]["records"] == 2
    assert summary["files"]["selector_balanced_sft"]["records"] == 4
    assert summary["files"]["tool_belief_sft"]["records"] == 2
    assert summary["files"]["joint_curriculum_sft"]["records"] == 14
    assert summary["joint_audit"]["target_metadata_exposed_to_model"] is False
