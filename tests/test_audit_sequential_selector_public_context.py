import json

from scripts.audit_sequential_selector_public_context import audit_rows


def _row(task_id: str, trajectory_id: str, evidence_id: str, *, all_visible: bool):
    state = {
        "controller_stage": "SELECT",
        "policy_snapshot": "frozen-policy",
        "evidence_id": evidence_id,
        "available_tools": ["RASTER_SEGMENT"],
        "budget": {"initial": 0.75, "spent": 0.0, "remaining": 0.75},
        "direct_draft": {"edit": "ADD", "confidence": 0.9},
    }
    if all_visible:
        state["candidate_evidence_ids"] = ["evidence-a", "evidence-b", "evidence-c"]
    return {
        "task_id": task_id,
        "trajectory_id": trajectory_id,
        "split": "val",
        "stage": "SELECT",
        "messages": [
            {"role": "system", "content": []},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "images/task.jpg"},
                    {"type": "text", "text": json.dumps(state)},
                ],
            },
        ],
    }


def test_single_candidate_prompts_are_not_listwise_eligible():
    rows = [
        _row("task-1", "seq-1", "evidence-a", all_visible=False),
        _row("task-1", "seq-2", "evidence-b", all_visible=False),
        _row("task-1", "seq-3", "evidence-c", all_visible=False),
    ]

    summary, tasks = audit_rows(rows, "val")

    assert summary["single_public_candidate_state_rate"] == 1.0
    assert summary["joint_candidate_visibility_task_rate"] == 0.0
    assert not summary["current_prompt_listwise_eligible"]
    assert summary["explicit_multicandidate_reformulation_task_rate"] == 1.0
    assert not tasks[0]["all_group_candidates_visible_in_each_prompt"]


def test_explicit_candidate_list_makes_joint_scoring_visible():
    rows = [
        _row("task-1", "seq-1", "evidence-a", all_visible=True),
        _row("task-1", "seq-2", "evidence-b", all_visible=True),
        _row("task-1", "seq-3", "evidence-c", all_visible=True),
    ]

    summary, tasks = audit_rows(rows, "val")

    assert summary["joint_candidate_visibility_task_rate"] == 1.0
    assert summary["current_prompt_listwise_eligible"]
    assert tasks[0]["all_group_candidates_visible_in_each_prompt"]
