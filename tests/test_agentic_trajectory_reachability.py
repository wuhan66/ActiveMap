import json

from scripts.audit_agentic_trajectory_reachability import audit_reachability


def _row(task, source, selected, action, trajectory="trajectory"):
    observation = {
        "task_id": task,
        "split": "val",
        "selected_evidence_ids": selected,
    }
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": json.dumps(observation)},
            {"role": "assistant", "content": json.dumps(action)},
        ],
        "trajectory_id": trajectory,
        "step": 0,
        "composition_source": source,
    }


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_reachability_accepts_tool_on_acquired_evidence(tmp_path):
    path = tmp_path / "sft.jsonl"
    _write(
        path,
        [
            _row(
                "task-a",
                "acquisition_agent",
                ["anchor"],
                {"action": "ACQUIRE", "evidence_id": "e1"},
            ),
            _row(
                "task-a",
                "sparse_tool_controller",
                ["anchor", "e1"],
                {
                    "action": "USE_TOOL",
                    "tool_call": {"tool": "IMAGE_QUALITY", "inputs": {"evidence_id": "e1"}},
                },
            ),
        ],
    )

    report = audit_reachability(path)

    assert report["passed"] is True
    assert report["reachable_tool_state_rate"] == 1.0


def test_reachability_rejects_disconnected_tool_trajectory(tmp_path):
    path = tmp_path / "sft.jsonl"
    _write(
        path,
        [
            _row("task-a", "acquisition_agent", ["anchor"], {"action": "COMMIT", "edit": "ADD"}),
            _row(
                "task-a",
                "sparse_tool_controller",
                ["e2"],
                {
                    "action": "USE_TOOL",
                    "tool_call": {"tool": "TEMPORAL_CHANGE", "inputs": {"evidence_id": "e2"}},
                },
            ),
        ],
    )

    report = audit_reachability(path)

    assert report["passed"] is False
    assert report["reachable_tool_state_rate"] == 0.0
