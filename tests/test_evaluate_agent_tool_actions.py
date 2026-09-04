from __future__ import annotations

from activemap.agent.records import AgentAction
from scripts.evaluate_agent_actions import _action_key, _is_executable


def _tool_action(evidence_id: str) -> AgentAction:
    return AgentAction.model_validate(
        {
            "action": "USE_TOOL",
            "tool_call": {
                "call_id": "call-1",
                "tool": "IMAGE_QUALITY",
                "inputs": {"evidence_id": evidence_id},
                "parameters": {},
            },
        }
    )


def test_tool_key_includes_grounded_evidence() -> None:
    assert _action_key(_tool_action("evidence-a")) == (
        "USE_TOOL:IMAGE_QUALITY:evidence-a"
    )
    assert _action_key(_tool_action("evidence-a")) != _action_key(
        _tool_action("evidence-b")
    )


def test_tool_execution_requires_available_tool_and_acquired_evidence() -> None:
    observation = {
        "available_tools": ["IMAGE_QUALITY"],
        "selected_evidence_ids": ["evidence-a"],
    }
    assert _is_executable(_tool_action("evidence-a"), observation)
    assert not _is_executable(_tool_action("evidence-b"), observation)
    assert not _is_executable(
        _tool_action("evidence-a"),
        {"available_tools": [], "selected_evidence_ids": ["evidence-a"]},
    )
