from __future__ import annotations

import pytest

from activemap.agent.records import AgentBelief
from activemap.agent.tool_belief_data import (
    ToolBeliefSequenceExample,
    ToolBeliefSequenceStep,
)
from activemap.agent.tool_sft import build_sparse_tool_examples, select_sparse_tool_stage
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import EditOperation


def _belief(edit: EditOperation) -> AgentBelief:
    probabilities = [0.05, 0.05, 0.05, 0.05]
    probabilities[list(EditOperation).index(edit)] = 0.85
    return AgentBelief(
        edit_probabilities=probabilities,
        confidence=0.85,
        uncertainty=0.2,
        recommended_edit=edit,
    )


def _result(tool: GeoToolName, step: int) -> GeoToolResult:
    outputs = (
        {"valid_fraction": 1.0, "sharpness": 0.1}
        if tool == GeoToolName.IMAGE_QUALITY
        else {"changed_fraction": 0.4, "stats": {"mean": 0.2, "p95": 0.6}}
    )
    return GeoToolResult(
        call_id=f"call-{tool.value}-{step}",
        tool=tool,
        success=True,
        outputs=outputs,
        cost=0.03 if tool == GeoToolName.IMAGE_QUALITY else 0.15,
    )


def _sequence() -> ToolBeliefSequenceExample:
    steps = []
    for index in range(1, 4):
        steps.append(
            ToolBeliefSequenceStep(
                evidence_id=f"evidence-{index}",
                quality_result=_result(GeoToolName.IMAGE_QUALITY, index),
                temporal_result=_result(GeoToolName.TEMPORAL_CHANGE, index),
                individual_target_belief=_belief(EditOperation.ADD),
                cumulative_target_belief=_belief(EditOperation.ADD),
            )
        )
    return ToolBeliefSequenceExample(
        sequence_id="sequence-1",
        episode_id="task-1",
        split="train",
        initial_belief=_belief(EditOperation.KEEP),
        steps=steps,
        gt_edit=EditOperation.ADD,
        metadata={"test_assets_read": False},
    )


def _details(paired: list[str]) -> list[dict[str, object]]:
    return [
        {
            "sequence_id": "sequence-1",
            "step": index,
            "target": "ADD",
            "baseline": "KEEP",
            "paired": prediction,
            "spent_cost": 0.18 * index,
        }
        for index, prediction in enumerate(paired, start=1)
    ]


def test_selects_first_beneficial_tool_prefix() -> None:
    stage, utilities, _ = select_sparse_tool_stage(_details(["ADD", "ADD", "ADD"]))
    assert stage == 1
    assert utilities[1] > utilities[0]


def test_tie_prefers_no_tool() -> None:
    rows = _details(["KEEP", "KEEP", "KEEP"])
    stage, _, _ = select_sparse_tool_stage(rows)
    assert stage == 0


def test_builds_grounded_tool_calls_and_terminal_action() -> None:
    sft, preferences, summary = build_sparse_tool_examples(
        [_sequence()], _details(["ADD", "ADD", "ADD"])
    )
    assert summary["oracle_stage_counts"] == {"0": 0, "1": 1, "2": 0, "3": 0}
    assert len(sft) == 3
    assert len(preferences) == 3
    assert '"tool":"IMAGE_QUALITY"' in sft[0]["messages"][2]["content"]
    assert '"evidence_index":0' in sft[0]["messages"][2]["content"]
    assert '"evidence_id"' not in sft[0]["messages"][2]["content"]
    assert '"tool":"TEMPORAL_CHANGE"' in sft[1]["messages"][2]["content"]
    assert '"edit":"ADD"' in sft[2]["messages"][2]["content"]
    assert '"action":"USE_TOOL"' in preferences[0]["chosen"]
    assert '"action":"USE_TOOL"' in preferences[1]["chosen"]
    assert '"edit":"ADD"' in preferences[2]["chosen"]
    prompt = sft[2]["messages"][1]["content"]
    assert "changed_fraction" in prompt
    assert "minimum" not in prompt


def test_no_tool_sequence_teaches_terminal_and_rejects_call() -> None:
    rows = _details(["KEEP", "KEEP", "KEEP"])
    sft, preferences, summary = build_sparse_tool_examples([_sequence()], rows)
    assert summary["tool_positive_sequence_count"] == 0
    assert len(sft) == 1
    assert '"edit":"ADD"' in sft[0]["messages"][2]["content"]
    assert '"action":"USE_TOOL"' in preferences[0]["rejected"]


def test_test_sparse_tool_labels_require_frozen_authorization() -> None:
    sequence = _sequence().model_copy(update={"split": "test"})
    with pytest.raises(PermissionError, match="run_frozen_paper_test"):
        build_sparse_tool_examples([sequence], _details(["ADD", "ADD", "ADD"]))
