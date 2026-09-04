import pytest

from activemap.agent.records import AgentBelief
from activemap.agent.tool_belief_data import ToolBeliefExample
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import EditOperation


def _belief(probabilities: list[float]) -> AgentBelief:
    return AgentBelief(
        edit_probabilities=probabilities,
        confidence=max(probabilities),
        uncertainty=0.5,
        geometry_delta=[0.0] * 8,
    )


def test_failed_tool_supervision_must_preserve_prior_belief() -> None:
    prior = _belief([0.7, 0.1, 0.1, 0.1])
    failed = GeoToolResult(
        call_id="failed-1",
        tool=GeoToolName.TEMPORAL_CHANGE,
        success=False,
        cost=0.15,
        error="registration failed",
    )

    with pytest.raises(ValueError, match="preserve the prior belief"):
        ToolBeliefExample(
            record_id="record-1",
            episode_id="task-1",
            split="train",
            evidence_id="evidence-1",
            prior_belief=prior,
            tool_result=failed,
            target_belief=_belief([0.1, 0.7, 0.1, 0.1]),
            gt_edit=EditOperation.ADD,
            metadata={},
        )


def test_tool_supervision_rejects_test_split() -> None:
    prior = _belief([0.7, 0.1, 0.1, 0.1])
    result = GeoToolResult(
        call_id="quality-1",
        tool=GeoToolName.IMAGE_QUALITY,
        success=True,
        outputs={"valid_fraction": 1.0},
        cost=0.03,
    )

    with pytest.raises(ValueError, match="only train or val"):
        ToolBeliefExample(
            record_id="record-1",
            episode_id="task-1",
            split="test",
            evidence_id="evidence-1",
            prior_belief=prior,
            tool_result=result,
            target_belief=prior,
            gt_edit=EditOperation.KEEP,
            metadata={},
        )
