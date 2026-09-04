import pytest

from activemap.agent.records import AgentAction, AgentActionType
from activemap.agent.sequential_controller import (
    ControllerStage,
    SelectionDecision,
    SequentialControllerAction,
    SequentialControllerTrajectory,
)
from activemap.models import EditOperation


def test_sequential_trajectory_accepts_complete_acquisition_path():
    actions = [
        SequentialControllerAction(
            stage=ControllerStage.DRAFT,
            draft_edit=EditOperation.KEEP,
            confidence=0.1,
        ),
        SequentialControllerAction(
            stage=ControllerStage.SELECT,
            selection=SelectionDecision.ACQUIRE,
            evidence_id="evidence",
        ),
        SequentialControllerAction(
            stage=ControllerStage.TOOL,
            executable_action=AgentAction(
                action=AgentActionType.USE_TOOL,
                tool_call={
                    "call_id": "call",
                    "tool": "RASTER_SEGMENT",
                    "inputs": {"evidence_id": "evidence"},
                    "parameters": {},
                },
            ),
        ),
        SequentialControllerAction(
            stage=ControllerStage.BELIEF_UPDATE,
            updated_edit=EditOperation.ADD,
            confidence=0.9,
        ),
        SequentialControllerAction(
            stage=ControllerStage.TERMINAL,
            executable_action=AgentAction(
                action=AgentActionType.COMMIT, edit=EditOperation.ADD
            ),
        ),
    ]

    trajectory = SequentialControllerTrajectory(
        trajectory_id="trajectory",
        task_id="task",
        split="train",
        policy_snapshot="snapshot",
        selected_tool=True,
        policy_relative_advantage=1.0,
        direct_operation=EditOperation.KEEP,
        post_tool_operation=EditOperation.ADD,
        chosen_operation=EditOperation.ADD,
        target_operation=EditOperation.ADD,
        direct_utility=-0.75,
        post_tool_utility=0.25,
        total_cost=0.75,
        actions=actions,
    )

    assert [action.stage.value for action in trajectory.actions] == [
        "DRAFT",
        "SELECT",
        "TOOL",
        "BELIEF_UPDATE",
        "TERMINAL",
    ]


def test_select_stop_rejects_evidence_payload():
    with pytest.raises(ValueError, match="STOP selection"):
        SequentialControllerAction(
            stage=ControllerStage.SELECT,
            selection=SelectionDecision.STOP,
            evidence_id="not-allowed",
        )
