import pytest

from activemap.agent.records import AgentAction, AgentActionType
from activemap.agent.sequential_controller import (
    ControllerStage,
    SelectionDecision,
    SequentialControllerAction,
    SequentialControllerTrajectory,
)
from activemap.models import EditOperation
from scripts.audit_sequential_controller_splits import split_isolation


def _trajectory(task_id: str, split: str) -> SequentialControllerTrajectory:
    operation = EditOperation.KEEP
    return SequentialControllerTrajectory(
        trajectory_id=f"trajectory-{split}-{task_id}",
        task_id=task_id,
        split=split,
        policy_snapshot="snapshot",
        selected_tool=False,
        policy_relative_advantage=-0.5,
        direct_operation=operation,
        post_tool_operation=operation,
        chosen_operation=operation,
        target_operation=operation,
        direct_utility=1.0,
        post_tool_utility=0.25,
        total_cost=0.0,
        actions=[
            SequentialControllerAction(
                stage=ControllerStage.DRAFT,
                draft_edit=operation,
                confidence=0.9,
            ),
            SequentialControllerAction(
                stage=ControllerStage.SELECT,
                selection=SelectionDecision.STOP,
            ),
            SequentialControllerAction(
                stage=ControllerStage.TERMINAL,
                executable_action=AgentAction(action=AgentActionType.REJECT),
            ),
        ],
    )


def test_split_isolation_accepts_disjoint_tasks_with_same_policy():
    result = split_isolation([_trajectory("train-task", "train")], [_trajectory("val-task", "val")])

    assert result["task_overlap"] == 0
    assert result["policy_snapshot"] == "snapshot"


def test_split_isolation_rejects_task_overlap():
    with pytest.raises(ValueError, match="overlap"):
        split_isolation([_trajectory("shared", "train")], [_trajectory("shared", "val")])
