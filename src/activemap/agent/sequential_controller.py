"""Validated actions and trajectories for draft-conditioned sequential control."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from activemap.agent.records import AgentAction, AgentActionType
from activemap.models import EditOperation, StrictModel


class ControllerStage(str, Enum):
    DRAFT = "DRAFT"
    SELECT = "SELECT"
    TOOL = "TOOL"
    BELIEF_UPDATE = "BELIEF_UPDATE"
    TERMINAL = "TERMINAL"


class SelectionDecision(str, Enum):
    ACQUIRE = "ACQUIRE"
    STOP = "STOP"


class SequentialControllerAction(StrictModel):
    stage: ControllerStage
    draft_edit: EditOperation | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    selection: SelectionDecision | None = None
    evidence_id: str | None = None
    updated_edit: EditOperation | None = None
    executable_action: AgentAction | None = None

    @model_validator(mode="after")
    def payload_matches_stage(self) -> SequentialControllerAction:
        populated = {
            "draft_edit": self.draft_edit is not None,
            "confidence": self.confidence is not None,
            "selection": self.selection is not None,
            "evidence_id": self.evidence_id is not None,
            "updated_edit": self.updated_edit is not None,
            "executable_action": self.executable_action is not None,
        }
        if self.stage == ControllerStage.DRAFT:
            required = {"draft_edit", "confidence"}
        elif self.stage == ControllerStage.SELECT:
            required = {"selection"}
            if self.selection == SelectionDecision.ACQUIRE:
                required.add("evidence_id")
            elif self.evidence_id is not None:
                raise ValueError("STOP selection cannot include evidence_id")
        elif self.stage == ControllerStage.TOOL:
            required = {"executable_action"}
            if (
                self.executable_action is not None
                and self.executable_action.action != AgentActionType.USE_TOOL
            ):
                raise ValueError("TOOL stage requires an executable USE_TOOL action")
        elif self.stage == ControllerStage.BELIEF_UPDATE:
            required = {"updated_edit", "confidence"}
        else:
            required = {"executable_action"}
            if self.executable_action is not None and self.executable_action.action not in {
                AgentActionType.COMMIT,
                AgentActionType.REJECT,
            }:
                raise ValueError("TERMINAL stage requires COMMIT or REJECT")
        unexpected = {
            name for name, present in populated.items() if present and name not in required
        }
        missing = {name for name in required if not populated[name]}
        if unexpected or missing:
            raise ValueError(
                f"invalid {self.stage.value} payload: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        return self


class SequentialControllerTrajectory(StrictModel):
    trajectory_id: str
    task_id: str
    split: str
    policy_snapshot: str
    selected_tool: bool
    policy_relative_advantage: float
    direct_operation: EditOperation
    post_tool_operation: EditOperation
    chosen_operation: EditOperation
    target_operation: EditOperation
    direct_utility: float
    post_tool_utility: float
    total_cost: float = Field(ge=0.0)
    actions: list[SequentialControllerAction]

    @model_validator(mode="after")
    def stage_order_and_choice_are_consistent(self) -> SequentialControllerTrajectory:
        expected = [
            ControllerStage.DRAFT,
            ControllerStage.SELECT,
            *(
                [ControllerStage.TOOL, ControllerStage.BELIEF_UPDATE]
                if self.selected_tool
                else []
            ),
            ControllerStage.TERMINAL,
        ]
        if [action.stage for action in self.actions] != expected:
            raise ValueError("sequential controller stages are incomplete or out of order")
        selection = self.actions[1]
        expected_selection = (
            SelectionDecision.ACQUIRE if self.selected_tool else SelectionDecision.STOP
        )
        if selection.selection != expected_selection:
            raise ValueError("selection action disagrees with selected_tool")
        expected_operation = (
            self.post_tool_operation if self.selected_tool else self.direct_operation
        )
        if self.chosen_operation != expected_operation:
            raise ValueError("chosen operation disagrees with the selected branch")
        return self
