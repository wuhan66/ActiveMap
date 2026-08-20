"""Validated records for closed-loop map-maintenance trajectories."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from activemap.geo_tools.records import GeoToolCall, GeoToolName, GeoToolResult
from activemap.models import EditOperation, StrictModel


class AgentActionType(str, Enum):
    ACQUIRE = "ACQUIRE"
    USE_TOOL = "USE_TOOL"
    COMMIT = "COMMIT"
    REJECT = "REJECT"


class AgentBelief(StrictModel):
    edit_probabilities: list[float]
    confidence: float = Field(ge=0.0, le=1.0)
    geometry_delta: list[float] = Field(default_factory=lambda: [0.0] * 8)
    uncertainty: float = Field(ge=0.0, le=1.0)
    recommended_edit: EditOperation | None = None

    @model_validator(mode="after")
    def dimensions_and_probability_are_valid(self) -> AgentBelief:
        if len(self.edit_probabilities) != len(EditOperation):
            raise ValueError("edit_probabilities must match EditOperation")
        if any(value < 0.0 or value > 1.0 for value in self.edit_probabilities):
            raise ValueError("edit probabilities must be between zero and one")
        if abs(sum(self.edit_probabilities) - 1.0) > 1e-4:
            raise ValueError("edit probabilities must sum to one")
        if len(self.geometry_delta) != 8:
            raise ValueError("geometry_delta must have eight values")
        return self

    @property
    def predicted_edit(self) -> EditOperation:
        if self.recommended_edit is not None:
            return self.recommended_edit
        index = max(range(len(self.edit_probabilities)), key=self.edit_probabilities.__getitem__)
        return list(EditOperation)[index]


class AgentCandidate(StrictModel):
    evidence_id: str
    cost: float = Field(gt=0.0)
    selector_score: float
    features: list[float]


class AgentObservation(StrictModel):
    task_id: str
    split: str
    step: int = Field(ge=0)
    initial_budget: float = Field(ge=0.0)
    remaining_budget: float = Field(ge=0.0)
    spent_cost: float = Field(ge=0.0)
    selected_evidence_ids: list[str]
    belief: AgentBelief
    candidates: list[AgentCandidate]
    terminal_score: float | None = None
    available_tools: list[GeoToolName] = Field(default_factory=list)
    tool_history: list[GeoToolResult] = Field(default_factory=list)
    terminal_actions: list[AgentActionType] = Field(
        default_factory=lambda: [AgentActionType.COMMIT, AgentActionType.REJECT]
    )

    @field_validator("split")
    @classmethod
    def split_is_known(cls, value: str) -> str:
        if value not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        return value


class AgentAction(StrictModel):
    action: AgentActionType
    evidence_id: str | None = None
    edit: EditOperation | None = None
    tool_call: GeoToolCall | None = None

    @model_validator(mode="after")
    def payload_matches_action(self) -> AgentAction:
        if self.action == AgentActionType.ACQUIRE:
            if self.evidence_id is None or self.edit is not None or self.tool_call is not None:
                raise ValueError("ACQUIRE requires only evidence_id")
        elif self.action == AgentActionType.USE_TOOL:
            if self.tool_call is None or self.evidence_id is not None or self.edit is not None:
                raise ValueError("USE_TOOL requires only tool_call")
        elif self.action == AgentActionType.COMMIT:
            if (
                self.edit is None
                or self.edit == EditOperation.KEEP
                or self.evidence_id is not None
                or self.tool_call is not None
            ):
                raise ValueError("COMMIT requires only a non-KEEP edit")
        elif self.evidence_id is not None or self.edit is not None or self.tool_call is not None:
            raise ValueError("REJECT does not accept an action payload")
        return self

    @property
    def key(self) -> str:
        if self.action == AgentActionType.ACQUIRE:
            return f"ACQUIRE:{self.evidence_id}"
        if self.action == AgentActionType.USE_TOOL:
            return f"USE_TOOL:{self.tool_call.tool.value if self.tool_call else 'UNKNOWN'}"
        if self.action == AgentActionType.COMMIT:
            return f"COMMIT:{self.edit.value if self.edit else 'UNKNOWN'}"
        return AgentActionType.REJECT.value


class AgentTransition(StrictModel):
    observation: AgentObservation
    action: AgentAction
    reward: float
    done: bool
    next_observation: AgentObservation | None = None
    oracle_action: AgentAction | None = None
    oracle_utilities: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def terminal_contract_is_valid(self) -> AgentTransition:
        if self.done and self.next_observation is not None:
            raise ValueError("terminal transition cannot have next_observation")
        if not self.done and self.next_observation is None:
            raise ValueError("non-terminal transition requires next_observation")
        return self


class AgentTrajectory(StrictModel):
    trajectory_id: str
    task_id: str
    split: str
    budget: float = Field(gt=0.0)
    transitions: list[AgentTransition]
    total_reward: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def trajectory_ends_once(self) -> AgentTrajectory:
        if not self.transitions or not self.transitions[-1].done:
            raise ValueError("trajectory must end with a terminal transition")
        if any(step.done for step in self.transitions[:-1]):
            raise ValueError("only the final transition may be terminal")
        return self
