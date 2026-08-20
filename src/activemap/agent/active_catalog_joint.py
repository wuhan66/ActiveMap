"""Auditable recurrent transitions for Active-Catalog joint training."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import Field, model_validator

from activemap.agent.records import (
    AgentAction,
    AgentActionType,
    AgentObservation,
    AgentTransition,
)
from activemap.geo_tools.records import GeoToolResult
from activemap.models import EditOperation, StrictModel


class ActiveCatalogJointTransition(StrictModel):
    """One model-selected evidence acquisition and its realized next state."""

    schema_version: Literal["active-catalog-joint-transition-v1"] = (
        "active-catalog-joint-transition-v1"
    )
    transition_id: str
    source_episode: str
    aoi_id: str
    split: Literal["train", "val", "test"]
    policy_snapshot: str
    budget: float = Field(gt=0.0)
    step: int = Field(ge=0)
    action: AgentAction
    prior_observation: AgentObservation
    post_acquisition_observation: AgentObservation
    evidence_id: str
    evidence_cost: float = Field(gt=0.0)
    operation_update_threshold: float = Field(ge=0.0, le=1.0)
    tool_results: list[GeoToolResult] = Field(default_factory=list)
    tool_execution_mode: Literal["none", "explicit"] = "none"
    target_edit: EditOperation
    selected_by_model: Literal[True] = True
    oracle_next_state_replay: Literal[False] = False
    metadata: dict[str, Any]

    @model_validator(mode="after")
    def transition_is_reachable_and_test_free(self) -> ActiveCatalogJointTransition:
        if self.action.action != AgentActionType.ACQUIRE:
            raise ValueError("joint transition action must be ACQUIRE")
        if self.action.evidence_id != self.evidence_id:
            raise ValueError("action and transition evidence IDs disagree")
        before = self.prior_observation
        after = self.post_acquisition_observation
        if before.split != self.split or after.split != self.split:
            raise ValueError("transition observations must match the declared split")
        if before.task_id != after.task_id:
            raise ValueError("transition observations must belong to the same task")
        if before.step != self.step or after.step <= before.step:
            raise ValueError("post-acquisition step must advance")
        if self.evidence_id in before.selected_evidence_ids:
            raise ValueError("acquired evidence was already selected")
        if self.evidence_id not in after.selected_evidence_ids:
            raise ValueError("post-acquisition state must contain acquired evidence")
        before_ids = set(before.selected_evidence_ids)
        after_ids = set(after.selected_evidence_ids)
        if after_ids != before_ids | {self.evidence_id}:
            raise ValueError("an acquisition may add exactly one evidence ID")
        if after.remaining_budget > before.remaining_budget + 1e-8:
            raise ValueError("remaining budget cannot increase")
        observed_cost = before.remaining_budget - after.remaining_budget
        expected_cost = self.evidence_cost + sum(result.cost for result in self.tool_results)
        if abs(observed_cost - expected_cost) > 1e-5:
            raise ValueError("budget delta does not match evidence and tool costs")
        if self.tool_execution_mode == "explicit" and not self.tool_results:
            raise ValueError("explicit tool execution requires grounded tool results")
        if self.tool_execution_mode == "none" and self.tool_results:
            raise ValueError("tool results require explicit tool execution mode")
        for result in self.tool_results:
            result_evidence = result.outputs.get("evidence_id")
            if result_evidence is not None and result_evidence != self.evidence_id:
                raise ValueError("tool result references different evidence")
        if self.metadata.get("test_assets_read") is not False:
            raise ValueError("joint transition must certify test_assets_read=false")
        return self


def joint_transition_from_agent_transition(
    transition: AgentTransition,
    *,
    source_episode: str,
    aoi_id: str,
    policy_snapshot: str,
    target_edit: EditOperation,
    selected_by_model: bool,
    operation_update_threshold: float,
) -> ActiveCatalogJointTransition | None:
    """Convert a realized acquisition without copying oracle supervision into inputs."""

    if transition.action.action != AgentActionType.ACQUIRE:
        return None
    if not selected_by_model:
        raise ValueError("joint training export accepts only model-selected acquisitions")
    after = transition.next_observation
    if after is None:
        raise ValueError("acquisition transition lacks a realized next observation")
    evidence_id = str(transition.action.evidence_id)
    candidate = next(
        (
            item
            for item in transition.observation.candidates
            if item.evidence_id == evidence_id
        ),
        None,
    )
    if candidate is None:
        raise ValueError("selected evidence is absent from the observable shortlist")
    identity = (
        f"{source_episode}|{transition.observation.initial_budget:g}|"
        f"{transition.observation.step}|{evidence_id}|{policy_snapshot}"
    )
    return ActiveCatalogJointTransition(
        transition_id=hashlib.sha256(identity.encode()).hexdigest()[:24],
        source_episode=source_episode,
        aoi_id=aoi_id,
        split=transition.observation.split,
        policy_snapshot=policy_snapshot,
        budget=transition.observation.initial_budget,
        step=transition.observation.step,
        action=transition.action,
        prior_observation=transition.observation,
        post_acquisition_observation=after,
        evidence_id=evidence_id,
        evidence_cost=candidate.cost,
        operation_update_threshold=operation_update_threshold,
        tool_results=[],
        tool_execution_mode="none",
        target_edit=target_edit,
        selected_by_model=True,
        oracle_next_state_replay=False,
        metadata={
            "transition_source": "executed_agent_rollout",
            "oracle_action_exported": False,
            "test_assets_read": False,
        },
    )
