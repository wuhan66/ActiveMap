import pytest

from activemap.agent.active_catalog_joint import (
    ActiveCatalogJointTransition,
    joint_transition_from_agent_transition,
)
from activemap.agent.records import (
    AgentAction,
    AgentActionType,
    AgentBelief,
    AgentCandidate,
    AgentObservation,
    AgentTransition,
)
from activemap.models import EditOperation


def _belief(probabilities):
    return AgentBelief(
        edit_probabilities=probabilities,
        confidence=0.8,
        uncertainty=0.3,
    )


def _transition() -> AgentTransition:
    evidence_id = "evidence-public"
    before = AgentObservation(
        task_id="task-public",
        split="train",
        step=0,
        initial_budget=3.0,
        remaining_budget=3.0,
        spent_cost=0.0,
        selected_evidence_ids=["anchor-public"],
        belief=_belief([0.6, 0.2, 0.1, 0.1]),
        candidates=[
            AgentCandidate(
                evidence_id=evidence_id,
                cost=1.0,
                selector_score=0.4,
                features=[0.0] * 13,
            )
        ],
    )
    after = before.model_copy(
        update={
            "step": 1,
            "remaining_budget": 2.0,
            "spent_cost": 1.0,
            "selected_evidence_ids": ["anchor-public", evidence_id],
            "belief": _belief([0.2, 0.7, 0.05, 0.05]),
            "candidates": [],
        }
    )
    return AgentTransition(
        observation=before,
        action=AgentAction(action=AgentActionType.ACQUIRE, evidence_id=evidence_id),
        reward=0.2,
        done=False,
        next_observation=after,
        oracle_action=AgentAction(action=AgentActionType.REJECT),
        oracle_utilities={"ACQUIRE:evidence-public": 0.2},
    )


def test_exported_joint_transition_is_reachable_and_hides_oracle() -> None:
    row = joint_transition_from_agent_transition(
        _transition(),
        source_episode="train-episode",
        aoi_id="aoi",
        policy_snapshot="adapter",
        target_edit=EditOperation.ADD,
        selected_by_model=True,
        operation_update_threshold=0.69,
    )
    assert row is not None
    assert row.post_acquisition_observation.belief.predicted_edit == EditOperation.ADD
    assert row.oracle_next_state_replay is False
    payload = row.model_dump(mode="json")
    assert "oracle_action" not in payload
    assert "oracle_utilities" not in payload


def test_joint_transition_rejects_oracle_selected_export() -> None:
    with pytest.raises(ValueError, match="model-selected"):
        joint_transition_from_agent_transition(
            _transition(),
            source_episode="train-episode",
            aoi_id="aoi",
            policy_snapshot="adapter",
            target_edit=EditOperation.ADD,
            selected_by_model=False,
            operation_update_threshold=0.69,
        )


def test_joint_transition_rejects_unreachable_next_state() -> None:
    row = joint_transition_from_agent_transition(
        _transition(),
        source_episode="train-episode",
        aoi_id="aoi",
        policy_snapshot="adapter",
        target_edit=EditOperation.ADD,
        selected_by_model=True,
        operation_update_threshold=0.69,
    )
    assert row is not None
    payload = row.model_dump(mode="json")
    payload["post_acquisition_observation"]["selected_evidence_ids"] = ["anchor-public"]
    with pytest.raises(ValueError, match="must contain acquired evidence"):
        ActiveCatalogJointTransition.model_validate(payload)


def test_joint_transition_rejects_validation_as_training_mix() -> None:
    row = joint_transition_from_agent_transition(
        _transition(),
        source_episode="train-episode",
        aoi_id="aoi",
        policy_snapshot="adapter",
        target_edit=EditOperation.ADD,
        selected_by_model=True,
        operation_update_threshold=0.69,
    )
    assert row is not None
    payload = row.model_dump(mode="json")
    payload["split"] = "val"
    with pytest.raises(ValueError, match="declared split"):
        ActiveCatalogJointTransition.model_validate(payload)


def test_joint_transition_supports_frozen_test_export() -> None:
    transition = _transition()
    before = transition.observation.model_copy(update={"split": "test"})
    after = transition.next_observation.model_copy(update={"split": "test"})
    transition = transition.model_copy(
        update={"observation": before, "next_observation": after}
    )

    row = joint_transition_from_agent_transition(
        transition,
        source_episode="test-episode",
        aoi_id="held-out-aoi",
        policy_snapshot="frozen-adapter",
        target_edit=EditOperation.ADD,
        selected_by_model=True,
        operation_update_threshold=0.69,
    )

    assert row is not None
    assert row.split == "test"
    assert row.prior_observation.split == "test"
    assert row.post_acquisition_observation.split == "test"
