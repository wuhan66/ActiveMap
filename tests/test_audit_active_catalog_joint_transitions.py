from pathlib import Path

import pytest

from scripts.audit_active_catalog_joint_transitions import audit
from activemap.agent.active_catalog_joint import joint_transition_from_agent_transition
from activemap.agent.records import (
    AgentAction,
    AgentActionType,
    AgentBelief,
    AgentCandidate,
    AgentObservation,
    AgentTransition,
)
from activemap.models import EditOperation


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
        belief=AgentBelief(
            edit_probabilities=[0.6, 0.2, 0.1, 0.1],
            confidence=0.8,
            uncertainty=0.3,
        ),
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
            "belief": AgentBelief(
                edit_probabilities=[0.2, 0.7, 0.05, 0.05],
                confidence=0.8,
                uncertainty=0.3,
            ),
            "candidates": [],
        }
    )
    return AgentTransition(
        observation=before,
        action=AgentAction(action=AgentActionType.ACQUIRE, evidence_id=evidence_id),
        reward=0.2,
        done=False,
        next_observation=after,
    )


def _write(path: Path, *, split: str, episode: str, aoi: str) -> None:
    transition = _transition()
    transition = transition.model_copy(
        update={
            "observation": transition.observation.model_copy(update={"split": split}),
            "next_observation": transition.next_observation.model_copy(update={"split": split}),
        }
    )
    row = joint_transition_from_agent_transition(
        transition,
        source_episode=episode,
        aoi_id=aoi,
        policy_snapshot="adapter",
        target_edit=EditOperation.ADD,
        selected_by_model=True,
        operation_update_threshold=0.69,
    )
    assert row is not None
    path.write_text(row.model_dump_json() + "\n", encoding="utf-8")


def test_joint_transition_audit_passes_disjoint_splits(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write(train, split="train", episode="train-episode", aoi="train-aoi")
    _write(val, split="val", episode="val-episode", aoi="val-aoi")
    summary = audit(train, val)
    assert summary["passed"] is True
    assert summary["aoi_overlap"] == 0
    assert summary["train"]["belief_change_rate"] == 1.0


def test_joint_transition_audit_rejects_aoi_leakage(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write(train, split="train", episode="train-episode", aoi="same-aoi")
    _write(val, split="val", episode="val-episode", aoi="same-aoi")
    with pytest.raises(ValueError, match="train/validation leakage"):
        audit(train, val)


def test_joint_transition_audit_rejects_wrong_split(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write(train, split="val", episode="train-episode", aoi="train-aoi")
    _write(val, split="val", episode="val-episode", aoi="val-aoi")
    with pytest.raises(ValueError, match="unexpected split"):
        audit(train, val)
