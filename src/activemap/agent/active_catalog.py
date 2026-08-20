"""Observable active-catalog retrieval and recurrent controller state construction."""

from __future__ import annotations

from typing import Any

import numpy as np

from activemap.agent.identifiers import public_evidence_id
from activemap.agent.records import AgentAction, AgentActionType, AgentObservation
from activemap.models import EditOperation, EpisodeRecord
from activemap.selector_records import SelectorSample

ACTIVE_CATALOG_SYSTEM_PROMPT = (
    "You are the ActiveMap active evidence selector. Given the current editable-map draft, belief, "
    "budget, evidence history, and a catalog of unobserved remote-sensing candidates, return "
    "exactly one SELECT JSON action. STOP when no candidate has positive quality-safety value. "
    "Otherwise ACQUIRE exactly one available evidence_id. The image shows the direct crop on the "
    "left and same-timestamp wide context with the editable prior outlined in cyan on the right. "
    "Return JSON only."
)


def observable_shortlist_indices(
    sample: SelectorSample, max_candidates: int = 16
) -> list[int]:
    if max_candidates <= 0:
        raise ValueError("max-candidates must be positive")
    if len(sample.evidence_ids) <= max_candidates:
        return list(range(len(sample.evidence_ids)))
    features = sample.evidence_features
    selected: list[int] = []

    def add(index: int) -> None:
        if index not in selected and len(selected) < max_candidates:
            selected.append(index)

    for scale in range(3):
        group = [
            index
            for index, values in enumerate(features)
            if int(np.argmax(np.asarray(values[4:7]))) == scale
        ]
        if not group:
            continue
        ordered = sorted(group, key=lambda index: (features[index][2], index))
        add(ordered[0])
        add(min(group, key=lambda index: (abs(features[index][2]), index)))
        add(ordered[-1])
    remaining = sorted(
        range(len(sample.evidence_ids)),
        key=lambda index: (
            -float(features[index][0]),
            float(sample.evidence_costs[index]),
            abs(float(features[index][2])),
            index,
        ),
    )
    for index in remaining:
        add(index)
    return sorted(selected)


def observable_shortlist_scores(sample: SelectorSample, max_candidates: int = 16) -> np.ndarray:
    selected = observable_shortlist_indices(sample, max_candidates)
    scores = np.full(len(sample.evidence_ids), -1e6, dtype=np.float64)
    for rank, index in enumerate(selected):
        scores[index] = float(len(selected) - rank)
    return scores


def candidate_payload(
    evidence_id: str,
    features: list[float],
    cost: float,
    episode: EpisodeRecord,
) -> dict[str, Any]:
    evidence = next(item for item in episode.evidence_catalog if item.evidence_id == evidence_id)
    scale_index = int(np.argmax(np.asarray(features[4:7])))
    return {
        "evidence_id": public_evidence_id(evidence_id),
        "timestamp": evidence.timestamp,
        "scale": (1, 2, 4)[scale_index],
        "clear_fraction": round(float(features[0]), 6),
        "temporal_offset_normalized": round(float(features[2]), 6),
        "cost": round(float(cost), 6),
    }


def active_catalog_state_from_observation(
    observation: AgentObservation,
    sample: SelectorSample,
    episode: EpisodeRecord,
    *,
    policy_snapshot: str,
) -> dict[str, Any]:
    raw_by_public = {
        public_evidence_id(item.evidence_id): item.evidence_id
        for item in episode.evidence_catalog
    }
    feature_by_raw = {
        evidence_id: (features, cost)
        for evidence_id, features, cost in zip(
            sample.evidence_ids,
            sample.evidence_features,
            sample.evidence_costs,
            strict=True,
        )
    }
    candidates = []
    for candidate in observation.candidates:
        raw_id = raw_by_public.get(candidate.evidence_id)
        if raw_id is None or raw_id not in feature_by_raw:
            raise ValueError(f"observation exposes unknown evidence: {candidate.evidence_id}")
        features, _ = feature_by_raw[raw_id]
        candidates.append(candidate_payload(raw_id, features, candidate.cost, episode))
    return {
        "controller_stage": "SELECT",
        "policy_snapshot": policy_snapshot,
        "direct_draft": {
            "edit": sample.edit_type.value,
            "confidence": round(float(sample.hypothesis_features[13]), 6),
        },
        "belief": {
            "edit_probabilities": [
                round(float(value), 6) for value in observation.belief.edit_probabilities
            ],
            "uncertainty": round(float(observation.belief.uncertainty), 6),
        },
        "budget": {
            "remaining": round(float(observation.remaining_budget), 6),
            "initial_evidence_cost_excluded": bool(
                sample.metadata.get("initial_evidence_cost_excluded", False)
            ),
        },
        "selected_evidence_ids": list(observation.selected_evidence_ids),
        "candidate_evidence": candidates,
        "safety": {"false_edit_risk_limit": 0.02},
    }


def terminal_agent_action(observation: AgentObservation) -> AgentAction:
    edit = observation.belief.recommended_edit or observation.belief.predicted_edit
    if edit == EditOperation.KEEP:
        return AgentAction(action=AgentActionType.REJECT)
    return AgentAction(action=AgentActionType.COMMIT, edit=edit)
