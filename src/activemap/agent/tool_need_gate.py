"""Structured, naturally calibrated gate for admitting Agent tool calls."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from activemap.agent.records import AgentAction, AgentActionType, AgentObservation
from activemap.agent.tool_belief_model import encode_belief
from activemap.agent.tool_features import TOOL_RESULT_FEATURE_DIM, encode_tool_result
from activemap.geo_tools.records import GeoToolCall, GeoToolName
from activemap.models import EditOperation

_TOOLS = tuple(GeoToolName)
TOOL_NEED_FEATURE_NAMES = tuple(
    [f"belief_{index}" for index in range(14)]
    + [f"best_candidate_{index}" for index in range(13)]
    + [f"mean_candidate_{index}" for index in range(13)]
    + [f"last_tool_result_{index}" for index in range(TOOL_RESULT_FEATURE_DIM)]
    + [f"available_{tool.value}" for tool in _TOOLS]
    + [
        "step",
        "initial_budget",
        "remaining_budget",
        "spent_cost",
        "remaining_budget_fraction",
        "spent_budget_fraction",
        "selected_evidence_count",
        "candidate_count",
        "best_selector_score",
        "mean_selector_score",
        "terminal_score",
        "tool_history_count",
    ]
)


def structured_tool_need_features(observation: AgentObservation) -> np.ndarray:
    """Encode only information available before the proposed tool call."""

    candidate_matrix = np.asarray(
        [candidate.features for candidate in observation.candidates], dtype=np.float32
    )
    if candidate_matrix.size:
        if candidate_matrix.ndim != 2 or candidate_matrix.shape[1] != 13:
            raise ValueError("Agent candidate features must have dimension 13")
        scores = np.asarray(
            [candidate.selector_score for candidate in observation.candidates],
            dtype=np.float32,
        )
        best = candidate_matrix[int(np.argmax(scores))]
        mean = candidate_matrix.mean(axis=0)
        best_score = float(scores.max())
        mean_score = float(scores.mean())
    else:
        best = np.zeros(13, dtype=np.float32)
        mean = np.zeros(13, dtype=np.float32)
        best_score = 0.0
        mean_score = 0.0
    last_tool = (
        np.asarray(encode_tool_result(observation.tool_history[-1]), dtype=np.float32)
        if observation.tool_history
        else np.zeros(TOOL_RESULT_FEATURE_DIM, dtype=np.float32)
    )
    initial_budget = max(float(observation.initial_budget), 1e-8)
    values = np.asarray(
        encode_belief(observation.belief)
        + best.tolist()
        + mean.tolist()
        + last_tool.tolist()
        + [float(tool in observation.available_tools) for tool in _TOOLS]
        + [
            float(observation.step),
            float(observation.initial_budget),
            float(observation.remaining_budget),
            float(observation.spent_cost),
            float(observation.remaining_budget) / initial_budget,
            float(observation.spent_cost) / initial_budget,
            float(len(observation.selected_evidence_ids)),
            float(len(observation.candidates)),
            best_score,
            mean_score,
            float(observation.terminal_score or 0.0),
            float(len(observation.tool_history)),
        ],
        dtype=np.float32,
    )
    if values.shape != (len(TOOL_NEED_FEATURE_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError("invalid structured Tool-Need feature vector")
    return values


class CalibratedToolNeedPolicy:
    """Gate tool proposals and optionally initiate a calibrated paired call."""

    def __init__(
        self,
        policy: Any,
        gate: Any,
        *,
        threshold: float,
        records: list[dict[str, Any]] | None = None,
        proactive_pair: bool = False,
        pair_cost: float = 0.36,
        controller: str = "unspecified",
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Tool-Need threshold must be in [0, 1]")
        self.policy = policy
        self.gate = gate
        self.threshold = float(threshold)
        self.records = records if records is not None else []
        self.proactive_pair = bool(proactive_pair)
        self.pair_cost = float(pair_cost)
        self.controller = controller
        self.pending_temporal_evidence_id: str | None = None
        self.processed_evidence_ids: set[str] = set()

    @staticmethod
    def _terminal_fallback(observation: AgentObservation) -> AgentAction:
        predicted = observation.belief.predicted_edit
        return (
            AgentAction(action=AgentActionType.REJECT)
            if predicted == EditOperation.KEEP
            else AgentAction(action=AgentActionType.COMMIT, edit=predicted)
        )

    @staticmethod
    def _tool_action(
        observation: AgentObservation,
        tool: GeoToolName,
        evidence_id: str,
    ) -> AgentAction:
        identity = f"{observation.task_id}|{evidence_id}|{tool.value}|{observation.step}"
        return AgentAction(
            action=AgentActionType.USE_TOOL,
            tool_call=GeoToolCall(
                call_id=f"tn-{hashlib.sha256(identity.encode()).hexdigest()[:20]}",
                tool=tool,
                inputs={"evidence_id": evidence_id},
            ),
        )

    def _record(
        self,
        observation: AgentObservation,
        probability: float,
        *,
        admitted: bool,
        proposed_action: str,
        executed_action: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        features = structured_tool_need_features(observation)
        record = {
            "task_id": observation.task_id,
            "step": observation.step,
            "controller": self.controller,
            "call_probability": probability,
            "threshold": self.threshold,
            "admitted": admitted,
            "source": source,
            "proposed_action": proposed_action,
            "executed_action": executed_action,
            "pre_tool_features": features.tolist(),
            "pre_tool_feature_names": TOOL_NEED_FEATURE_NAMES,
            "remaining_budget": observation.remaining_budget,
            "predicted_edit": observation.belief.predicted_edit.value,
            "belief_confidence": observation.belief.confidence,
            "belief_uncertainty": observation.belief.uncertainty,
            "selected_evidence_count": len(observation.selected_evidence_ids),
            "candidate_count": len(observation.candidates),
        }
        if metadata:
            record.update(metadata)
        self.records.append(record)

    def act(self, observation: AgentObservation) -> AgentAction:
        if self.pending_temporal_evidence_id is not None:
            evidence_id = self.pending_temporal_evidence_id
            self.pending_temporal_evidence_id = None
            if GeoToolName.TEMPORAL_CHANGE in observation.available_tools:
                action = self._tool_action(
                    observation, GeoToolName.TEMPORAL_CHANGE, evidence_id
                )
                self.processed_evidence_ids.add(evidence_id)
                self._record(
                    observation,
                    1.0,
                    admitted=True,
                    proposed_action="PROACTIVE_PAIR_PENDING",
                    executed_action=action.key,
                    source="proactive_temporal_pair",
                )
                return action

        features = structured_tool_need_features(observation)
        probability = float(self.gate.predict_proba(features[None, :])[0, 1])
        if self.proactive_pair:
            evidence_id = (
                observation.selected_evidence_ids[-1]
                if observation.selected_evidence_ids
                else None
            )
            eligible = (
                evidence_id is not None
                and evidence_id not in self.processed_evidence_ids
                and observation.remaining_budget + 1e-8 >= self.pair_cost
                and GeoToolName.IMAGE_QUALITY in observation.available_tools
                and GeoToolName.TEMPORAL_CHANGE in observation.available_tools
            )
            if eligible and probability >= self.threshold:
                assert evidence_id is not None
                action = self._tool_action(
                    observation, GeoToolName.IMAGE_QUALITY, evidence_id
                )
                self.pending_temporal_evidence_id = evidence_id
                self._record(
                    observation,
                    probability,
                    admitted=True,
                    proposed_action="PROACTIVE_GATE",
                    executed_action=action.key,
                    source="proactive_quality_pair",
                )
                return action
            self._record(
                observation,
                probability,
                admitted=False,
                proposed_action="PROACTIVE_GATE_CHECK",
                executed_action="NO_PROACTIVE_CALL",
                source="proactive_gate_check",
                metadata={
                    "eligible": eligible,
                    "evidence_id": evidence_id,
                    "affordable": observation.remaining_budget + 1e-8
                    >= self.pair_cost,
                    "has_image_quality": GeoToolName.IMAGE_QUALITY
                    in observation.available_tools,
                    "has_temporal_change": GeoToolName.TEMPORAL_CHANGE
                    in observation.available_tools,
                    "already_processed": evidence_id in self.processed_evidence_ids
                    if evidence_id is not None
                    else False,
                },
            )

        proposed = self.policy.act(observation)
        if proposed.action != AgentActionType.USE_TOOL:
            return proposed
        admitted = probability >= self.threshold
        replacement = proposed
        if not admitted:
            masked = observation.model_copy(update={"available_tools": []})
            replacement = self.policy.act(masked)
            if replacement.action == AgentActionType.USE_TOOL:
                replacement = self._terminal_fallback(observation)
        self._record(
            observation,
            probability,
            admitted=admitted,
            proposed_action=proposed.key,
            executed_action=replacement.key,
            source="controller_proposal",
        )
        return replacement
