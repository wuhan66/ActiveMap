"""Tool adapters used by the agentic maintenance environment."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

import numpy as np

from activemap.agent.records import AgentAction, AgentActionType, AgentBelief, AgentObservation
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import EditOperation
from activemap.selector_records import SelectorSample

ScoreFunction = Callable[[SelectorSample], np.ndarray]


class AgentPolicy(Protocol):
    def act(self, observation: AgentObservation) -> AgentAction: ...


class BeliefUpdater(Protocol):
    def fuse(self, selected_evidence_ids: Sequence[str]) -> AgentBelief: ...


class ToolBeliefUpdater(Protocol):
    def update(self, belief: AgentBelief, result: GeoToolResult) -> AgentBelief: ...


ToolBeliefAdapter = Callable[[AgentBelief, GeoToolResult], AgentBelief]


class RegisteredToolBeliefUpdater:
    """Apply explicit per-tool adapters; unhandled or failed calls are observational only."""

    def __init__(self, adapters: Mapping[GeoToolName, ToolBeliefAdapter]) -> None:
        self.adapters = dict(adapters)

    def update(self, belief: AgentBelief, result: GeoToolResult) -> AgentBelief:
        adapter = self.adapters.get(result.tool)
        if not result.success or adapter is None:
            return belief
        updated = adapter(belief, result)
        if not isinstance(updated, AgentBelief):
            raise TypeError("tool belief adapter must return AgentBelief")
        return updated


def belief_from_operation_prediction(prediction: dict[str, object]) -> AgentBelief:
    probabilities = np.asarray(prediction["edit_probabilities"], dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-8, None)
    probabilities /= probabilities.sum()
    geometry = np.asarray(prediction.get("geometry_delta", [0.0] * 8), dtype=np.float64)
    uncertainty = prediction.get("uncertainty")
    if uncertainty is None:
        uncertainty = float(
            -np.sum(probabilities * np.log(probabilities)) / np.log(len(EditOperation))
        )
    return AgentBelief(
        edit_probabilities=probabilities.tolist(),
        confidence=float(np.clip(prediction.get("confidence", probabilities.max()), 0.0, 1.0)),
        geometry_delta=geometry.tolist(),
        uncertainty=float(np.clip(uncertainty, 0.0, 1.0)),
        recommended_edit=(
            EditOperation(str(prediction["gated_edit"]))
            if prediction.get("gated_edit") is not None
            else None
        ),
    )


def belief_from_features(sample: SelectorSample) -> AgentBelief:
    probabilities = np.asarray(sample.hypothesis_features[:4], dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-7, None)
    probabilities /= probabilities.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)) / np.log(len(EditOperation)))
    return AgentBelief(
        edit_probabilities=probabilities.tolist(),
        confidence=float(np.clip(sample.hypothesis_features[13], 0.0, 1.0)),
        geometry_delta=[0.0] * 8,
        uncertainty=entropy,
    )


class CounterfactualBeliefUpdater:
    """Fuse cached per-evidence updater predictions with confidence weighting."""

    def __init__(self, sample: SelectorSample) -> None:
        payload = sample.metadata.get("evidence_predictions")
        if not isinstance(payload, dict):
            raise ValueError("selector sample lacks metadata.evidence_predictions")
        self.predictions = payload
        self.operation_update_threshold = sample.metadata.get("operation_update_threshold")

    def fuse(self, selected_evidence_ids: Sequence[str]) -> AgentBelief:
        if not selected_evidence_ids:
            raise ValueError("belief fusion requires at least one selected evidence item")
        missing = [item for item in selected_evidence_ids if item not in self.predictions]
        if missing:
            raise ValueError(f"missing updater predictions for evidence: {missing}")
        rows = [self.predictions[item] for item in selected_evidence_ids]
        probabilities = np.asarray([row["edit_probabilities"] for row in rows], dtype=np.float64)
        confidences = np.asarray([row["confidence"] for row in rows], dtype=np.float64)
        geometry = np.asarray([row["geometry_delta"] for row in rows], dtype=np.float64)
        weights = np.clip(confidences, 0.05, 1.0)
        fused = np.average(probabilities, axis=0, weights=weights)
        fused = np.clip(fused, 1e-7, None)
        fused /= fused.sum()
        fused_geometry = np.average(geometry, axis=0, weights=weights)
        entropy = float(-np.sum(fused * np.log(fused)) / np.log(len(EditOperation)))
        agreement = 1.0 - float(np.mean(np.argmax(probabilities, axis=1) != int(np.argmax(fused))))
        confidence = float(
            np.clip(0.5 * np.average(confidences, weights=weights) + 0.5 * agreement, 0.0, 1.0)
        )
        update_threshold = self.operation_update_threshold
        recommended_edit = None
        if update_threshold is not None:
            recommended_index = (
                int(np.argmax(fused[1:])) + 1
                if 1.0 - float(fused[0]) >= float(update_threshold)
                else 0
            )
            recommended_edit = list(EditOperation)[recommended_index]
        return AgentBelief(
            edit_probabilities=fused.tolist(),
            confidence=confidence,
            geometry_delta=fused_geometry.tolist(),
            uncertainty=entropy,
            recommended_edit=recommended_edit,
        )


class StaticBeliefUpdater:
    """No-belief-update ablation and compatibility adapter for legacy samples."""

    def __init__(self, sample: SelectorSample) -> None:
        self.belief = belief_from_features(sample)

    def fuse(self, selected_evidence_ids: Sequence[str]) -> AgentBelief:
        return self.belief


class GreedyAgentPolicy:
    """Executable non-LLM baseline over selector recommendations and STOP score."""

    def act(self, observation: AgentObservation) -> AgentAction:
        if observation.candidates:
            best = max(observation.candidates, key=lambda item: item.selector_score)
            stop_score = observation.terminal_score
            if stop_score is None or best.selector_score > stop_score:
                return AgentAction(action=AgentActionType.ACQUIRE, evidence_id=best.evidence_id)
        predicted_edit = observation.belief.predicted_edit
        if predicted_edit == EditOperation.KEEP:
            return AgentAction(action=AgentActionType.REJECT)
        return AgentAction(action=AgentActionType.COMMIT, edit=predicted_edit)


class TemporalPairClosurePolicy(GreedyAgentPolicy):
    """Acquire a near-tied temporal counterpart before a terminal map edit."""

    _TIMESTAMP_SUFFIX = re.compile(r"__y[0-9]{4}$")

    def __init__(self, *, closure_margin: float = 0.0) -> None:
        if closure_margin < 0.0:
            raise ValueError("closure_margin must be non-negative")
        self.closure_margin = float(closure_margin)

    @classmethod
    def _pair_key(cls, evidence_id: str) -> str | None:
        if not cls._TIMESTAMP_SUFFIX.search(evidence_id):
            return None
        return cls._TIMESTAMP_SUFFIX.sub("", evidence_id)

    def act(self, observation: AgentObservation) -> AgentAction:
        default = super().act(observation)
        if default.action == AgentActionType.ACQUIRE or observation.terminal_score is None:
            return default
        selected_keys = {
            key
            for evidence_id in observation.selected_evidence_ids
            if (key := self._pair_key(evidence_id)) is not None
        }
        if not selected_keys:
            return default
        counterparts = [
            candidate
            for candidate in observation.candidates
            if self._pair_key(candidate.evidence_id) in selected_keys
        ]
        if not counterparts:
            return default
        counterpart = max(counterparts, key=lambda item: item.selector_score)
        if counterpart.selector_score + self.closure_margin > observation.terminal_score:
            return AgentAction(action=AgentActionType.ACQUIRE, evidence_id=counterpart.evidence_id)
        return default


class UncertaintyAwareAgentPolicy:
    """Acquire evidence for uncertain beliefs, otherwise execute the calibrated edit."""

    def __init__(self, *, max_uncertainty: float = 0.55, min_confidence: float = 0.65) -> None:
        if not 0.0 <= max_uncertainty <= 1.0:
            raise ValueError("max_uncertainty must be between zero and one")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between zero and one")
        self.max_uncertainty = max_uncertainty
        self.min_confidence = min_confidence

    def act(self, observation: AgentObservation) -> AgentAction:
        belief = observation.belief
        needs_evidence = (
            belief.uncertainty > self.max_uncertainty
            or belief.confidence < self.min_confidence
        )
        if needs_evidence and observation.candidates:
            best = max(observation.candidates, key=lambda item: item.selector_score)
            should_acquire = (
                observation.terminal_score is None
                or best.selector_score > observation.terminal_score
            )
            if should_acquire:
                return AgentAction(action=AgentActionType.ACQUIRE, evidence_id=best.evidence_id)
        if belief.predicted_edit == EditOperation.KEEP:
            return AgentAction(action=AgentActionType.REJECT)
        return AgentAction(action=AgentActionType.COMMIT, edit=belief.predicted_edit)
