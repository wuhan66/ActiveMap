"""Deterministic, label-free evidence-acquisition baselines."""

from __future__ import annotations

import hashlib

import numpy as np

from activemap.agent.records import AgentAction, AgentActionType, AgentObservation
from activemap.models import EditOperation
from activemap.selector_records import SelectorSample


def _terminal_action(observation: AgentObservation) -> AgentAction:
    predicted = observation.belief.predicted_edit
    if predicted == EditOperation.KEEP:
        return AgentAction(action=AgentActionType.REJECT)
    return AgentAction(action=AgentActionType.COMMIT, edit=predicted)


class AcquireUntilBudgetPolicy:
    """Follow a ranking until no further evidence is affordable, then terminate."""

    def act(self, observation: AgentObservation) -> AgentAction:
        if observation.candidates:
            best = max(observation.candidates, key=lambda item: item.selector_score)
            return AgentAction(
                action=AgentActionType.ACQUIRE,
                evidence_id=best.evidence_id,
            )
        return _terminal_action(observation)


def stable_random_scores(sample: SelectorSample) -> np.ndarray:
    """Produce reproducible pseudo-random rankings without runtime RNG state."""

    scores = []
    for evidence_id in sample.evidence_ids:
        digest = hashlib.sha256(
            f"activemap-random-v1\0{sample.sample_id}\0{evidence_id}".encode()
        ).digest()
        scores.append(int.from_bytes(digest[:8], "big") / float(2**64 - 1))
    return np.asarray(scores, dtype=np.float64)


def cheapest_scores(sample: SelectorSample) -> np.ndarray:
    return -np.asarray(sample.evidence_costs, dtype=np.float64)


def acquire_all_scores(sample: SelectorSample) -> np.ndarray:
    """Use catalog order while acquiring every affordable item under budget."""

    return -np.arange(len(sample.evidence_ids), dtype=np.float64)


def quality_first_scores(sample: SelectorSample) -> np.ndarray:
    features = np.asarray(sample.evidence_features, dtype=np.float64)
    clarity = np.clip(features[:, 0], 0.0, 1.0)
    probabilities = _candidate_probabilities(sample)
    if probabilities is None:
        return clarity
    return clarity * probabilities.max(axis=1)


def _candidate_probabilities(sample: SelectorSample) -> np.ndarray | None:
    predictions = sample.metadata.get("evidence_predictions")
    if not isinstance(predictions, dict):
        return None
    rows = []
    for evidence_id in sample.evidence_ids:
        prediction = predictions.get(evidence_id)
        if not isinstance(prediction, dict):
            return None
        probabilities = np.asarray(prediction.get("edit_probabilities"), dtype=np.float64)
        if probabilities.shape != (4,) or not np.all(np.isfinite(probabilities)):
            return None
        total = float(probabilities.sum())
        if total <= 0:
            return None
        rows.append(probabilities / total)
    return np.stack(rows)


def uncertainty_scores(sample: SelectorSample) -> np.ndarray:
    probabilities = _candidate_probabilities(sample)
    if probabilities is not None:
        clipped = np.clip(probabilities, 1e-12, 1.0)
        return -np.sum(clipped * np.log(clipped), axis=1) / np.log(probabilities.shape[1])
    features = np.asarray(sample.evidence_features, dtype=np.float64)
    return features[:, 11]


def mapex_scores(sample: SelectorSample) -> np.ndarray:
    """MapEx-style observable belief-disagreement-per-cost proxy."""

    features = np.asarray(sample.evidence_features, dtype=np.float64)
    clarity = np.clip(features[:, 0], 0.0, 1.0)
    costs = np.maximum(np.asarray(sample.evidence_costs, dtype=np.float64), 1e-8)
    probabilities = _candidate_probabilities(sample)
    if probabilities is None:
        uncertainty = np.clip(features[:, 11], 0.0, 1.0)
        return uncertainty * (0.25 + 0.75 * clarity) / costs

    belief = np.asarray(sample.hypothesis_features[:4], dtype=np.float64)
    belief = np.clip(belief / max(float(belief.sum()), 1e-12), 1e-12, 1.0)
    candidates = np.clip(probabilities, 1e-12, 1.0)
    mixture = 0.5 * (candidates + belief[None, :])
    divergence = 0.5 * (
        np.sum(candidates * np.log(candidates / mixture), axis=1)
        + np.sum(belief[None, :] * np.log(belief[None, :] / mixture), axis=1)
    ) / np.log(2.0)
    return divergence * (0.25 + 0.75 * clarity) / costs


def greedy_utility_scores(sample: SelectorSample) -> np.ndarray:
    """Risk-aware hand-crafted utility plus a fixed observable STOP score."""

    features = np.asarray(sample.evidence_features, dtype=np.float64)
    costs = np.asarray(sample.evidence_costs, dtype=np.float64)
    normalized_cost = costs / max(float(costs.max()), 1e-8)
    quality = np.clip(features[:, :2].mean(axis=1), 0.0, 1.0)
    uncertainty = np.clip(features[:, 11], 0.0, 1.0)
    risk = np.asarray(sample.false_edit_risks, dtype=np.float64)
    candidate_scores = (
        0.45 * quality
        + 0.35 * uncertainty
        + 0.20 * (1.0 - normalized_cost)
        - 0.50 * risk
    )
    belief_confidence = float(np.clip(sample.hypothesis_features[13], 0.0, 1.0))
    stop_score = 0.20 + 0.25 * belief_confidence
    return np.append(candidate_scores, stop_score)
