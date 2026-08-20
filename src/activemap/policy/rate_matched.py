"""Train-calibrated sparse heuristic policies for matched-rate validation.

These policies deliberately operate on the same frozen candidate interpretation
catalog as ActiveMap. They are controls for downstream evidence use, not claims
about avoiding raw candidate-image inference.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from activemap.agent.active_catalog import terminal_agent_action
from activemap.agent.identifiers import public_evidence_id
from activemap.agent.records import AgentAction, AgentActionType, AgentObservation
from activemap.selector_records import SelectorSample

RateMatchedMode = Literal[
    "random",
    "uncertainty",
    "low_confidence",
    "clear_per_cost",
    "cheap_positive",
    "minimum_entropy",
]

RATE_MATCHED_MODES: tuple[RateMatchedMode, ...] = (
    "random",
    "uncertainty",
    "low_confidence",
    "clear_per_cost",
    "cheap_positive",
    "minimum_entropy",
)


def _hash_unit_interval(*parts: object) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _operation_entropy(prediction: dict[str, object]) -> float:
    probabilities = np.asarray(prediction["edit_probabilities"], dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-8, None)
    probabilities /= probabilities.sum()
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(len(probabilities)))


def _prediction(sample: SelectorSample, evidence_id: str) -> dict[str, object] | None:
    payload = sample.metadata.get("evidence_predictions")
    if not isinstance(payload, dict):
        return None
    row = payload.get(evidence_id)
    return row if isinstance(row, dict) else None


def candidate_score(sample: SelectorSample, mode: RateMatchedMode, index: int) -> float:
    """Return an observable candidate-ranking score without oracle fields."""

    features = sample.evidence_features[index]
    cost = max(float(sample.evidence_costs[index]), 1e-8)
    evidence_id = sample.evidence_ids[index]
    if mode == "random":
        return _hash_unit_interval(sample.sample_id, evidence_id, "candidate")
    if mode == "uncertainty":
        return float(features[11]) / cost
    if mode == "clear_per_cost":
        return float(features[0]) / cost
    prediction = _prediction(sample, evidence_id)
    if prediction is None:
        # A missing front-end prediction is not silently replaced by target data.
        return float("-inf")
    if mode == "low_confidence":
        return (1.0 - float(np.clip(prediction.get("confidence", 0.0), 0.0, 1.0))) / cost
    if mode == "cheap_positive":
        probabilities = np.asarray(prediction["edit_probabilities"], dtype=np.float64)
        change_probability = float(np.clip(1.0 - probabilities[0], 0.0, 1.0))
        confidence = float(np.clip(prediction.get("confidence", 0.0), 0.0, 1.0))
        return change_probability * confidence / cost
    if mode == "minimum_entropy":
        return (1.0 - _operation_entropy(prediction)) * float(features[0]) / cost
    raise ValueError(f"unsupported matched-rate mode: {mode}")


def admission_score(sample: SelectorSample, mode: RateMatchedMode) -> float:
    """Return a scalar used only for the one-step ACQUIRE/STOP calibration."""

    if mode == "random":
        return _hash_unit_interval(sample.sample_id, "admit")
    if mode == "uncertainty":
        return float(np.clip(sample.hypothesis_features[12], 0.0, 1.0))
    if mode == "low_confidence":
        return 1.0 - float(np.clip(sample.hypothesis_features[13], 0.0, 1.0))
    scores = [candidate_score(sample, mode, index) for index in range(len(sample.evidence_ids))]
    return float(max(scores, default=float("-inf")))


@dataclass(frozen=True)
class RateCalibration:
    mode: RateMatchedMode
    target_call_rate: float
    target_call_count: int
    sample_count: int
    threshold: float
    tie_hash_threshold: float
    tie_sample_id_threshold: str
    strict_count: int
    tied_count: int

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def fit_rate_calibration(
    samples: list[SelectorSample], mode: RateMatchedMode, target_call_rate: float
) -> RateCalibration:
    """Fit a threshold with an exact, deterministic training tie break.

    The tie rule is learned only from train sample identifiers.  It realizes
    exactly ``target_call_count`` admissions on the calibration states while
    still transferring as a fixed score-and-hash rule to unseen validation
    states.  It deliberately never inspects oracle utilities or target edits.
    """

    if not samples:
        raise ValueError("rate calibration requires at least one sample")
    if not 0.0 <= target_call_rate <= 1.0:
        raise ValueError("target_call_rate must be between zero and one")
    target_count = int(round(target_call_rate * len(samples)))
    scores = np.asarray([admission_score(sample, mode) for sample in samples], dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"non-finite observable score for mode={mode}")
    if target_count <= 0:
        return RateCalibration(
            mode=mode,
            target_call_rate=target_call_rate,
            target_call_count=0,
            sample_count=len(samples),
            threshold=float("inf"),
            tie_hash_threshold=0.0,
            tie_sample_id_threshold="",
            strict_count=0,
            tied_count=0,
        )
    if target_count >= len(samples):
        return RateCalibration(
            mode=mode,
            target_call_rate=target_call_rate,
            target_call_count=len(samples),
            sample_count=len(samples),
            threshold=float("-inf"),
            tie_hash_threshold=1.0,
            tie_sample_id_threshold="~",
            strict_count=len(samples),
            tied_count=0,
        )
    threshold = float(np.sort(scores)[-target_count])
    strict_count = int(np.count_nonzero(scores > threshold))
    tied_count = int(np.count_nonzero(scores == threshold))
    tie_needed = target_count - strict_count
    tied_keys = sorted(
        (_hash_unit_interval(sample.sample_id, mode, "tie"), sample.sample_id)
        for sample, score in zip(samples, scores, strict=True)
        if score == threshold
    )
    if len(tied_keys) != tied_count:
        raise RuntimeError("rate-calibration tie accounting mismatch")
    return RateCalibration(
        mode=mode,
        target_call_rate=target_call_rate,
        target_call_count=target_count,
        sample_count=len(samples),
        threshold=threshold,
        tie_hash_threshold=float(tied_keys[tie_needed - 1][0]),
        tie_sample_id_threshold=tied_keys[tie_needed - 1][1],
        strict_count=strict_count,
        tied_count=tied_count,
    )


def admits(sample: SelectorSample, calibration: RateCalibration) -> bool:
    score = admission_score(sample, calibration.mode)
    if score > calibration.threshold:
        return True
    if score < calibration.threshold:
        return False
    if calibration.tie_hash_threshold >= 1.0:
        return True
    return (
        _hash_unit_interval(sample.sample_id, calibration.mode, "tie"),
        sample.sample_id,
    ) <= (calibration.tie_hash_threshold, calibration.tie_sample_id_threshold)


class RateMatchedSingleAcquirePolicy:
    """Admit at most one observable candidate, then execute the current belief."""

    def __init__(self, sample: SelectorSample, calibration: RateCalibration) -> None:
        self.sample = sample
        self.calibration = calibration
        self._candidate_indices = {
            public_evidence_id(item): index
            for index, item in enumerate(sample.evidence_ids)
        }

    def act(self, observation: AgentObservation) -> AgentAction:
        if not observation.selected_evidence_ids and observation.candidates and admits(
            self.sample, self.calibration
        ):
            available = [
                candidate
                for candidate in observation.candidates
                if candidate.evidence_id in self._candidate_indices
            ]
            if available:
                chosen = max(
                    available,
                    key=lambda candidate: (
                        candidate_score(
                            self.sample,
                            self.calibration.mode,
                            self._candidate_indices[candidate.evidence_id],
                        ),
                        candidate.evidence_id,
                    ),
                )
                return AgentAction(
                    action=AgentActionType.ACQUIRE,
                    evidence_id=chosen.evidence_id,
                )
        return terminal_agent_action(observation)


def achieved_rate(samples: list[SelectorSample], calibration: RateCalibration) -> float:
    if not samples:
        raise ValueError("rate requires at least one sample")
    return sum(admits(sample, calibration) for sample in samples) / len(samples)
