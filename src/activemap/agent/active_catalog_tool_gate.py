"""Leakage-free pre-tool features and selective tool policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from activemap.agent.records import AgentAction, AgentActionType, AgentBelief, AgentObservation
from activemap.agent.tool_belief_model import encode_belief
from activemap.geo_tools.records import GeoToolCall, GeoToolName
from activemap.selector_records import SelectorSample

PRE_TOOL_FEATURE_NAMES = tuple(
    [f"belief_{index}" for index in range(14)]
    + [f"candidate_{index}" for index in range(13)]
    + ["evidence_cost", "tool_cost", "remaining_budget_fraction",
       "spent_budget_fraction", "pre_acquisition_step"]
)


class LinearProbabilityGate:
    """Portable StandardScaler plus binary logistic-regression inference."""

    def __init__(
        self,
        mean: list[float],
        scale: list[float],
        coefficient: list[float],
        intercept: float,
    ) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.coefficient = np.asarray(coefficient, dtype=np.float64)
        self.intercept = float(intercept)
        if not (
            self.mean.shape
            == self.scale.shape
            == self.coefficient.shape
            == (53,)
        ):
            raise ValueError("semantic gate parameters must be 53-D")

    @classmethod
    def from_json(cls, path: str | Path) -> "LinearProbabilityGate":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != "linear-probability-gate-v1":
            raise ValueError("unsupported semantic gate schema")
        return cls(
            payload["mean"],
            payload["scale"],
            payload["coefficient"],
            payload["intercept"],
        )

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        logits = ((values - self.mean) / self.scale) @ self.coefficient
        logits = np.clip(logits + self.intercept, -60.0, 60.0)
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - positive, positive])


class BeliefUncertaintyGate:
    """Outcome-free tool gate calibrated from train-split belief uncertainty."""

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(PRE_TOOL_FEATURE_NAMES):
            raise ValueError("uncertainty gate requires PRE_TOOL feature rows")
        positive = np.clip(values[:, 5], 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])


def policy_relative_semantic_gate_features(sample: SelectorSample) -> np.ndarray:
    evidence = np.asarray(sample.evidence_features, dtype=np.float32)
    if evidence.ndim != 2 or evidence.shape[1] != 13:
        raise ValueError("semantic gate requires 13-D evidence features")
    if len(evidence):
        mean = evidence.mean(axis=0)
        maximum = evidence.max(axis=0)
        costs = np.asarray(sample.evidence_costs, dtype=np.float32)
        cost_summary = [float(costs.min()), float(costs.mean())]
    else:
        mean = np.zeros(13, dtype=np.float32)
        maximum = np.zeros(13, dtype=np.float32)
        cost_summary = [0.0, 0.0]
    features = np.asarray(
        sample.hypothesis_features
        + sample.state_features
        + mean.tolist()
        + maximum.tolist()
        + cost_summary
        + [float(len(sample.evidence_ids))],
        dtype=np.float32,
    )
    if features.shape != (53,) or not np.all(np.isfinite(features)):
        raise ValueError("invalid policy-relative semantic gate features")
    return features


def pre_tool_features(
    belief: AgentBelief,
    candidate_features: list[float],
    *,
    evidence_cost: float,
    tool_cost: float,
    initial_budget: float,
    remaining_budget: float,
    spent_cost: float,
    step: int,
) -> np.ndarray:
    if len(candidate_features) != 13 or initial_budget <= 0.0:
        raise ValueError("invalid pre-tool context")
    values = np.asarray(
        encode_belief(belief) + candidate_features + [
            evidence_cost, tool_cost, remaining_budget / initial_budget,
            spent_cost / initial_budget, float(step),
        ],
        dtype=np.float32,
    )
    if values.shape != (len(PRE_TOOL_FEATURE_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError("invalid PRE_TOOL feature vector")
    return values


class SelectorPolicy(Protocol):
    def act(self, observation: AgentObservation) -> AgentAction: ...


class SelectiveToolPolicy:
    """Interleave selector actions with a quality/temporal tool pair."""

    def __init__(self, selector: SelectorPolicy, *, mode: str, gate: Any | None,
                 threshold: float, tool_cost: float = 0.18) -> None:
        if mode not in {"forced", "selective"}:
            raise ValueError("tool mode must be forced or selective")
        if mode == "selective" and gate is None:
            raise ValueError("selective mode requires a gate")
        self.selector, self.mode, self.gate = selector, mode, gate
        self.threshold, self.tool_cost = threshold, tool_cost
        self.pending: dict[str, Any] | None = None
        self.phase: str | None = None
        self.events: list[dict[str, Any]] = []

    def _tool_action(self, observation: AgentObservation, tool: GeoToolName) -> AgentAction:
        assert self.pending is not None
        evidence_id = str(self.pending["evidence_id"])
        identity = f"{observation.task_id}|{evidence_id}|{tool.value}|{observation.step}"
        return AgentAction(
            action=AgentActionType.USE_TOOL,
            tool_call=GeoToolCall(
                call_id=f"ac-{hashlib.sha256(identity.encode()).hexdigest()[:20]}",
                tool=tool,
                inputs={"evidence_id": evidence_id},
            ),
        )

    def act(self, observation: AgentObservation) -> AgentAction:
        if self.phase == "temporal":
            self.phase = None
            action = self._tool_action(observation, GeoToolName.TEMPORAL_CHANGE)
            self.pending = None
            self.events.append({"step": observation.step, "stage": "TOOL",
                                "decision": "TEMPORAL_CHANGE", "valid_action": True})
            return action
        if self.phase == "quality":
            assert self.pending is not None
            affordable = (
                observation.remaining_budget + 1e-8 >= self.tool_cost
                and GeoToolName.IMAGE_QUALITY in observation.available_tools
                and GeoToolName.TEMPORAL_CHANGE in observation.available_tools
            )
            features = pre_tool_features(
                observation.belief, self.pending["candidate_features"],
                evidence_cost=self.pending["evidence_cost"], tool_cost=self.tool_cost,
                initial_budget=self.pending["initial_budget"],
                remaining_budget=self.pending["remaining_budget"],
                spent_cost=self.pending["spent_cost"], step=self.pending["step"],
            )
            probability = 1.0 if self.mode == "forced" else float(
                self.gate.predict_proba(features[None, :])[0, 1]
            )
            call = affordable and probability >= self.threshold
            self.events.append({"step": observation.step, "stage": "TOOL_GATE",
                                "decision": "CALL" if call else "SKIP",
                                "call_probability": probability, "valid_action": True})
            if call:
                self.phase = "temporal"
                return self._tool_action(observation, GeoToolName.IMAGE_QUALITY)
            self.phase = None
            self.pending = None
        action = self.selector.act(observation)
        selector_events = getattr(self.selector, "events", None)
        if selector_events:
            self.events.append(selector_events[-1])
        if action.action == AgentActionType.ACQUIRE:
            candidate = next(item for item in observation.candidates
                             if item.evidence_id == action.evidence_id)
            self.pending = {
                "evidence_id": action.evidence_id,
                "candidate_features": candidate.features,
                "evidence_cost": candidate.cost,
                "initial_budget": observation.initial_budget,
                "remaining_budget": observation.remaining_budget,
                "spent_cost": observation.spent_cost,
                "step": observation.step,
            }
            self.phase = "quality"
        return action


class ForcedSemanticToolPolicy:
    """Invoke one map-relative semantic tool after each acquisition."""

    def __init__(self, selector: SelectorPolicy, *, tool_cost: float = 0.75) -> None:
        self.selector = selector
        self.tool_cost = float(tool_cost)
        self.pending_evidence_id: str | None = None
        self.events: list[dict[str, Any]] = []

    def act(self, observation: AgentObservation) -> AgentAction:
        if self.pending_evidence_id is not None:
            evidence_id = self.pending_evidence_id
            self.pending_evidence_id = None
            affordable = (
                observation.remaining_budget + 1e-8 >= self.tool_cost
                and GeoToolName.RASTER_SEGMENT in observation.available_tools
            )
            self.events.append(
                {
                    "step": observation.step,
                    "stage": "SEMANTIC_TOOL_GATE",
                    "decision": "CALL" if affordable else "SKIP",
                    "valid_action": True,
                }
            )
            if affordable:
                return AgentAction(
                    action=AgentActionType.USE_TOOL,
                    tool_call=GeoToolCall(
                        call_id=(
                            "semantic-"
                            + hashlib.sha256(
                                (
                                    f"{observation.task_id}|{evidence_id}|"
                                    f"{observation.step}"
                                ).encode()
                            ).hexdigest()[:20]
                        ),
                        tool=GeoToolName.RASTER_SEGMENT,
                        inputs={"evidence_id": evidence_id},
                    ),
                )
        action = self.selector.act(observation)
        if action.action == AgentActionType.ACQUIRE:
            self.pending_evidence_id = action.evidence_id
        return action


class ForcedInitialSemanticToolPolicy:
    """Invoke semantic evidence once on the registered initial observation."""

    def __init__(self, selector: SelectorPolicy, *, tool_cost: float = 0.75) -> None:
        self.selector = selector
        self.tool_cost = float(tool_cost)
        self.attempted = False
        self.events: list[dict[str, Any]] = []

    def act(self, observation: AgentObservation) -> AgentAction:
        if not self.attempted:
            self.attempted = True
            affordable = (
                bool(observation.selected_evidence_ids)
                and observation.remaining_budget + 1e-8 >= self.tool_cost
                and GeoToolName.RASTER_SEGMENT in observation.available_tools
            )
            self.events.append(
                {
                    "step": observation.step,
                    "stage": "INITIAL_SEMANTIC_TOOL",
                    "decision": "CALL" if affordable else "SKIP",
                    "valid_action": True,
                }
            )
            if affordable:
                evidence_id = observation.selected_evidence_ids[-1]
                identity = (
                    f"{observation.task_id}|{evidence_id}|initial-semantic"
                )
                return AgentAction(
                    action=AgentActionType.USE_TOOL,
                    tool_call=GeoToolCall(
                        call_id=(
                            "semantic-initial-"
                            + hashlib.sha256(identity.encode()).hexdigest()[:20]
                        ),
                        tool=GeoToolName.RASTER_SEGMENT,
                        inputs={"evidence_id": evidence_id},
                    ),
                )
        return self.selector.act(observation)


class SelectiveInitialSemanticToolPolicy:
    """Use a train-calibrated gate before the initial semantic call."""

    def __init__(
        self,
        selector: SelectorPolicy,
        current_sample: Any,
        gate: Any,
        *,
        threshold: float,
        tool_cost: float = 0.75,
    ) -> None:
        self.selector = selector
        self.current_sample = current_sample
        self.gate = gate
        self.threshold = float(threshold)
        self.tool_cost = float(tool_cost)
        self.attempted = False
        self.events: list[dict[str, Any]] = []

    def act(self, observation: AgentObservation) -> AgentAction:
        if not self.attempted:
            self.attempted = True
            features = policy_relative_semantic_gate_features(
                self.current_sample()
            )
            probability = float(self.gate.predict_proba(features[None, :])[0, 1])
            call = (
                probability >= self.threshold
                and bool(observation.selected_evidence_ids)
                and observation.remaining_budget + 1e-8 >= self.tool_cost
                and GeoToolName.RASTER_SEGMENT in observation.available_tools
            )
            self.events.append(
                {
                    "step": observation.step,
                    "stage": "INITIAL_SEMANTIC_GATE",
                    "decision": "CALL" if call else "SKIP",
                    "call_probability": probability,
                    "valid_action": True,
                }
            )
            if call:
                evidence_id = observation.selected_evidence_ids[-1]
                identity = (
                    f"{observation.task_id}|{evidence_id}|selective-semantic"
                )
                return AgentAction(
                    action=AgentActionType.USE_TOOL,
                    tool_call=GeoToolCall(
                        call_id=(
                            "semantic-selective-"
                            + hashlib.sha256(identity.encode()).hexdigest()[:20]
                        ),
                        tool=GeoToolName.RASTER_SEGMENT,
                        inputs={"evidence_id": evidence_id},
                    ),
                )
        return self.selector.act(observation)
