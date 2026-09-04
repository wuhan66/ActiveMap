import numpy as np

from activemap.agent.active_catalog_tool_gate import (
    BeliefUncertaintyGate,
    ForcedInitialSemanticToolPolicy,
    ForcedSemanticToolPolicy,
    PRE_TOOL_FEATURE_NAMES,
    SelectiveInitialSemanticToolPolicy,
    SelectiveToolPolicy,
)
from activemap.agent.records import (
    AgentAction,
    AgentActionType,
    AgentBelief,
    AgentCandidate,
    AgentObservation,
)
from activemap.geo_tools.records import GeoToolName


def test_belief_uncertainty_gate_reads_only_public_uncertainty():
    features = np.zeros((2, len(PRE_TOOL_FEATURE_NAMES)), dtype=np.float32)
    features[:, 5] = [0.2, 0.8]
    probabilities = BeliefUncertaintyGate().predict_proba(features)
    np.testing.assert_allclose(probabilities[:, 1], [0.2, 0.8])


def _belief():
    return AgentBelief(
        edit_probabilities=[0.7, 0.1, 0.1, 0.1],
        confidence=0.8,
        uncertainty=0.3,
    )


def _observation(step, selected, *, candidates=None, remaining=3.0):
    return AgentObservation(
        task_id="task",
        split="val",
        step=step,
        initial_budget=3.0,
        remaining_budget=remaining,
        spent_cost=3.0 - remaining,
        selected_evidence_ids=selected,
        belief=_belief(),
        candidates=candidates or [],
        available_tools=[GeoToolName.IMAGE_QUALITY, GeoToolName.TEMPORAL_CHANGE],
    )


class _Selector:
    def __init__(self):
        self.calls = 0
        self.events = []

    def act(self, observation):
        self.calls += 1
        self.events.append({"step": observation.step, "valid_action": True})
        if observation.candidates:
            return AgentAction(action=AgentActionType.ACQUIRE, evidence_id="evidence")
        return AgentAction(action=AgentActionType.REJECT)


class _Gate:
    def __init__(self, probability):
        self.probability = probability
        self.feature_shapes = []

    def predict_proba(self, features):
        self.feature_shapes.append(features.shape)
        return np.asarray([[1.0 - self.probability, self.probability]])


def test_selective_policy_executes_paired_tools_then_returns_to_selector():
    selector, gate = _Selector(), _Gate(0.9)
    policy = SelectiveToolPolicy(selector, mode="selective", gate=gate, threshold=0.5)
    candidate = AgentCandidate(
        evidence_id="evidence", cost=1.0, selector_score=1.0, features=[0.1] * 13
    )
    assert policy.act(_observation(0, ["anchor"], candidates=[candidate])).action == AgentActionType.ACQUIRE
    quality = policy.act(_observation(1, ["anchor", "evidence"], remaining=2.0))
    temporal = policy.act(_observation(2, ["anchor", "evidence"], remaining=1.97))
    terminal = policy.act(_observation(3, ["anchor", "evidence"], remaining=1.82))
    assert quality.tool_call.tool == GeoToolName.IMAGE_QUALITY
    assert temporal.tool_call.tool == GeoToolName.TEMPORAL_CHANGE
    assert terminal.action == AgentActionType.REJECT
    assert gate.feature_shapes == [(1, 32)]


def test_selective_policy_skip_avoids_tools():
    selector, gate = _Selector(), _Gate(0.1)
    policy = SelectiveToolPolicy(selector, mode="selective", gate=gate, threshold=0.5)
    candidate = AgentCandidate(
        evidence_id="evidence", cost=1.0, selector_score=1.0, features=[0.1] * 13
    )
    policy.act(_observation(0, ["anchor"], candidates=[candidate]))
    action = policy.act(_observation(1, ["anchor", "evidence"], remaining=2.0))
    assert action.action == AgentActionType.REJECT
    assert selector.calls == 2


def test_forced_semantic_policy_calls_one_tool_after_acquisition():
    selector = _Selector()
    policy = ForcedSemanticToolPolicy(selector)
    candidate = AgentCandidate(
        evidence_id="evidence", cost=1.0, selector_score=1.0, features=[0.1] * 13
    )
    observation = _observation(0, ["anchor"], candidates=[candidate])
    observation = observation.model_copy(
        update={"available_tools": [GeoToolName.RASTER_SEGMENT]}
    )
    assert policy.act(observation).action == AgentActionType.ACQUIRE
    after_acquire = _observation(1, ["anchor", "evidence"], remaining=2.0)
    after_acquire = after_acquire.model_copy(
        update={"available_tools": [GeoToolName.RASTER_SEGMENT]}
    )
    tool_action = policy.act(after_acquire)
    assert tool_action.action == AgentActionType.USE_TOOL
    assert tool_action.tool_call.tool == GeoToolName.RASTER_SEGMENT
    terminal = policy.act(after_acquire.model_copy(update={"step": 2}))
    assert terminal.action == AgentActionType.REJECT


def test_forced_initial_semantic_policy_calls_before_selector():
    selector = _Selector()
    policy = ForcedInitialSemanticToolPolicy(selector)
    observation = _observation(0, ["anchor"], remaining=3.0).model_copy(
        update={"available_tools": [GeoToolName.RASTER_SEGMENT]}
    )
    tool_action = policy.act(observation)
    assert tool_action.action == AgentActionType.USE_TOOL
    assert tool_action.tool_call.inputs["evidence_id"] == "anchor"
    assert policy.act(observation.model_copy(update={"step": 1})).action == (
        AgentActionType.REJECT
    )


class _Sample:
    hypothesis_features = [0.0] * 16
    state_features = [0.0] * 8
    evidence_features = [[0.0] * 13]
    evidence_costs = [1.0]
    evidence_ids = ["anchor"]


def test_selective_initial_semantic_policy_calls_then_delegates():
    selector, gate = _Selector(), _Gate(0.9)
    policy = SelectiveInitialSemanticToolPolicy(
        selector,
        lambda: _Sample(),
        gate,
        threshold=0.5,
    )
    observation = _observation(0, ["anchor"], remaining=3.0).model_copy(
        update={"available_tools": [GeoToolName.RASTER_SEGMENT]}
    )
    tool_action = policy.act(observation)
    assert tool_action.action == AgentActionType.USE_TOOL
    assert tool_action.tool_call.inputs["evidence_id"] == "anchor"
    assert gate.feature_shapes == [(1, 53)]
    assert policy.act(observation.model_copy(update={"step": 1})).action == (
        AgentActionType.REJECT
    )


def test_selective_initial_semantic_policy_can_skip():
    selector, gate = _Selector(), _Gate(0.1)
    policy = SelectiveInitialSemanticToolPolicy(
        selector,
        lambda: _Sample(),
        gate,
        threshold=0.5,
    )
    observation = _observation(0, ["anchor"], remaining=3.0).model_copy(
        update={"available_tools": [GeoToolName.RASTER_SEGMENT]}
    )
    assert policy.act(observation).action == AgentActionType.REJECT
    assert selector.calls == 1
