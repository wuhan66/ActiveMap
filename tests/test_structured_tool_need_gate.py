import json

import numpy as np

from activemap.agent.records import (
    AgentAction,
    AgentActionType,
    AgentBelief,
    AgentCandidate,
    AgentObservation,
)
from activemap.agent.tool_need_gate import (
    TOOL_NEED_FEATURE_NAMES,
    CalibratedToolNeedPolicy,
    structured_tool_need_features,
)
from activemap.geo_tools.records import GeoToolCall, GeoToolName
from activemap.models import EditOperation
from scripts.build_structured_tool_need_features import build


def _observation(*, tools: bool = True) -> AgentObservation:
    return AgentObservation(
        task_id="task-1",
        split="train",
        step=1,
        initial_budget=3.0,
        remaining_budget=1.5,
        spent_cost=1.5,
        selected_evidence_ids=["evidence-1", "evidence-2"],
        belief=AgentBelief(
            edit_probabilities=[0.7, 0.1, 0.1, 0.1],
            confidence=0.6,
            uncertainty=0.4,
            geometry_delta=[0.0] * 8,
            recommended_edit=EditOperation.KEEP,
        ),
        candidates=[
            AgentCandidate(
                evidence_id="evidence-3",
                cost=1.0,
                selector_score=0.25,
                features=[float(index) / 13.0 for index in range(13)],
            )
        ],
        terminal_score=0.2,
        available_tools=[GeoToolName.IMAGE_QUALITY] if tools else [],
    )


def _tool_action() -> AgentAction:
    return AgentAction(
        action=AgentActionType.USE_TOOL,
        tool_call=GeoToolCall(
            call_id="call-1",
            tool=GeoToolName.IMAGE_QUALITY,
            inputs={"evidence_id": "evidence-2"},
        ),
    )


class _Gate:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.asarray([[1.0 - self.probability, self.probability]])


class _Policy:
    def __init__(self) -> None:
        self.observations = []

    def act(self, observation: AgentObservation) -> AgentAction:
        self.observations.append(observation)
        if observation.available_tools:
            return _tool_action()
        return AgentAction(action=AgentActionType.REJECT)


def test_structured_features_are_finite_and_stable() -> None:
    features = structured_tool_need_features(_observation())

    assert features.shape == (len(TOOL_NEED_FEATURE_NAMES),)
    assert np.all(np.isfinite(features))


def test_calibrated_gate_vetoes_and_requests_non_tool_action() -> None:
    base = _Policy()
    records = []
    policy = CalibratedToolNeedPolicy(base, _Gate(0.2), threshold=0.5, records=records)

    action = policy.act(_observation())

    assert action.action == AgentActionType.REJECT
    assert len(base.observations) == 2
    assert base.observations[-1].available_tools == []
    assert records[0]["admitted"] is False


def test_calibrated_gate_admits_safe_tool_action() -> None:
    base = _Policy()
    policy = CalibratedToolNeedPolicy(base, _Gate(0.8), threshold=0.5)

    action = policy.act(_observation())

    assert action.action == AgentActionType.USE_TOOL
    assert len(base.observations) == 1


def test_proactive_gate_initiates_one_quality_temporal_pair() -> None:
    base = _Policy()
    records = []
    policy = CalibratedToolNeedPolicy(
        base,
        _Gate(0.8),
        threshold=0.5,
        records=records,
        proactive_pair=True,
        controller="test",
    )
    observation = _observation().model_copy(
        update={
            "available_tools": [
                GeoToolName.IMAGE_QUALITY,
                GeoToolName.TEMPORAL_CHANGE,
            ]
        }
    )

    quality = policy.act(observation)
    temporal = policy.act(observation.model_copy(update={"step": 2}))
    terminal = policy.act(
        observation.model_copy(update={"step": 3, "available_tools": []})
    )

    assert quality.tool_call is not None
    assert quality.tool_call.tool == GeoToolName.IMAGE_QUALITY
    assert temporal.tool_call is not None
    assert temporal.tool_call.tool == GeoToolName.TEMPORAL_CHANGE
    assert terminal.action == AgentActionType.REJECT
    assert [row["source"] for row in records[:2]] == [
        "proactive_quality_pair",
        "proactive_temporal_pair",
    ]
    assert records[0]["pre_tool_feature_names"] == TOOL_NEED_FEATURE_NAMES
    assert len(records[0]["pre_tool_features"]) == len(TOOL_NEED_FEATURE_NAMES)
    assert records[0]["remaining_budget"] == 1.5
    assert records[0]["predicted_edit"] == EditOperation.KEEP.value


def test_builder_deduplicates_natural_sft_states(tmp_path) -> None:
    observation = _observation()
    row = {
        "messages": [
            {"role": "user", "content": observation.model_dump_json(exclude_none=True)},
            {"role": "assistant", "content": _tool_action().model_dump_json(exclude_none=True)},
        ]
    }
    negative_observation = observation.model_copy(
        update={"task_id": "task-2", "available_tools": []}
    )
    negative = {
        "messages": [
            {
                "role": "user",
                "content": negative_observation.model_dump_json(exclude_none=True),
            },
            {
                "role": "assistant",
                "content": AgentAction(action=AgentActionType.REJECT).model_dump_json(),
            },
        ]
    }
    source = tmp_path / "sft.jsonl"
    source.write_text(
        "\n".join(json.dumps(item) for item in (row, row, negative)) + "\n",
        encoding="utf-8",
    )

    summary = build(source, tmp_path / "features", split="train")

    assert summary["records"] == 2
    assert summary["positive_count"] == 1
    assert summary["deduplicated_records"] == 1
    assert np.load(tmp_path / "features" / "features.npy").shape == (
        2,
        len(TOOL_NEED_FEATURE_NAMES),
    )
