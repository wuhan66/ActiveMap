import random
from pathlib import Path

import pytest

from activemap.agent.active_catalog import (
    active_catalog_state_from_observation,
    terminal_agent_action,
)
from activemap.agent.identifiers import public_evidence_id
from activemap.agent.records import AgentBelief, AgentCandidate, AgentObservation
from activemap.models import (
    CandidateHypothesis,
    EditOperation,
    EditRecord,
    EpisodeRecord,
    EvidenceItem,
    GeoJSONGeometry,
)
from activemap.selector_records import SelectorSample
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from scripts.evaluate_active_catalog_closed_loop import (
    AlwaysStopPolicy,
    EpsilonExploreSelector,
    QwenGateRankerPolicy,
    QwenGeoMMAgentPolicy,
    QwenPlanThenExecutePolicy,
    QwenReactPolicy,
    QwenSenseSearchPolicy,
    build_model_tool_registry,
    metrics,
    remap_asset_path,
)


def episode():
    geometry = GeoJSONGeometry(
        type="Polygon",
        coordinates=[[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    )
    return EpisodeRecord(
        episode_id="train-RESHAPE-a-0",
        aoi_id="a",
        split="val",
        source_dataset="fixture",
        map_before="/unused/before.geojson",
        target_map="/unused/after.geojson",
        hypothesis=CandidateHypothesis(
            op=EditOperation.RESHAPE,
            object_id="object",
            geometry=geometry,
            source="fixture",
        ),
        gt_edit=EditRecord(
            op=EditOperation.RESHAPE,
            object_id="object",
            geometry=geometry,
        ),
        evidence_catalog=[
            EvidenceItem(
                evidence_id="e-0",
                timestamp="2026_01",
                region=(0, 0, 16, 16),
                scale=1,
                image_path="/unused/image.tif",
                clear_fraction=1.0,
                cost=1.0,
            )
        ],
        is_synthetic=False,
        derivation_version="fixture-v1",
    )


def sample():
    return SelectorSample(
        sample_id="sample",
        split="val",
        edit_type=EditOperation.RESHAPE,
        hypothesis_features=[0.0] * 13 + [0.7, 0.0, 0.0],
        state_features=[1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.0],
        evidence_ids=["e-0", "e-1"],
        evidence_features=[
            [1.0, 0.0, -0.5, 0.5, 1.0, 0.0, 0.0, 0.5, 0.5, 0.1, 0.1, 0.2, 0.5],
            [0.8, 0.2, 0.5, 0.5, 0.0, 1.0, 0.0, 0.5, 0.5, 0.2, 0.2, 0.2, 1.0],
        ],
        evidence_costs=[1.0, 2.0],
        false_edit_risks=[0.0, 0.0],
        oracle_utilities=[-0.1, 0.4],
        stop_utility=0.0,
        metadata={
            "source_episode": "train-RESHAPE-a-0",
            "selected_evidence_ids": ["e-0"],
            "budget": 3.0,
            "aoi_id": "a",
            "gt_edit": "RESHAPE",
        },
    )


def observation(edit=EditOperation.RESHAPE):
    probabilities = [0.1, 0.1, 0.1, 0.7]
    if edit == EditOperation.KEEP:
        probabilities = [0.8, 0.1, 0.05, 0.05]
    return AgentObservation(
        task_id="task",
        split="val",
        step=1,
        initial_budget=3.0,
        remaining_budget=2.0,
        spent_cost=1.0,
        selected_evidence_ids=[public_evidence_id("e-0")],
        belief=AgentBelief(
            edit_probabilities=probabilities,
            confidence=0.8,
            uncertainty=0.2,
            recommended_edit=edit,
        ),
        candidates=[
            AgentCandidate(
                evidence_id=public_evidence_id("e-1"),
                cost=2.0,
                selector_score=1.0,
                features=[0.0] * 13,
            )
        ],
    )


def test_recurrent_state_uses_updated_belief_and_budget():
    current_episode = episode()
    current_episode.evidence_catalog.append(
        current_episode.evidence_catalog[0].model_copy(
            update={"evidence_id": "e-1", "timestamp": "2026_02", "scale": 2}
        )
    )
    current_sample = sample()
    state = active_catalog_state_from_observation(
        observation(), current_sample, current_episode, policy_snapshot="snapshot"
    )
    assert state["budget"]["remaining"] == 2.0
    assert state["belief"]["edit_probabilities"] == [0.1, 0.1, 0.1, 0.7]
    assert state["candidate_evidence"][0]["evidence_id"] == public_evidence_id("e-1")


def test_stop_executes_current_belief_edit():
    assert terminal_agent_action(observation()).action.value == "COMMIT"
    assert terminal_agent_action(observation(EditOperation.KEEP)).action.value == "REJECT"


def test_epsilon_exploration_executes_and_records_counterfactual_action():
    class StopSelector:
        def __init__(self):
            self.events = []

        def act(self, current):
            action = terminal_agent_action(current)
            self.events.append({
                "valid_action": True,
                "selector_action": {"stage": "SELECT", "selection": "STOP"},
                "executed_action": action.model_dump(mode="json", exclude_none=True),
            })
            return action

    policy = EpsilonExploreSelector(
        StopSelector(), epsilon=1.0, rng=random.Random(7)
    )
    action = policy.act(observation())
    assert action.action.value == "ACQUIRE"
    assert policy.events[-1]["exploration_override"] is True
    assert policy.events[-1]["model_selector_action"]["selection"] == "STOP"
    assert policy.events[-1]["selector_action"]["selection"] == "ACQUIRE"


def test_closed_loop_metrics_keep_quality_cost_and_safety():
    row = {
        "terminal_correct": True,
        "false_edit": False,
        "missed_edit": False,
        "acquisitions": 1,
        "steps": 2,
        "spent_cost": 1.0,
        "tool_cost": 0.25,
        "quality_gain": 0.4,
        "quality_cost_utility": 0.2,
        "valid_action_count": 2,
        "model_action_count": 2,
        "fallback_count": 0,
    }
    result = metrics([row])
    assert result["terminal_accuracy"] == 1.0
    assert result["mean_quality_cost_utility"] == 0.2
    assert result["mean_tool_cost"] == 0.25


def test_closed_loop_metrics_expose_policy_collapse_diagnostics():
    rows = []
    for target, prediction, acquisitions in (
        ("KEEP", "KEEP", 0),
        ("ADD", "ADD", 1),
        ("DELETE", "KEEP", 0),
        ("RESHAPE", "RESHAPE", 2),
    ):
        rows.append(
            {
                "target_edit": target,
                "predicted_edit": prediction,
                "terminal_correct": target == prediction,
                "false_edit": False,
                "missed_edit": target != prediction,
                "acquisitions": acquisitions,
                "steps": acquisitions + 1,
                "spent_cost": float(acquisitions),
                "quality_gain": 0.0,
                "quality_cost_utility": 0.0,
                "valid_action_count": 1,
                "model_action_count": 1,
                "fallback_count": 0,
            }
        )

    result = metrics(rows)

    assert result["zero_acquisition_rate"] == 0.5
    assert result["two_plus_acquisition_rate"] == 0.25
    assert result["predicted_keep_rate"] == 0.5
    assert result["recall_delete"] == 0.0
    assert 0.0 < result["prediction_entropy_normalized"] < 1.0


def test_always_stop_control_emits_auditable_stop_event():
    current_episode = episode()
    current_sample = sample()
    policy = AlwaysStopPolicy(
        None,
        None,
        "cpu",
        current_sample,
        current_episode,
        Path("unused.jpg"),
        policy_snapshot="nonparametric",
    )

    current_observation = observation(EditOperation.KEEP).model_copy(
        update={"candidates": []}
    )
    action = policy.act(current_observation)

    assert action.action.value == "REJECT"
    assert policy.events[0]["valid_action"] is True
    assert policy.events[0]["selector_action"]["selection"] == "STOP"


class _Ranker:
    safety_margin = 0.1

    def score_state(self, state):
        return {state["candidate_evidence"][0]["evidence_id"]: 0.4}


def _gate_policy(raw):
    current_episode = episode()
    current_episode.evidence_catalog.append(
        current_episode.evidence_catalog[0].model_copy(
            update={"evidence_id": "e-1", "timestamp": "2026_02", "scale": 2}
        )
    )
    policy = object.__new__(QwenGateRankerPolicy)
    policy.sample = sample()
    policy.episode = current_episode
    policy.policy_snapshot = "gate-adapter+ranker"
    policy.ranker = _Ranker()
    policy.events = []
    policy.generate_raw = lambda state, prompt: raw
    return policy


def test_gate_ranker_composes_gate_decision_with_ranked_candidate():
    policy = _gate_policy('{"selection":"ACQUIRE"}')
    action = policy.act(observation())
    assert action.action.value == "ACQUIRE"
    assert action.evidence_id == public_evidence_id("e-1")
    assert policy.events[0]["valid_action"] is True
    assert policy.events[0]["ranker_scores"][public_evidence_id("e-1")] == 0.4


def test_gate_ranker_stop_commits_current_belief():
    policy = _gate_policy('{"selection":"STOP"}')
    action = policy.act(observation())
    assert action.action.value == "COMMIT"
    assert policy.events[0]["selector_action"]["selection"] == "STOP"


def _react_policy(raw):
    current_episode = episode()
    current_episode.evidence_catalog.append(
        current_episode.evidence_catalog[0].model_copy(
            update={"evidence_id": "e-1", "timestamp": "2026_02", "scale": 2}
        )
    )
    policy = object.__new__(QwenReactPolicy)
    policy.sample = sample()
    policy.episode = current_episode
    policy.policy_snapshot = "react-adapter"
    policy.tool_costs = {"IMAGE_QUALITY": 0.03, "TEMPORAL_CHANGE": 0.15}
    policy.events = []
    policy.generate_raw = lambda state, prompt: raw
    return policy


def _react_observation():
    return observation().model_copy(
        update={
            "available_tools": [
                GeoToolName.IMAGE_QUALITY,
                GeoToolName.TEMPORAL_CHANGE,
            ],
            "tool_history": [
                GeoToolResult(
                    call_id="previous",
                    tool=GeoToolName.IMAGE_QUALITY,
                    success=True,
                    outputs={"evidence_id": public_evidence_id("e-0"), "sharpness": 0.7},
                    cost=0.03,
                )
            ],
        }
    )


def test_react_acquires_only_an_available_candidate():
    evidence_id = public_evidence_id("e-1")
    policy = _react_policy(
        f'{{"reason":"newer view may resolve uncertainty","action":"ACQUIRE",'
        f'"evidence_id":"{evidence_id}"}}'
    )
    action = policy.act(_react_observation())
    assert action.action.value == "ACQUIRE"
    assert action.evidence_id == evidence_id
    assert policy.events[0]["valid_action"] is True
    assert policy.events[0]["observable_state"]["tool_observations"][0]["success"] is True


def test_react_accepts_equivalent_trained_selection_serialization():
    policy = _react_policy('{"stage":"REACT","selection":"STOP"}')
    action = policy.act(_react_observation())
    assert action.action.value == "COMMIT"
    assert policy.events[0]["valid_action"] is True
    assert policy.events[0]["react_action_serialization"] == "selection"


def test_react_builds_executable_tool_call_for_acquired_evidence():
    evidence_id = public_evidence_id("e-0")
    policy = _react_policy(
        f'{{"reason":"check image quality first","action":"USE_TOOL",'
        f'"tool":"IMAGE_QUALITY","evidence_id":"{evidence_id}"}}'
    )
    action = policy.act(_react_observation())
    assert action.action.value == "USE_TOOL"
    assert action.tool_call.tool == GeoToolName.IMAGE_QUALITY
    assert action.tool_call.inputs == {"evidence_id": evidence_id}
    assert action.tool_call.call_id.startswith("react-")


def test_react_invalid_tool_action_falls_back_to_current_belief():
    policy = _react_policy(
        '{"reason":"invalid target","action":"USE_TOOL",'
        '"tool":"RASTER_SEGMENT","evidence_id":"missing"}'
    )
    action = policy.act(_react_observation())
    assert action.action.value == "COMMIT"
    assert action.edit == EditOperation.RESHAPE
    assert policy.events[0]["valid_action"] is False
    assert "unavailable tool" in policy.events[0]["parse_error"]


def _protocol_policy(policy_class, raw):
    policy = _react_policy(raw)
    policy.__class__ = policy_class
    return policy


def test_geommagent_accept_executes_current_belief():
    policy = _protocol_policy(
        QwenGeoMMAgentPolicy,
        '{"stage":"SELF_EVALUATE","reason":"evidence is sufficient","action":"ACCEPT"}',
    )

    action = policy.act(_react_observation())

    assert action.action.value == "COMMIT"
    assert policy.events[0]["controller_protocol"] == "geommagent_style"
    assert policy.events[0]["observable_state"]["controller_stage"] == "PLAN"


def test_geommagent_reexecute_builds_tool_call():
    evidence_id = public_evidence_id("e-0")
    policy = _protocol_policy(
        QwenGeoMMAgentPolicy,
        '{"stage":"SELF_EVALUATE","reason":"inspect again","action":"REEXECUTE",'
        f'"tool":"IMAGE_QUALITY","evidence_id":"{evidence_id}"}}',
    )

    action = policy.act(_react_observation())

    assert action.action.value == "USE_TOOL"
    assert action.tool_call.call_id.startswith("geommagent-")


def test_geommagent_normalizes_legacy_reason_action():
    policy = _protocol_policy(
        QwenGeoMMAgentPolicy,
        '{"stage":"PLAN","reason":"STOP"}',
    )

    action = policy.act(_react_observation())

    assert action.action.value == "COMMIT"
    assert policy.events[0]["valid_action"] is True
    assert policy.events[0]["react_action_serialization"] == "reason_action"
    assert policy.events[0]["protocol_action_raw"]["stage"] == "PLAN"


def test_geommagent_normalizes_legacy_terminal_stage():
    policy = _protocol_policy(
        QwenGeoMMAgentPolicy,
        '{"stage":"STOP","reason":"STOP"}',
    )

    action = policy.act(_react_observation())

    assert action.action.value == "COMMIT"
    assert policy.events[0]["valid_action"] is True
    assert (
        policy.events[0]["react_action_serialization"]
        == "reason_action_terminal_stage"
    )
    assert policy.events[0]["protocol_action"]["legacy_stage_action"] == "STOP"


def test_sensesearch_normalizes_compound_stage_action():
    policy = _protocol_policy(
        QwenSenseSearchPolicy,
        '{"stage":"SEARCH|STOP"}',
    )

    action = policy.act(_react_observation())

    assert action.action.value == "COMMIT"
    assert policy.events[0]["valid_action"] is True
    assert (
        policy.events[0]["react_action_serialization"]
        == "compound_stage_action"
    )
    assert (
        policy.events[0]["protocol_action"]["legacy_compound_stage_action"]
        == "SEARCH|STOP"
    )


def test_sensesearch_rejects_ambiguous_compound_stage():
    policy = _protocol_policy(
        QwenSenseSearchPolicy,
        '{"stage":"SEARCH|UNKNOWN"}',
    )

    policy.act(_react_observation())

    assert policy.events[0]["valid_action"] is False
    assert "unsupported sensesearch_style stage" in policy.events[0]["parse_error"]


def test_sensesearch_accepts_crop_tool():
    evidence_id = public_evidence_id("e-0")
    current = _react_observation().model_copy(
        update={
            "available_tools": [
                GeoToolName.RASTER_CROP,
                GeoToolName.RASTER_SEGMENT,
                GeoToolName.IMAGE_QUALITY,
            ]
        }
    )
    policy = _protocol_policy(
        QwenSenseSearchPolicy,
        '{"stage":"SEARCH","reason":"zoom into boundary","action":"USE_TOOL",'
        f'"tool":"RASTER_CROP","evidence_id":"{evidence_id}"}}',
    )
    policy.tool_costs.update({"RASTER_CROP": 0.05, "RASTER_SEGMENT": 0.2})

    action = policy.act(current)

    assert action.action.value == "USE_TOOL"
    assert action.tool_call.call_id.startswith("sensesearch-")
    available = policy.events[0]["observable_state"]["available_tools"]
    assert {item["tool"] for item in available} == {"RASTER_CROP", "RASTER_SEGMENT"}


def test_sensesearch_rejects_out_of_protocol_tool():
    evidence_id = public_evidence_id("e-0")
    policy = _protocol_policy(
        QwenSenseSearchPolicy,
        '{"stage":"SEARCH","reason":"wrong tool","action":"USE_TOOL",'
        f'"tool":"IMAGE_QUALITY","evidence_id":"{evidence_id}"}}',
    )

    action = policy.act(_react_observation())

    assert action.action.value == "COMMIT"
    assert policy.events[0]["valid_action"] is False
    assert "disallowed tool" in policy.events[0]["parse_error"]


def test_sensesearch_registry_exposes_real_search_tools(tmp_path):
    pytest.importorskip("rasterio")
    pytest.importorskip("geopandas")
    sensesearch = build_model_tool_registry("sensesearch_style", tmp_path)
    react = build_model_tool_registry("react", tmp_path)

    assert set(sensesearch.names()) == {
        GeoToolName.IMAGE_QUALITY,
        GeoToolName.TEMPORAL_CHANGE,
        GeoToolName.RASTER_CROP,
        GeoToolName.RASTER_SEGMENT,
        GeoToolName.VECTOR_INSPECT,
    }
    assert set(react.names()) == {
        GeoToolName.IMAGE_QUALITY,
        GeoToolName.TEMPORAL_CHANGE,
    }


def test_asset_root_remap_preserves_suffix():
    assert remap_asset_path(
        "/mnt/mydisk/wh/ActiveMap/datasets/sn7/a.geojson",
        [("/mnt/mydisk/wh/ActiveMap", "/home/wh/ActiveMap")],
    ) == "/home/wh/ActiveMap/datasets/sn7/a.geojson"


def _plan_policy(raw):
    policy = object.__new__(QwenPlanThenExecutePolicy)
    policy.sample = sample()
    policy.episode = episode()
    policy.episode.evidence_catalog.append(
        policy.episode.evidence_catalog[0].model_copy(
            update={"evidence_id": "e-1", "timestamp": "2026_02"}
        )
    )
    policy.policy_snapshot = "plan-adapter"
    policy.max_plan_acquisitions = 2
    policy.plan = None
    policy.events = []
    policy.generate_raw = lambda state, prompt: raw
    return policy


def test_plan_then_execute_calls_model_once_and_uses_frozen_plan():
    evidence_id = public_evidence_id("e-1")
    policy = _plan_policy(f'{{"evidence_ids":["{evidence_id}"]}}')
    first = policy.act(observation())
    second = policy.act(
        observation().model_copy(
            update={"candidates": [], "selected_evidence_ids": [evidence_id]}
        )
    )
    assert first.action.value == "ACQUIRE"
    assert first.evidence_id == evidence_id
    assert second.action.value == "COMMIT"
    assert policy.events[0]["stage"] == "PLAN_OPEN_LOOP"
    assert policy.events[1]["stage"] == "EXECUTE_OPEN_LOOP_PLAN"


def test_plan_then_execute_rejects_over_budget_plan():
    evidence_id = public_evidence_id("e-1")
    policy = _plan_policy(f'{{"evidence_ids":["{evidence_id}"]}}')
    action = policy.act(observation().model_copy(update={"remaining_budget": 0.1}))
    assert action.action.value == "COMMIT"
    assert "remaining budget" in policy.events[0]["parse_error"]


def test_plan_then_execute_accepts_legacy_stop_as_empty_plan():
    policy = _plan_policy('{"stage":"SELECT","selection":"STOP"}')
    action = policy.act(observation())
    assert action.action.value == "COMMIT"
    assert policy.events[0]["valid_action"] is True
