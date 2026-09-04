from pathlib import Path

from activemap.agent.active_catalog import terminal_agent_action
from activemap.agent.identifiers import public_evidence_id
from activemap.agent.records import (
    AgentActionType,
    AgentBelief,
    AgentCandidate,
    AgentObservation,
)
from activemap.agent.tools import GreedyAgentPolicy
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import (
    CandidateHypothesis,
    EditOperation,
    EditRecord,
    EpisodeRecord,
    EvidenceItem,
    GeoJSONGeometry,
)
from activemap.selector_records import SelectorSample
from scripts.evaluate_active_catalog_closed_loop_baselines import (
    PostToolAdaptedPolicy,
    RankerOnlyPolicy,
    deterministic_sample,
    make_environment,
    parse_learned_selector,
    parse_learned_selector_stop_margin,
)


def episode() -> EpisodeRecord:
    geometry = GeoJSONGeometry(
        type="Polygon",
        coordinates=[[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    )
    evidence = [
        EvidenceItem(
            evidence_id=f"e-{index}",
            timestamp=f"2026_0{index + 1}",
            region=(0, 0, 16, 16),
            scale=index + 1,
            image_path="/unused/image.tif",
            clear_fraction=1.0,
            cost=float(index + 1),
        )
        for index in range(2)
    ]
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
        evidence_catalog=evidence,
        is_synthetic=False,
        derivation_version="fixture-v1",
    )


def sample() -> SelectorSample:
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
            "evidence_predictions": {
                evidence_id: {
                    "edit_probabilities": [0.1, 0.1, 0.1, 0.7],
                    "confidence": 0.8,
                    "geometry_delta": [0.0] * 8,
                }
                for evidence_id in ("e-0", "e-1")
            },
        },
    )


def observation() -> AgentObservation:
    return AgentObservation(
        task_id="task",
        split="val",
        step=1,
        initial_budget=3.0,
        remaining_budget=2.0,
        spent_cost=1.0,
        selected_evidence_ids=[public_evidence_id("e-0")],
        belief=AgentBelief(
            edit_probabilities=[0.1, 0.1, 0.1, 0.7],
            confidence=0.8,
            uncertainty=0.2,
            recommended_edit=EditOperation.RESHAPE,
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


class FakeRanker:
    def __init__(self, decision: str, evidence_id: str | None) -> None:
        self.decision = decision
        self.evidence_id = evidence_id

    def decide(self, state):
        assert state["candidate_evidence"]
        return self.decision, self.evidence_id, 0.1


def test_ranker_only_acquires_selected_evidence():
    current_sample = sample()
    current_episode = episode()
    current_observation = observation()
    expected = current_observation.candidates[0].evidence_id
    policy = RankerOnlyPolicy(
        FakeRanker("ACQUIRE", expected), current_sample, current_episode
    )

    action = policy.act(current_observation)

    assert action.action == AgentActionType.ACQUIRE
    assert action.evidence_id == expected


def test_ranker_only_stop_uses_current_terminal_decision():
    current_sample = sample()
    current_episode = episode()
    current_observation = observation()
    policy = RankerOnlyPolicy(
        FakeRanker("STOP", None), current_sample, current_episode
    )

    action = policy.act(current_observation)

    assert action == terminal_agent_action(current_observation)


def test_learned_selector_logits_control_acquire_and_stop():
    current_sample = sample()
    acquire_environment = make_environment(
        current_sample,
        16,
        score_fn=lambda current: [0.1] * (len(current.evidence_ids) - 1)
        + [0.8, 0.2],
    )
    acquire_observation = acquire_environment.reset()
    acquire = GreedyAgentPolicy().act(acquire_observation)
    assert acquire.action == AgentActionType.ACQUIRE

    stop_environment = make_environment(
        current_sample,
        16,
        score_fn=lambda current: [0.1] * len(current.evidence_ids) + [0.9],
    )
    stop_observation = stop_environment.reset()
    stop = GreedyAgentPolicy().act(stop_observation)
    assert stop == terminal_agent_action(stop_observation)


def test_parse_learned_selector_keeps_label_and_checkpoint():
    assert parse_learned_selector("generic_selector=/tmp/generic.pt") == (
        "generic_selector",
        Path("/tmp/generic.pt"),
    )


def test_parse_learned_selector_stop_margin_keeps_label_and_value():
    assert parse_learned_selector_stop_margin("generic_selector=0.75") == (
        "generic_selector",
        0.75,
    )


def test_deterministic_sample_is_stable_and_not_prefix_limited():
    samples = []
    for index in range(20):
        current = sample().model_copy(deep=True)
        current.sample_id = f"sample-{index}"
        current.metadata["source_episode"] = f"episode-{index}"
        samples.append(current)

    selected = deterministic_sample(samples, 6, 20260729)

    assert selected == deterministic_sample(samples, 6, 20260729)
    assert selected != samples[:6]
    assert len({row.sample_id for row in selected}) == 6


def test_post_tool_adapter_overrides_greedy_terminal_edit():
    current = observation().model_copy(
        update={
            "candidates": [],
            "tool_history": [
                GeoToolResult(
                    call_id="quality",
                    tool=GeoToolName.IMAGE_QUALITY,
                    success=True,
                    outputs={"valid_fraction": 1.0},
                    cost=0.03,
                )
            ],
        }
    )

    class KeepAdapter:
        def predict(self, belief, tool_history):
            assert belief.predicted_edit == EditOperation.RESHAPE
            assert tool_history
            return EditOperation.KEEP

    policy = PostToolAdaptedPolicy(GreedyAgentPolicy(), KeepAdapter())
    action = policy.act(current)

    assert action.action == AgentActionType.REJECT
