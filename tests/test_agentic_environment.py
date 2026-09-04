import json
from pathlib import Path

import numpy as np
import pytest

from activemap.agent.environment import MapMaintenanceEnv, rollout_agent_policy
from activemap.agent.identifiers import public_evidence_id, public_task_id
from activemap.agent.records import AgentAction, AgentActionType
from activemap.agent.tools import (
    CounterfactualBeliefUpdater,
    GreedyAgentPolicy,
    RegisteredToolBeliefUpdater,
    UncertaintyAwareAgentPolicy,
    belief_from_operation_prediction,
)
from activemap.agent.trajectories import build_agent_trajectories, write_agent_datasets
from activemap.features import EVIDENCE_DIM, HYPOTHESIS_DIM, STATE_DIM
from activemap.geo_tools.records import GeoToolCall, GeoToolName, GeoToolResult
from activemap.geo_tools.registry import GeoToolRegistry
from activemap.models import EditOperation
from activemap.selector_records import SelectorSample


def _prediction(probabilities: list[float], confidence: float) -> dict[str, object]:
    return {
        "edit_probabilities": probabilities,
        "confidence": confidence,
        "geometry_delta": [0.0] * 8,
    }


def _sample() -> SelectorSample:
    hypothesis = [0.8, 0.1, 0.05, 0.05, *([0.0] * 8), 0.4, 0.5, 0.0, 0.0]
    return SelectorSample(
        sample_id="aoi-1__b1p0__s0",
        split="train",
        edit_type=EditOperation.KEEP,
        hypothesis_features=hypothesis,
        state_features=[1.0, 0.0, 0.8, 0.1, 0.05, 0.05, 0.1, 0.0],
        evidence_ids=["e1", "e2"],
        evidence_features=[[0.0] * EVIDENCE_DIM, [0.0] * EVIDENCE_DIM],
        evidence_costs=[0.5, 0.7],
        false_edit_risks=[0.0, 0.0],
        oracle_utilities=[0.6, 0.1],
        stop_utility=0.0,
        metadata={
            "source_episode": "aoi-1",
            "budget": 0.6,
            "oracle_step": 0,
            "selected_evidence_ids": ["seed"],
            "gt_edit": "ADD",
            "evidence_predictions": {
                "seed": _prediction([0.8, 0.1, 0.05, 0.05], 0.5),
                "e1": _prediction([0.05, 0.9, 0.03, 0.02], 0.9),
                "e2": _prediction([0.1, 0.2, 0.1, 0.6], 0.7),
            },
        },
    )


def _scores(sample: SelectorSample) -> np.ndarray:
    return np.asarray([2.0 - index for index in range(len(sample.evidence_ids))] + [0.0])


def test_agent_action_contract_rejects_invalid_payloads() -> None:
    with pytest.raises(ValueError):
        AgentAction(action=AgentActionType.ACQUIRE)
    with pytest.raises(ValueError):
        AgentAction(action=AgentActionType.COMMIT, edit=EditOperation.KEEP)


def test_operation_prediction_converts_to_agent_belief() -> None:
    belief = belief_from_operation_prediction(
        {
            "edit_probabilities": [0.1, 0.7, 0.1, 0.1],
            "confidence": 0.7,
            "uncertainty": 0.4,
            "gated_edit": "KEEP",
            "geometry_delta": [0.0] * 8,
        }
    )
    assert sum(belief.edit_probabilities) == pytest.approx(1.0)
    assert belief.edit_probabilities[1] == pytest.approx(0.7)
    assert belief.predicted_edit == EditOperation.KEEP
    assert belief.uncertainty == pytest.approx(0.4)


def test_uncertainty_aware_policy_acquires_then_executes_calibrated_edit() -> None:
    sample = _sample()
    environment = MapMaintenanceEnv(
        sample,
        budget=0.6,
        score_fn=_scores,
        belief_updater=CounterfactualBeliefUpdater(sample),
        top_k=2,
    )
    policy = UncertaintyAwareAgentPolicy(max_uncertainty=0.25, min_confidence=0.8)
    first = policy.act(environment.reset())
    assert first.action == AgentActionType.ACQUIRE

    confident_keep = belief_from_operation_prediction(
        {
            "edit_probabilities": [0.2, 0.7, 0.05, 0.05],
            "confidence": 0.8,
            "uncertainty": 0.2,
            "gated_edit": "KEEP",
            "geometry_delta": [0.0] * 8,
        }
    )
    observation = environment.observation().model_copy(update={"belief": confident_keep})
    assert policy.act(observation).action == AgentActionType.REJECT


def test_closed_loop_updates_belief_then_commits() -> None:
    sample = _sample()
    updater = CounterfactualBeliefUpdater(sample)
    initial = updater.fuse(["seed"])
    updated = updater.fuse(["seed", "e1"])
    assert initial.predicted_edit == EditOperation.KEEP
    assert updated.predicted_edit == EditOperation.ADD

    environment = MapMaintenanceEnv(
        sample,
        budget=0.6,
        score_fn=_scores,
        belief_updater=updater,
        top_k=2,
    )
    trajectory = rollout_agent_policy(environment, GreedyAgentPolicy())
    assert [step.action.action for step in trajectory.transitions] == [
        AgentActionType.ACQUIRE,
        AgentActionType.COMMIT,
    ]
    assert trajectory.transitions[0].next_observation is not None
    assert trajectory.transitions[0].next_observation.belief.predicted_edit == EditOperation.ADD
    assert trajectory.transitions[-1].reward == 1.0


def test_environment_uses_executable_terminal_scores() -> None:
    sample = _sample()
    metadata = dict(sample.metadata)
    metadata.update(
        {
            "utility_mode": "executable",
            "utility_profile": "balanced",
            "executable_outcomes": {
                "seed": {"terminal_score_before_cost": 0.2},
                "e1": {"terminal_score_before_cost": 0.6},
                "e2": {"terminal_score_before_cost": -0.1},
            },
        }
    )
    state = list(sample.state_features)
    state[7] = 0.2
    environment = MapMaintenanceEnv(
        sample.model_copy(update={"metadata": metadata, "state_features": state}),
        budget=0.6,
        score_fn=_scores,
        belief_updater=CounterfactualBeliefUpdater(sample),
    )
    assert environment.current_sample().state_features[7] == pytest.approx(0.2)
    assert environment._marginal_utility(0) == pytest.approx(0.4 - 0.1 * 0.5 / 0.6)
    assert environment._marginal_utility(1) == pytest.approx(-0.3 - 0.1 * 0.7 / 0.6)


def test_environment_preserves_full_catalog_count_after_budget_filtering() -> None:
    sample = _sample().model_copy(
        update={
            "evidence_ids": ["e1"],
            "evidence_features": [[0.0] * EVIDENCE_DIM],
            "evidence_costs": [0.5],
            "false_edit_risks": [0.0],
            "oracle_utilities": [0.6],
        }
    )
    environment = MapMaintenanceEnv(
        sample,
        budget=0.6,
        score_fn=_scores,
        belief_updater=CounterfactualBeliefUpdater(sample),
    )
    assert environment.current_sample().state_features[6] == pytest.approx(1 / 3)


def test_counterfactual_fusion_reapplies_operation_safety_gate() -> None:
    sample = _sample()
    metadata = dict(sample.metadata)
    metadata["operation_update_threshold"] = 0.8
    metadata["evidence_predictions"] = {
        "seed": _prediction([0.3, 0.6, 0.05, 0.05], 0.8),
        "e1": _prediction([0.35, 0.55, 0.05, 0.05], 0.8),
        "e2": _prediction([0.1, 0.2, 0.1, 0.6], 0.7),
    }
    belief = CounterfactualBeliefUpdater(sample.model_copy(update={"metadata": metadata})).fuse(
        ["seed", "e1"]
    )
    assert belief.edit_probabilities[1] > belief.edit_probabilities[0]
    assert belief.predicted_edit == EditOperation.KEEP


def test_tool_use_consumes_budget_and_is_added_to_history() -> None:
    class DummyQualityTool:
        name = GeoToolName.IMAGE_QUALITY
        cost = 0.1

        def run(self, call: GeoToolCall) -> GeoToolResult:
            return GeoToolResult(
                call_id=call.call_id,
                tool=call.tool,
                success=True,
                outputs={"valid_fraction": 0.9},
                cost=self.cost,
            )

    sample = _sample()
    registry = GeoToolRegistry()
    registry.register(DummyQualityTool())
    environment = MapMaintenanceEnv(
        sample,
        budget=0.6,
        score_fn=_scores,
        belief_updater=CounterfactualBeliefUpdater(sample),
        tool_registry=registry,
    )
    transition = environment.step(
        AgentAction(
            action=AgentActionType.USE_TOOL,
            tool_call=GeoToolCall(
                call_id="quality-1",
                tool=GeoToolName.IMAGE_QUALITY,
                inputs={"image_path": "unused"},
            ),
        )
    )
    assert not transition.done
    assert transition.next_observation is not None
    assert transition.next_observation.remaining_budget == pytest.approx(0.5)
    assert transition.next_observation.tool_history[0].success


def test_tool_call_does_not_consume_the_acquisition_limit() -> None:
    class DummyQualityTool:
        name = GeoToolName.IMAGE_QUALITY
        cost = 0.1

        def run(self, call: GeoToolCall) -> GeoToolResult:
            return GeoToolResult(
                call_id=call.call_id,
                tool=call.tool,
                success=True,
                outputs={"valid_fraction": 0.9},
                cost=self.cost,
            )

    class ToolThenAcquirePolicy:
        def act(self, observation):
            if not observation.tool_history:
                return AgentAction(
                    action=AgentActionType.USE_TOOL,
                    tool_call=GeoToolCall(
                        call_id="quality-rollout",
                        tool=GeoToolName.IMAGE_QUALITY,
                        inputs={"image_path": "unused"},
                    ),
                )
            if "e1" not in observation.selected_evidence_ids:
                return AgentAction(action=AgentActionType.ACQUIRE, evidence_id="e1")
            return AgentAction(action=AgentActionType.COMMIT, edit=EditOperation.ADD)

    sample = _sample()
    registry = GeoToolRegistry()
    registry.register(DummyQualityTool())
    environment = MapMaintenanceEnv(
        sample,
        budget=0.6,
        score_fn=_scores,
        belief_updater=CounterfactualBeliefUpdater(sample),
        tool_registry=registry,
    )
    trajectory = rollout_agent_policy(
        environment,
        ToolThenAcquirePolicy(),
        max_acquisitions=1,
        max_tool_calls=1,
    )
    assert [transition.action.action for transition in trajectory.transitions] == [
        AgentActionType.USE_TOOL,
        AgentActionType.ACQUIRE,
        AgentActionType.COMMIT,
    ]
    assert trajectory.metadata["tool_call_count"] == 1
    assert trajectory.metadata["acquisition_count"] == 1


def test_successful_tool_result_can_update_belief_before_replanning() -> None:
    class ChangeTool:
        name = GeoToolName.TEMPORAL_CHANGE
        cost = 0.1

        def run(self, call: GeoToolCall) -> GeoToolResult:
            return GeoToolResult(
                call_id=call.call_id,
                tool=call.tool,
                success=True,
                outputs={"changed_fraction": 0.8},
                cost=self.cost,
            )

    def change_adapter(belief, result):
        assert result.outputs["changed_fraction"] == 0.8
        return belief.model_copy(
            update={
                "edit_probabilities": [0.1, 0.8, 0.05, 0.05],
                "confidence": 0.9,
                "uncertainty": 0.2,
                "recommended_edit": EditOperation.ADD,
            }
        )

    sample = _sample()
    registry = GeoToolRegistry()
    registry.register(ChangeTool())
    environment = MapMaintenanceEnv(
        sample,
        budget=0.6,
        score_fn=_scores,
        belief_updater=CounterfactualBeliefUpdater(sample),
        tool_registry=registry,
        tool_belief_updater=RegisteredToolBeliefUpdater(
            {GeoToolName.TEMPORAL_CHANGE: change_adapter}
        ),
    )

    transition = environment.step(
        AgentAction(
            action=AgentActionType.USE_TOOL,
            tool_call=GeoToolCall(
                call_id="change-1",
                tool=GeoToolName.TEMPORAL_CHANGE,
                inputs={"image_path": "unused"},
            ),
        )
    )

    assert transition.next_observation is not None
    assert transition.next_observation.belief.predicted_edit == EditOperation.ADD
    assert transition.next_observation.belief.confidence == pytest.approx(0.9)


def test_agent_tools_resolve_only_acquired_evidence_assets() -> None:
    class ResolvingTool:
        name = GeoToolName.IMAGE_QUALITY
        cost = 0.1

        def run(self, call: GeoToolCall) -> GeoToolResult:
            return GeoToolResult(
                call_id=call.call_id,
                tool=call.tool,
                success=True,
                outputs={"resolved_path": call.inputs["image_path"]},
                cost=self.cost,
            )

    sample = _sample()
    registry = GeoToolRegistry()
    registry.register(ResolvingTool())
    environment = MapMaintenanceEnv(
        sample,
        budget=0.6,
        score_fn=_scores,
        belief_updater=CounterfactualBeliefUpdater(sample),
        tool_registry=registry,
        asset_paths={"seed": "/data/seed.tif", "e1": "/data/e1.tif"},
        tool_parameters={
            "seed": {"pixel_window": [0, 0, 32, 32], "out_size": [16, 16]},
            "e1": {"pixel_window": [8, 8, 32, 32], "out_size": [16, 16]},
        },
    )

    def call(evidence_id: str) -> AgentAction:
        return AgentAction(
            action=AgentActionType.USE_TOOL,
            tool_call=GeoToolCall(
                call_id=f"quality-{evidence_id}",
                tool=GeoToolName.IMAGE_QUALITY,
                inputs={"evidence_id": evidence_id},
            ),
        )

    resolved = environment.step(call("seed"))
    assert resolved.next_observation is not None
    assert resolved.next_observation.tool_history[-1].outputs["resolved_path"] == "/data/seed.tif"
    with pytest.raises(ValueError, match="acquired evidence"):
        environment.step(call("e1"))


def test_temporal_tool_resolves_acquired_evidence_against_initial_anchor() -> None:
    class ResolvingTemporalTool:
        name = GeoToolName.TEMPORAL_CHANGE
        cost = 0.1

        def run(self, call: GeoToolCall) -> GeoToolResult:
            return GeoToolResult(
                call_id=call.call_id,
                tool=call.tool,
                success=True,
                outputs={
                    "before_path": call.inputs["before_path"],
                    "after_path": call.inputs["after_path"],
                    "parameters": call.parameters,
                },
                cost=self.cost,
            )

    sample = _sample()
    registry = GeoToolRegistry()
    registry.register(ResolvingTemporalTool())
    environment = MapMaintenanceEnv(
        sample,
        budget=0.7,
        score_fn=_scores,
        belief_updater=CounterfactualBeliefUpdater(sample),
        tool_registry=registry,
        asset_paths={"seed": "/data/seed.tif", "e1": "/data/e1.tif"},
        tool_parameters={
            "seed": {"pixel_window": [0, 0, 32, 32], "out_size": [16, 16]},
            "e1": {"pixel_window": [8, 8, 32, 32], "out_size": [16, 16]},
        },
    )
    environment.step(AgentAction(action=AgentActionType.ACQUIRE, evidence_id="e1"))
    transition = environment.step(
        AgentAction(
            action=AgentActionType.USE_TOOL,
            tool_call=GeoToolCall(
                call_id="temporal-e1",
                tool=GeoToolName.TEMPORAL_CHANGE,
                inputs={"evidence_id": "e1"},
            ),
        )
    )

    assert transition.next_observation is not None
    result = transition.next_observation.tool_history[-1]
    assert result.outputs["before_path"] == "/data/e1.tif"
    assert result.outputs["after_path"] == "/data/seed.tif"
    assert result.outputs["evidence_id"] == "e1"
    assert result.outputs["parameters"] == {
        "pixel_window": [8, 8, 32, 32],
        "out_size": [16, 16],
    }


def test_model_tool_receives_trusted_per_evidence_inputs() -> None:
    class ResolvingSegmentationTool:
        name = GeoToolName.RASTER_SEGMENT
        cost = 0.1

        def run(self, call: GeoToolCall) -> GeoToolResult:
            return GeoToolResult(
                call_id=call.call_id,
                tool=call.tool,
                success=True,
                outputs={
                    "image_path": call.inputs["image_path"],
                    "prior_mask_path": call.inputs["prior_mask_path"],
                },
                cost=self.cost,
            )

    sample = _sample()
    registry = GeoToolRegistry()
    registry.register(ResolvingSegmentationTool())
    environment = MapMaintenanceEnv(
        sample,
        budget=0.6,
        score_fn=_scores,
        belief_updater=CounterfactualBeliefUpdater(sample),
        tool_registry=registry,
        asset_paths={"seed": "/data/seed.tif", "e1": "/data/e1.tif"},
        tool_inputs_by_evidence={
            "seed": {"prior_mask_path": "/data/seed-prior.npy"},
            "e1": {"prior_mask_path": "/data/e1-prior.npy"},
        },
    )

    transition = environment.step(
        AgentAction(
            action=AgentActionType.USE_TOOL,
            tool_call=GeoToolCall(
                call_id="segment-seed",
                tool=GeoToolName.RASTER_SEGMENT,
                inputs={"evidence_id": "seed"},
            ),
        )
    )

    assert transition.next_observation is not None
    result = transition.next_observation.tool_history[-1]
    assert result.outputs["image_path"] == "/data/seed.tif"
    assert result.outputs["prior_mask_path"] == "/data/seed-prior.npy"


def test_agent_dataset_builder_writes_sft_and_preference_records(tmp_path: Path) -> None:
    trajectories, metrics = build_agent_trajectories([_sample()], score_fn=_scores, top_k=1)
    assert len(trajectories) == 1
    assert len(trajectories[0].transitions) == 2
    assert trajectories[0].transitions[-1].done
    assert metrics["natural_top_k_recall"] == 1.0

    summary = write_agent_datasets(trajectories, tmp_path)
    assert summary["sft_count"] == 2
    assert summary["preference_count"] == 2
    sft = json.loads((tmp_path / "sft.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert [message["role"] for message in sft["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    user_observation = json.loads(sft["messages"][1]["content"])
    assistant_action = json.loads(sft["messages"][2]["content"])
    assert user_observation["task_id"] == public_task_id("aoi-1")
    assert user_observation["candidates"][0]["evidence_id"] == public_evidence_id("e1")
    assert user_observation["selected_evidence_ids"] == [public_evidence_id("seed")]
    assert assistant_action["evidence_id"] == public_evidence_id("e1")
    assert "aoi-1" not in sft["messages"][1]["content"]
    preference = json.loads(
        (tmp_path / "preferences.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert preference["chosen_utility"] >= preference["rejected_utility"]


def test_public_identifier_environment_accepts_public_acquire_action() -> None:
    sample = _sample()
    environment = MapMaintenanceEnv(
        sample,
        budget=0.6,
        score_fn=_scores,
        belief_updater=CounterfactualBeliefUpdater(sample),
        public_identifiers=True,
    )
    observation = environment.reset()
    assert observation.task_id == public_task_id("aoi-1")
    assert observation.candidates[0].evidence_id == public_evidence_id("e1")
    transition = environment.step(
        AgentAction(
            action=AgentActionType.ACQUIRE,
            evidence_id=public_evidence_id("e1"),
        )
    )
    assert transition.next_observation is not None
    assert public_evidence_id("e1") in transition.next_observation.selected_evidence_ids


def test_environment_recomputes_marginal_utility_after_acquisition() -> None:
    sample = _sample()
    observed_states = []

    def recording_scores(current: SelectorSample) -> np.ndarray:
        observed_states.append(list(current.state_features))
        return np.asarray([2.0 - index for index in range(len(current.evidence_ids))] + [0.0])

    environment = MapMaintenanceEnv(
        sample,
        budget=2.0,
        score_fn=recording_scores,
        belief_updater=CounterfactualBeliefUpdater(sample),
    )
    first_oracle = environment.oracle_action()
    assert first_oracle.action == AgentActionType.ACQUIRE
    assert first_oracle.evidence_id == "e1"
    transition = environment.step(first_oracle)
    assert transition.reward == pytest.approx(0.6)
    assert transition.next_observation is not None
    assert environment.oracle_action().action == AgentActionType.COMMIT
    assert observed_states[-1][6] == pytest.approx(2 / 3)
    assert observed_states[-1][7] == pytest.approx(environment.current_gain)


def test_agent_fixture_dimensions_match_model_contract() -> None:
    sample = _sample()
    assert len(sample.hypothesis_features) == HYPOTHESIS_DIM
    assert len(sample.state_features) == STATE_DIM
