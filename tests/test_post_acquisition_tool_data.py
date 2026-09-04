from activemap.agent.identifiers import public_evidence_id, public_task_id
from activemap.agent.environment import MapMaintenanceEnv
from activemap.agent.tools import CounterfactualBeliefUpdater
from activemap.agent.records import AgentBelief
from activemap.agent.tool_belief_data import (
    ToolBeliefSequenceExample,
    ToolBeliefSequenceStep,
)
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import EditOperation
from activemap.selector_records import SelectorSample
from scripts.build_post_acquisition_mixed_sft import counterfactual_keep_rollout_sample
from scripts.build_post_acquisition_tool_data import build_examples


def _belief(probabilities):
    return AgentBelief(
        edit_probabilities=probabilities,
        confidence=max(probabilities),
        uncertainty=0.5,
    )


def _result(tool, call_id):
    return GeoToolResult(
        call_id=call_id,
        tool=tool,
        success=True,
        cost=0.1,
        outputs={},
    )


def test_build_post_acquisition_examples_uses_fused_candidate_belief():
    raw_task = "raw-task"
    initial_id = "initial"
    candidates = ["candidate-a", "candidate-b", "candidate-c"]
    predictions = {
        initial_id: {
            "edit_probabilities": [0.8, 0.1, 0.05, 0.05],
            "confidence": 0.8,
            "geometry_delta": [0.0] * 8,
        },
        **{
            evidence_id: {
                "edit_probabilities": [0.1, 0.8, 0.05, 0.05],
                "confidence": 0.95,
                "geometry_delta": [0.0] * 8,
            }
            for evidence_id in candidates
        },
    }
    sample = SelectorSample(
        sample_id="sample",
        split="train",
        edit_type=EditOperation.KEEP,
        hypothesis_features=[0.8, 0.1, 0.05, 0.05] + [0.0] * 12,
        state_features=[1.0, 0.0, 0.8, 0.1, 0.05, 0.05, 0.0, 0.0],
        evidence_ids=[initial_id, *candidates],
        evidence_features=[[0.0] * 13 for _ in range(4)],
        evidence_costs=[1.0] * 4,
        false_edit_risks=[0.0] * 4,
        oracle_utilities=[0.0] * 4,
        metadata={
            "source_episode": raw_task,
            "initial_evidence_id": initial_id,
            "selected_evidence_ids": [initial_id],
            "oracle_step": 0,
            "gt_edit": "ADD",
            "operation_update_threshold": 0.69,
            "evidence_predictions": predictions,
        },
    )
    steps = []
    for index, evidence_id in enumerate(candidates):
        belief = _belief([0.1, 0.8, 0.05, 0.05])
        steps.append(
            ToolBeliefSequenceStep(
                evidence_id=public_evidence_id(evidence_id),
                quality_result=_result(GeoToolName.IMAGE_QUALITY, f"q{index}"),
                temporal_result=_result(GeoToolName.TEMPORAL_CHANGE, f"t{index}"),
                individual_target_belief=belief,
                cumulative_target_belief=belief,
            )
        )
    sequence = ToolBeliefSequenceExample(
        sequence_id="sequence",
        episode_id=public_task_id(raw_task),
        split="train",
        initial_belief=_belief([0.8, 0.1, 0.05, 0.05]),
        steps=steps,
        gt_edit=EditOperation.ADD,
        metadata={"test_assets_read": False},
    )

    examples, summary = build_examples([sample], [sequence], split="train")

    assert len(examples) == 3
    assert max(
        range(4),
        key=examples[0].post_acquisition_belief.edit_probabilities.__getitem__,
    ) == list(EditOperation).index(EditOperation.ADD)
    assert examples[0].post_acquisition_belief.predicted_edit == EditOperation.KEEP
    assert examples[0].target_belief.predicted_edit == EditOperation.ADD
    assert (
        examples[0].target_belief.confidence
        == examples[0].post_acquisition_belief.confidence
    )
    assert summary["baseline_argmax_accuracy"] == 1.0
    assert summary["baseline_deployed_decision_accuracy"] == 0.0
    assert examples[0].metadata["operation_update_threshold"] == 0.69
    assert summary["reachable_by_construction"] is True


def test_counterfactual_keep_reentry_state_preserves_preacquired_budget() -> None:
    sample = SelectorSample(
        sample_id="keep-state",
        split="train",
        edit_type=EditOperation.KEEP,
        hypothesis_features=[0.8, 0.1, 0.05, 0.05] + [0.0] * 12,
        state_features=[1.0, 0.0, 0.8, 0.1, 0.05, 0.05, 0.25, 0.1],
        evidence_ids=["candidate-a", "candidate-b"],
        evidence_features=[[0.0] * 13, [0.0] * 13],
        evidence_costs=[0.3, 0.8],
        false_edit_risks=[0.1, 0.2],
        oracle_utilities=[0.4, 0.1],
        metadata={
            "source_episode": "keep-task",
            "budget": 1.0,
            "oracle_step": 0,
            "selected_evidence_ids": ["seed"],
            "gt_edit": "KEEP",
            "evidence_predictions": {
                "seed": {
                    "edit_probabilities": [0.8, 0.1, 0.05, 0.05],
                    "confidence": 0.8,
                    "geometry_delta": [0.0] * 8,
                },
                "candidate-a": {
                    "edit_probabilities": [0.6, 0.2, 0.1, 0.1],
                    "confidence": 0.6,
                    "geometry_delta": [0.0] * 8,
                },
                "candidate-b": {
                    "edit_probabilities": [0.2, 0.6, 0.1, 0.1],
                    "confidence": 0.6,
                    "geometry_delta": [0.0] * 8,
                },
            },
        },
    )

    reentry = counterfactual_keep_rollout_sample(sample)
    environment = MapMaintenanceEnv(
        reentry,
        budget=1.0,
        score_fn=lambda current: np.zeros(len(current.evidence_ids) + 1),
        belief_updater=CounterfactualBeliefUpdater(reentry),
    )
    observation = environment.reset()

    assert reentry.metadata["counterfactual_reentry"] is True
    assert reentry.metadata["counterfactual_acquired_evidence_id"] == "candidate-a"
    assert observation.selected_evidence_ids == ["seed", "candidate-a"]
    assert observation.remaining_budget == 0.7
    assert environment.spent_penalty > 0.0
