from activemap.agent.records import AgentBelief
from activemap.agent.tool_belief_data import PostAcquisitionToolPairExample
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import EditOperation
from scripts.build_active_catalog_tool_gate_features import (
    UtilityWeights,
    build_gate_features,
    deployed_edit,
    utility_gain,
)


def _belief(values):
    return AgentBelief(
        edit_probabilities=values,
        confidence=0.8,
        uncertainty=0.3,
    )


def _row(target=EditOperation.ADD):
    evidence = "evidence"
    return PostAcquisitionToolPairExample(
        example_id=f"example-{target.value}",
        task_id="task",
        split="train",
        evidence_id=evidence,
        post_acquisition_belief=_belief([0.7, 0.1, 0.1, 0.1]),
        quality_result=GeoToolResult(
            call_id="q",
            tool=GeoToolName.IMAGE_QUALITY,
            success=True,
            outputs={"evidence_id": evidence, "sharpness": 0.4},
            cost=0.03,
        ),
        temporal_result=GeoToolResult(
            call_id="t",
            tool=GeoToolName.TEMPORAL_CHANGE,
            success=True,
            outputs={"evidence_id": evidence, "changed_fraction": 0.3},
            cost=0.15,
        ),
        target_belief=_belief([0.05, 0.85, 0.05, 0.05]),
        gt_edit=target,
        evidence_cost=1.0,
        tool_cost=0.18,
        metadata={
            "source_transition_id": "transition",
            "operation_update_threshold": 0.69,
            "observable_candidate_features": [0.1] * 13,
            "pre_acquisition_initial_budget": 3.0,
            "pre_acquisition_remaining_budget": 3.0,
            "pre_acquisition_spent_cost": 0.0,
            "pre_acquisition_step": 0,
            "test_assets_read": False,
        },
    )


class _Updater:
    def update_pair(self, belief, quality_result, temporal_result):
        return _belief([0.1, 0.8, 0.05, 0.05])


def test_gate_features_are_pre_tool_only_and_utility_is_hidden():
    features, records, summary = build_gate_features(
        [_row()], _Updater(), weights=UtilityWeights()
    )
    assert features.shape == (1, 32)
    assert records[0]["oracle_use_tool"] is True
    assert records[0]["model_input_contains_tool_results"] is False
    assert summary["tool_results_in_features"] is False
    assert summary["target_in_features"] is False


def test_observable_gate_features_do_not_change_with_hidden_target():
    add_features, _, _ = build_gate_features(
        [_row(EditOperation.ADD)], _Updater(), weights=UtilityWeights()
    )
    keep_features, _, _ = build_gate_features(
        [_row(EditOperation.KEEP)], _Updater(), weights=UtilityWeights()
    )
    assert (add_features == keep_features).all()


def test_utility_rewards_correct_update_after_cost():
    gain, diagnostics = utility_gain(
        _belief([0.7, 0.1, 0.1, 0.1]),
        _belief([0.1, 0.8, 0.05, 0.05]),
        EditOperation.ADD,
        0.69,
        0.18,
        UtilityWeights(),
    )
    assert gain > 0.0
    assert diagnostics["prior_edit"] == "KEEP"
    assert diagnostics["forced_tool_edit"] == "ADD"


def test_deployed_edit_respects_operation_threshold():
    assert deployed_edit(_belief([0.4, 0.5, 0.05, 0.05]), 0.69) == EditOperation.KEEP
    assert deployed_edit(_belief([0.2, 0.7, 0.05, 0.05]), 0.69) == EditOperation.ADD
