import pytest

from activemap.agent.records import AgentBelief
from activemap.agent.tool_belief_data import ToolBeliefExample
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import EditOperation
from scripts.evaluate_tool_belief_interventions import evaluate


def _belief(index: int, confidence: float = 0.9) -> AgentBelief:
    probabilities = [0.02] * 4
    probabilities[index] = 0.94
    return AgentBelief(
        edit_probabilities=probabilities,
        confidence=confidence,
        uncertainty=0.1,
        geometry_delta=[0.0] * 8,
    )


class _PerfectUpdater:
    def update(self, belief: AgentBelief, result: GeoToolResult) -> AgentBelief:
        if result.tool == GeoToolName.IMAGE_QUALITY:
            return belief
        return _belief(int(result.outputs["class_index"]))


def _rows() -> list[ToolBeliefExample]:
    rows = []
    prior = _belief(0, confidence=0.8)
    for class_index, operation in enumerate(EditOperation):
        target = _belief(class_index)
        for evidence_index in range(3):
            for tool, target_belief, outputs, cost, kind in (
                (
                    GeoToolName.IMAGE_QUALITY,
                    prior,
                    {"sharpness": 0.1 + evidence_index},
                    0.03,
                    "observational_noop",
                ),
                (
                    GeoToolName.TEMPORAL_CHANGE,
                    target,
                    {"changed_fraction": 0.2, "class_index": class_index},
                    0.15,
                    "teacher_belief_update",
                ),
            ):
                key = f"{operation.value}-{evidence_index}-{tool.value}"
                rows.append(
                    ToolBeliefExample(
                        record_id=key,
                        episode_id=f"episode-{operation.value}",
                        split="val",
                        evidence_id=f"evidence-{evidence_index}",
                        prior_belief=prior,
                        tool_result=GeoToolResult(
                            call_id=key,
                            tool=tool,
                            success=True,
                            outputs=outputs,
                            cost=cost,
                        ),
                        target_belief=target_belief,
                        gt_edit=operation,
                        metadata={"target_kind": kind, "test_assets_read": False},
                    )
                )
    return rows


def test_intervention_evaluation_covers_one_step_recurrence_cost_and_noop() -> None:
    report, details = evaluate(
        _rows(),
        _PerfectUpdater(),
        max_false_edit_increase=0.02,
        max_missed_edit_increase=0.02,
        max_noop_probability_drift=0.02,
        max_noop_confidence_drift=0.05,
        max_noop_geometry_drift=0.02,
        minimum_macro_f1_gain=0.01,
    )

    assert report["protocol"]["episode_count"] == 4
    assert report["protocol"]["one_step_count"] == 12
    assert report["summaries"]["learned_one_step"]["macro_f1"] == 1.0
    assert report["summaries"]["learned_no_current_one_step"]["macro_f1"] == 1.0
    assert report["summaries"]["learned_temporal_3"]["macro_f1"] == 1.0
    assert report["summaries"]["learned_no_current_temporal_3"]["macro_f1"] == 1.0
    assert report["summaries"]["learned_quality_3"]["mean_tool_cost"] == pytest.approx(0.09)
    assert report["summaries"]["learned_temporal_3"]["mean_tool_cost"] == pytest.approx(0.45)
    assert report["quality_noop_delta_max"]["probability_l1"] == 0.0
    assert report["gates"]["passed"]
    assert len(details) == 12
