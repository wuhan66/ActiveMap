from activemap.agent.records import AgentBelief
from activemap.agent.tool_belief_data import (
    ToolBeliefSequenceExample,
    ToolBeliefSequenceStep,
)
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import EditOperation
from scripts.evaluate_tool_belief_sequences import evaluate_sequences


def _belief(index: int, confidence: float = 0.9) -> AgentBelief:
    probabilities = [0.02] * 4
    probabilities[index] = 0.94
    return AgentBelief(
        edit_probabilities=probabilities,
        confidence=confidence,
        uncertainty=0.1,
        geometry_delta=[0.0] * 8,
    )


class _PerfectPairedUpdater:
    def update_pair(
        self,
        belief: AgentBelief,
        quality_result: GeoToolResult,
        temporal_result: GeoToolResult,
        *,
        use_quality: bool = True,
    ) -> AgentBelief:
        del belief, quality_result, use_quality
        return _belief(int(temporal_result.outputs["class_index"]))


def _sequences() -> list[ToolBeliefSequenceExample]:
    rows = []
    prior = _belief(0, confidence=0.8)
    for class_index, operation in enumerate(EditOperation):
        target = _belief(class_index)
        steps = []
        for step in range(3):
            steps.append(
                ToolBeliefSequenceStep(
                    evidence_id=f"evidence-{operation.value}-{step}",
                    quality_result=GeoToolResult(
                        call_id=f"quality-{operation.value}-{step}",
                        tool=GeoToolName.IMAGE_QUALITY,
                        success=True,
                        outputs={"sharpness": 0.2},
                        cost=0.03,
                    ),
                    temporal_result=GeoToolResult(
                        call_id=f"temporal-{operation.value}-{step}",
                        tool=GeoToolName.TEMPORAL_CHANGE,
                        success=True,
                        outputs={"class_index": class_index, "changed_fraction": 0.2},
                        cost=0.15,
                    ),
                    individual_target_belief=target,
                    cumulative_target_belief=target,
                )
            )
        rows.append(
            ToolBeliefSequenceExample(
                sequence_id=f"sequence-{operation.value}",
                episode_id=f"episode-{operation.value}",
                split="val",
                initial_belief=prior,
                steps=steps,
                gt_edit=operation,
                metadata={"test_assets_read": False},
            )
        )
    return rows


def test_sequence_evaluation_uses_paired_path_and_exact_quality_noop() -> None:
    report, details = evaluate_sequences(
        _sequences(),
        _PerfectPairedUpdater(),
        minimum_macro_f1_gain=0.01,
        max_false_edit_increase=0.02,
        max_missed_edit_increase=0.02,
        max_noop_probability_drift=0.0,
        max_noop_confidence_drift=0.0,
        max_noop_geometry_drift=0.0,
    )

    assert report["protocol"]["primary_path"].startswith("quality_context")
    assert report["summaries"]["paired_3"]["macro_f1"] == 1.0
    assert report["quality_public_belief_delta_max"]["probability_l1"] == 0.0
    assert report["gates"]["passed"]
    assert len(details) == 12
