from pathlib import Path

from activemap.agent.active_catalog_joint import joint_transition_from_agent_transition
from activemap.agent.identifiers import public_evidence_id
from activemap.agent.records import (
    AgentAction,
    AgentActionType,
    AgentBelief,
    AgentCandidate,
    AgentObservation,
    AgentTransition,
)
from activemap.models import (
    CandidateHypothesis,
    EditOperation,
    EditRecord,
    EpisodeRecord,
    EvidenceItem,
    GeoJSONGeometry,
)
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from scripts.build_active_catalog_grounded_tool_data import build_grounded_examples


class _FakeRegistry:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, call):
        self.calls.append(call)
        outputs = (
            {"valid_fraction": 1.0, "saturation_fraction": 0.0, "sharpness": 0.4}
            if call.tool == GeoToolName.IMAGE_QUALITY
            else {"changed_fraction": 0.25}
        )
        return GeoToolResult(
            call_id=call.call_id,
            tool=call.tool,
            success=True,
            outputs=outputs,
            cost=0.03 if call.tool == GeoToolName.IMAGE_QUALITY else 0.15,
        )


class _PartiallyFailingRegistry(_FakeRegistry):
    def execute(self, call):
        if "acquired" in str(call.inputs):
            return GeoToolResult(
                call_id=call.call_id,
                tool=call.tool,
                success=False,
                cost=0.0,
                error="fixture has no valid pixels",
            )
        return super().execute(call)


def _belief(values):
    return AgentBelief(
        edit_probabilities=values,
        confidence=0.8,
        uncertainty=0.3,
    )


def _joint_transition(budget: float):
    anchor = public_evidence_id("anchor")
    acquired = public_evidence_id("acquired")
    before = AgentObservation(
        task_id="public-task",
        split="train",
        step=0,
        initial_budget=budget,
        remaining_budget=budget,
        spent_cost=0.0,
        selected_evidence_ids=[anchor],
        belief=_belief([0.7, 0.1, 0.1, 0.1]),
        candidates=[
            AgentCandidate(
                evidence_id=acquired,
                cost=1.0,
                selector_score=0.5,
                features=[0.0] * 13,
            )
        ],
    )
    after = before.model_copy(
        update={
            "step": 1,
            "remaining_budget": budget - 1.0,
            "spent_cost": 1.0,
            "selected_evidence_ids": [anchor, acquired],
            "belief": _belief([0.2, 0.7, 0.05, 0.05]),
            "candidates": [],
        }
    )
    transition = AgentTransition(
        observation=before,
        action=AgentAction(action=AgentActionType.ACQUIRE, evidence_id=acquired),
        reward=0.2,
        done=False,
        next_observation=after,
    )
    return joint_transition_from_agent_transition(
        transition,
        source_episode="episode",
        aoi_id="aoi-train",
        policy_snapshot="adapter",
        target_edit=EditOperation.ADD,
        selected_by_model=True,
        operation_update_threshold=0.69,
    )


def _episode(anchor_path: Path, acquired_path: Path) -> EpisodeRecord:
    geometry = GeoJSONGeometry(
        type="Polygon",
        coordinates=[[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    )
    return EpisodeRecord(
        episode_id="episode",
        aoi_id="aoi-train",
        split="train",
        source_dataset="fixture",
        map_before="unused-before.geojson",
        target_map="unused-after.geojson",
        hypothesis=CandidateHypothesis(
            op=EditOperation.ADD,
            object_id="object",
            geometry=geometry,
            source="fixture",
        ),
        gt_edit=EditRecord(
            op=EditOperation.ADD,
            object_id="object",
            geometry=geometry,
        ),
        evidence_catalog=[
            EvidenceItem(
                evidence_id="anchor",
                timestamp="2020_01",
                region=(0, 0, 16, 16),
                scale=1,
                image_path=str(anchor_path),
                clear_fraction=1.0,
                cost=1.0,
            ),
            EvidenceItem(
                evidence_id="acquired",
                timestamp="2020_02",
                region=(0, 0, 16, 16),
                scale=1,
                image_path=str(acquired_path),
                clear_fraction=0.9,
                cost=1.0,
            ),
        ],
        is_synthetic=False,
        derivation_version="fixture-v1",
    )


def test_grounded_tools_use_only_model_selected_evidence_and_cache(tmp_path: Path):
    anchor_path = tmp_path / "anchor.tif"
    acquired_path = tmp_path / "acquired.tif"
    rows = [_joint_transition(3.0), _joint_transition(4.0)]
    assert all(row is not None for row in rows)
    registry = _FakeRegistry()

    examples, summary = build_grounded_examples(
        rows,
        [_episode(anchor_path, acquired_path)],
        artifact_root=tmp_path / "artifacts",
        out_size=16,
        label_smoothing=0.05,
        registry=registry,
    )

    assert len(examples) == 2
    assert summary["explicit_geospatial_tool_calls"] is True
    assert summary["unique_grounded_tool_executions"] == 1
    assert summary["cache_reuse_count"] == 1
    assert len(registry.calls) == 2
    assert examples[0].evidence_id == public_evidence_id("acquired")
    assert examples[0].quality_result.outputs["evidence_id"] == examples[0].evidence_id
    assert examples[0].temporal_result.outputs["changed_fraction"] > 0.0
    assert examples[0].metadata["selected_by_model"] is True
    assert examples[0].metadata["operation_update_threshold"] == 0.69


def test_grounded_tools_skip_failed_pairs_and_report_them(tmp_path: Path):
    anchor_path = tmp_path / "anchor.tif"
    acquired_path = tmp_path / "acquired.tif"
    successful = _joint_transition(3.0)
    failed = _joint_transition(4.0).model_copy(
        update={"transition_id": "failed-transition"}
    )
    failed_episode = _episode(anchor_path, acquired_path).model_copy(
        update={
            "episode_id": "failed-episode",
                "evidence_catalog": [
                    item.model_copy(
                        update={
                            "evidence_id": f"failed-{item.evidence_id}",
                            "image_path": str(
                                tmp_path / f"failed-{item.evidence_id}.tif"
                            ),
                        }
                    )
                for item in _episode(anchor_path, acquired_path).evidence_catalog
            ],
        }
    )
    failed = failed.model_copy(
        update={
            "source_episode": "failed-episode",
            "evidence_id": public_evidence_id("failed-acquired"),
            "action": failed.action.model_copy(
                update={"evidence_id": public_evidence_id("failed-acquired")}
            ),
            "prior_observation": failed.prior_observation.model_copy(
                update={
                    "selected_evidence_ids": [public_evidence_id("failed-anchor")],
                    "candidates": [
                        failed.prior_observation.candidates[0].model_copy(
                            update={
                                "evidence_id": public_evidence_id("failed-acquired")
                            }
                        )
                    ],
                }
            ),
            "post_acquisition_observation": failed.post_acquisition_observation.model_copy(
                update={
                    "selected_evidence_ids": [
                        public_evidence_id("failed-anchor"),
                        public_evidence_id("failed-acquired"),
                    ]
                }
            ),
        }
    )

    class _OneFailureRegistry(_FakeRegistry):
        def execute(self, call):
            if "failed" in str(call.inputs):
                return GeoToolResult(
                    call_id=call.call_id,
                    tool=call.tool,
                    success=False,
                    cost=0.0,
                    error="fixture has no valid pixels",
                )
            return super().execute(call)

    examples, summary = build_grounded_examples(
        [successful, failed],
        [_episode(anchor_path, acquired_path), failed_episode],
        artifact_root=tmp_path / "artifacts",
        out_size=16,
        label_smoothing=0.05,
        registry=_OneFailureRegistry(),
    )

    assert len(examples) == 1
    assert summary["skipped_failed_transitions"] == 1
    assert summary["failed_transition_rate"] == 0.5
    assert summary["unique_failed_tool_executions"] == 1
    assert summary["failure_reasons"]
