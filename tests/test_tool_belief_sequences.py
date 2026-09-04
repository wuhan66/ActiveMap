import pytest

from activemap.agent.identifiers import public_evidence_id, public_task_id
from activemap.agent.tool_belief_data import ToolBeliefExample
from activemap.agent.tools import CounterfactualBeliefUpdater
from activemap.features import EVIDENCE_DIM, HYPOTHESIS_DIM, STATE_DIM
from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import (
    CandidateHypothesis,
    EditOperation,
    EditRecord,
    EpisodeRecord,
    EvidenceItem,
)
from activemap.selector_records import SelectorSample
from scripts.build_tool_belief_sequences import build_sequences


def _prediction(probabilities: list[float], confidence: float) -> dict[str, object]:
    return {
        "edit_probabilities": probabilities,
        "confidence": confidence,
        "geometry_delta": [0.0] * 8,
    }


def _sample() -> SelectorSample:
    evidence_ids = ["history-1", "history-2", "history-3"]
    return SelectorSample(
        sample_id="sequence-sample",
        split="train",
        edit_type=EditOperation.DELETE,
        hypothesis_features=[0.0] * HYPOTHESIS_DIM,
        state_features=[0.0] * STATE_DIM,
        evidence_ids=evidence_ids,
        evidence_features=[[0.0] * EVIDENCE_DIM for _ in evidence_ids],
        evidence_costs=[0.5] * 3,
        false_edit_risks=[0.0] * 3,
        oracle_utilities=[0.0] * 3,
        metadata={
            "source_episode": "raw-episode",
            "initial_evidence_id": "initial",
            "selected_evidence_ids": ["initial"],
            "oracle_step": 0,
            "gt_edit": "DELETE",
            "evidence_predictions": {
                "initial": _prediction([0.8, 0.1, 0.05, 0.05], 0.7),
                "history-1": _prediction([0.1, 0.1, 0.75, 0.05], 0.9),
                "history-2": _prediction([0.2, 0.1, 0.6, 0.1], 0.8),
                "history-3": _prediction([0.1, 0.1, 0.7, 0.1], 0.85),
            },
        },
    )


def _episode() -> EpisodeRecord:
    evidence = [
        EvidenceItem(
            evidence_id=evidence_id,
            timestamp=f"202{index}-01-01",
            region=(0, 0, 16, 16),
            scale=1,
            image_path=f"/{evidence_id}.jpg",
            clear_fraction=1.0,
            cost=0.5,
        )
        for index, evidence_id in enumerate(
            ["initial", "history-1", "history-2", "history-3"]
        )
    ]
    return EpisodeRecord(
        episode_id="raw-episode",
        split="train",
        source_dataset="MUNO21",
        map_before="before.geojson",
        target_map="target.geojson",
        hypothesis=CandidateHypothesis(
            op=EditOperation.DELETE,
            object_id="road-1",
            source="editable_prior",
            confidence=0.5,
        ),
        evidence_catalog=evidence,
        gt_edit=EditRecord(op=EditOperation.DELETE, object_id="road-1"),
        is_synthetic=False,
        derivation_version="test",
    )


def test_sequence_builder_reuses_real_results_and_computes_cumulative_teacher() -> None:
    sample = _sample()
    episode = _episode()
    updater = CounterfactualBeliefUpdater(sample)
    prior = updater.fuse(["initial"])
    rows = []
    for index, evidence_id in enumerate(["history-1", "history-2", "history-3"]):
        individual = updater.fuse(["initial", evidence_id])
        for tool, target, outputs, cost, kind in (
            (
                GeoToolName.IMAGE_QUALITY,
                prior,
                {"sharpness": 0.1 + index},
                0.03,
                "observational_noop",
            ),
            (
                GeoToolName.TEMPORAL_CHANGE,
                individual,
                {"changed_fraction": 0.2 + index / 10},
                0.15,
                "teacher_belief_update",
            ),
        ):
            rows.append(
                ToolBeliefExample(
                    record_id=f"record-{index}-{tool.value}",
                    episode_id=public_task_id(episode.episode_id),
                    split="train",
                    evidence_id=public_evidence_id(evidence_id),
                    prior_belief=prior,
                    tool_result=GeoToolResult(
                        call_id=f"call-{index}-{tool.value}",
                        tool=tool,
                        success=True,
                        outputs=outputs,
                        cost=cost,
                    ),
                    target_belief=target,
                    gt_edit=EditOperation.DELETE,
                    metadata={"target_kind": kind, "test_assets_read": False},
                )
            )

    sequences, summary = build_sequences(
        rows, [episode], {episode.episode_id: sample}
    )

    assert summary["sequence_count"] == 1
    assert summary["step_count"] == 3
    assert summary["test_assets_read"] is False
    sequence = sequences[0]
    assert len(sequence.steps) == 3
    assert sequence.steps[0].cumulative_target_belief == sequence.steps[0].individual_target_belief
    assert sequence.steps[1].cumulative_target_belief != sequence.steps[1].individual_target_belief
    assert sequence.steps[2].cumulative_target_belief.predicted_edit == EditOperation.DELETE


def test_test_belief_sequences_require_frozen_authorization() -> None:
    sample = _sample().model_copy(update={"split": "test"})
    episode = _episode().model_copy(update={"split": "test"})
    with pytest.raises(PermissionError, match="run_frozen_paper_test"):
        build_sequences([], [episode], {episode.episode_id: sample})
