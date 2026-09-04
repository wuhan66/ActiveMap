from __future__ import annotations

from activemap.models import EditOperation
from activemap.selector_records import SelectorSample
from scripts.build_episode_sequential_selector_sft import (
    _evaluation_index_row,
    _shortlist_indices,
    _state,
    _target,
)
from tests.test_episode_support_tools import _episode


def _sample(utilities: list[float], stop: float = 0.0) -> SelectorSample:
    return SelectorSample(
        sample_id="sample",
        split="train",
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
        oracle_utilities=utilities,
        stop_utility=stop,
        metadata={
            "source_episode": "train-RESHAPE-a-0",
            "initial_evidence_id": "e-0",
            "selected_evidence_ids": ["e-0"],
            "budget": 3.0,
            "oracle_step": 0,
            "aoi_id": "a",
            "gt_edit": "RESHAPE",
        },
    )


def test_active_catalog_state_hides_targets_and_selects_best_evidence() -> None:
    episode = _episode("train", EditOperation.RESHAPE, "a", 0)
    episode.evidence_catalog.append(
        episode.evidence_catalog[0].model_copy(
            update={"evidence_id": "e-1", "timestamp": "2026_02", "scale": 2}
        )
    )
    sample = _sample([-0.1, 0.4])
    state = _state(sample, episode, "snapshot")
    action, advantage = _target(sample)
    assert "gt_edit" not in state
    assert "oracle_utilities" not in state
    assert len(state["candidate_evidence"]) == 2
    assert set(state["candidate_evidence"][0]) == {
        "evidence_id",
        "timestamp",
        "scale",
        "clear_fraction",
        "temporal_offset_normalized",
        "cost",
    }
    assert action.selection.value == "ACQUIRE"
    assert action.evidence_id == state["candidate_evidence"][1]["evidence_id"]
    assert advantage == 0.4


def test_active_catalog_target_stops_when_best_gain_is_nonpositive() -> None:
    action, advantage = _target(_sample([-0.2, 0.0]))
    assert action.selection.value == "STOP"
    assert action.evidence_id is None
    assert advantage == 0.0


def test_shortlist_is_deterministic_and_never_reads_oracle_utility() -> None:
    sample = _sample([-100.0, 100.0])
    sample.evidence_ids.extend([f"extra-{index}" for index in range(8)])
    sample.evidence_features.extend(
        [[1.0, 0.0, index / 8, index / 8, 0.0, 0.0, 1.0, 0.5, 0.5, 0.1, 0.1, 0.2, 1.0]
         for index in range(8)]
    )
    sample.evidence_costs.extend([1.0] * 8)
    sample.false_edit_risks.extend([0.0] * 8)
    sample.oracle_utilities.extend([float(index) for index in range(8)])
    first = _shortlist_indices(sample, 5)
    sample.oracle_utilities = list(reversed(sample.oracle_utilities))
    second = _shortlist_indices(sample, 5)
    assert first == second
    assert len(first) == 5


def test_evaluation_index_is_separate_and_uses_public_candidate_ids() -> None:
    sample = _sample([-0.1, 0.4])
    action, _ = _target(sample)
    row = _evaluation_index_row(
        sample,
        example_id="example",
        task_id="task",
        candidate_indices=[0, 1],
        action=action,
    )
    assert row["model_visible"] is False
    assert row["target_selection"] == "ACQUIRE"
    assert row["target_evidence_id"] == row["candidates"][1]["evidence_id"]
    assert row["target_utility"] == 0.4
