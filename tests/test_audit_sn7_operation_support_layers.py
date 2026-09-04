from __future__ import annotations

import pytest

from scripts.audit_sn7_operation_support_layers import (
    _unique_index,
    bottleneck_flags,
    compact_state,
    compact_writeback,
    operation,
    summarize_rows,
)


def _state(*, split: str = "val", test_assets_read: bool = False) -> dict:
    return {
        "sample_id": "sample-a",
        "split": split,
        "edit_type": "ADD",
        "evidence_ids": ["candidate"],
        "oracle_utilities": [0.3],
        "stop_utility": 0.0,
        "metadata": {
            "source_episode": "episode-a",
            "aoi_id": "aoi-a",
            "budget": 1.5,
            "oracle_step": 0,
            "gt_edit": "ADD",
            "initial_evidence_id": "anchor",
            "test_assets_read": test_assets_read,
            "executable_outcomes": {
                "anchor": {
                    "final_raster_iou": 0.5,
                    "false_edit": False,
                    "missed_edit": True,
                    "wrong_edit": False,
                    "predicted_operation": "KEEP",
                },
                "candidate": {
                    "final_raster_iou": 0.8,
                    "false_edit": False,
                    "missed_edit": False,
                    "wrong_edit": False,
                    "predicted_operation": "ADD",
                },
            },
            "evidence_predictions": {
                "anchor": {"gated_edit": "KEEP"},
                "candidate": {"gated_edit": "ADD"},
            },
        },
    }


def _joined_row() -> dict:
    return {
        "seed": 1,
        "source_episode": "episode-a",
        "aoi_id": "aoi-a",
        "target_operation": "ADD",
        "candidate_count": 2,
        "distinct_outcome_count": 2,
        "target_candidate_count": 1,
        "target_candidate_covered": True,
        "safe_candidate_count": 2,
        "safe_map_headroom": 0.3,
        "safe_map_recovery_possible": True,
        "missed_edit_recovery_possible": True,
        "oracle_acquire": True,
        "policy_acquire": True,
        "policy_true_call": True,
        "policy_false_call": False,
        "policy_harmful_call": False,
        "exact_oracle_candidate": True,
        "raw_writeback_changed": True,
        "raw_raster_iou_gain": 0.3,
        "safe_commit_accepted": False,
        "safe_writeback_changed": False,
        "safe_raster_iou_gain": 0.0,
        "raw_false_edit": False,
        "safe_false_edit": False,
        "raw_missed_edit": False,
        "safe_missed_edit": True,
    }


def test_compact_state_recovers_candidate_and_label_support(tmp_path) -> None:
    row = compact_state(_state(), source=tmp_path / "states.jsonl", line_number=1)
    assert row is not None
    assert row["target_operation"] == "ADD"
    assert row["target_candidate_covered"] is True
    assert row["safe_map_headroom"] == pytest.approx(0.3)
    assert row["missed_edit_recovery_possible"] is True
    assert row["oracle_acquire"] is True


def test_summary_localizes_safe_commit_conversion_loss() -> None:
    summary = summarize_rows([_joined_row()])
    flags = bottleneck_flags(
        summary,
        minimum_candidate_recovery=0.1,
        minimum_label_rate=0.05,
        minimum_policy_recall=0.5,
        minimum_conversion=0.1,
    )
    assert summary["raw_positive_gain_given_acquire"] == 1.0
    assert summary["safe_positive_gain_given_acquire"] == 0.0
    assert flags["candidate_interface_weak"] is False
    assert flags["safe_commit_blocks_positive_raw_gain"] is True


def test_audit_rejects_test_provenance(tmp_path) -> None:
    with pytest.raises(ValueError, match="test provenance"):
        compact_state(
            _state(test_assets_read=True),
            source=tmp_path / "states.jsonl",
            line_number=1,
        )


def test_writeback_identity_includes_budget(tmp_path) -> None:
    base = {
        "task_id": "shared-task",
        "split": "val",
        "target": "COMMIT:ADD",
        "operation": "ADD",
        "effective_operation": "ADD",
        "writeback_changed": True,
        "raster_iou_gain": 0.1,
        "test_assets_read": False,
    }
    rows = [
        compact_writeback(
            {**base, "budget": budget},
            source=tmp_path / "writeback.jsonl",
            line_number=index,
        )
        for index, budget in enumerate((1.5, 3.0), start=1)
    ]
    index = _unique_index(rows, "key", source=tmp_path / "writeback.jsonl")
    assert set(index) == {("shared-task", 1.5), ("shared-task", 3.0)}


@pytest.mark.parametrize(
    ("value", "expected"),
    [("COMMIT:DELETE", "DELETE"), ("REJECT", "KEEP"), ("reshape", "RESHAPE")],
)
def test_operation_normalization(value: str, expected: str) -> None:
    assert operation(value) == expected
