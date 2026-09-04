from __future__ import annotations

import pytest

from scripts.calibrate_sn7_v5_common_safe_commit import POLICIES
from scripts.calibrate_sn7_v5_safe_commit import choose_common_threshold


def _row(
    task: str, *, aoi: str, target: str, confidence: float, final: float
) -> dict[str, object]:
    return {
        "task_id": task,
        "aoi_id": aoi,
        "budget": 1.5,
        "target": target,
        "effective_operation": "ADD",
        "writeback_changed": True,
        "fused_confidence": confidence,
        "vector_replay_iou": 1.0,
        "vector_delta_topology_valid": True,
        "prior_raster_iou": 0.5,
        "raster_iou": final,
        "raster_iou_gain": final - 0.5,
        "spent_cost": 0.2,
        "semantic_tool_called": True,
        "split": "train",
        "test_assets_read": False,
    }


def _matched_policy_rows(confidence_offset: float) -> dict[tuple[str, float], dict[str, object]]:
    return {
        ("keep", 1.5): _row(
            "keep", aoi="aoi-a", target="REJECT", confidence=0.1 + confidence_offset, final=0.2
        ),
        ("add", 1.5): _row(
            "add", aoi="aoi-b", target="COMMIT:ADD", confidence=0.9, final=0.9
        ),
    }


def test_common_gate_uses_all_deployable_policies_without_policy_labels() -> None:
    by_policy = {
        policy: _matched_policy_rows(index * 0.001)
        for index, policy in enumerate(POLICIES)
    }

    result = choose_common_threshold(
        by_policy,
        expected_policies=POLICIES,
        replay_iou_threshold=0.99,
        require_topology=True,
        threshold_count=11,
    )

    assert result["policies"] == list(POLICIES)
    assert result["selected"]["metrics"]["false_edit_rate"] == 0.0
    assert result["selected"]["confidence_threshold"] >= 0.1


def test_common_gate_rejects_policy_specific_support() -> None:
    by_policy = {
        policy: _matched_policy_rows(index * 0.001)
        for index, policy in enumerate(POLICIES)
    }
    del by_policy["generic"][("add", 1.5)]

    with pytest.raises(ValueError, match="lack paired support"):
        choose_common_threshold(
            by_policy,
            expected_policies=POLICIES,
            replay_iou_threshold=0.99,
            require_topology=True,
            threshold_count=11,
        )
