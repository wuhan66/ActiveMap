from __future__ import annotations

import numpy as np

from scripts.calibrate_sn7_operation_safe_commit import (
    apply_operation_safe_commit,
    choose_operation_thresholds,
    operation_threshold_candidates,
)
from scripts.calibrate_sn7_v5_safe_commit import apply_safe_commit, macro_aoi_summary


def _row(
    task: str,
    *,
    aoi: str,
    target: str,
    operation: str,
    confidence: float,
    final: float,
) -> dict[str, object]:
    return {
        "task_id": task,
        "aoi_id": aoi,
        "budget": 1.5,
        "target": target,
        "effective_operation": operation,
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


def _policy_rows() -> dict[tuple[str, float], dict[str, object]]:
    rows = [
        _row(
            "add-true",
            aoi="aoi-a",
            target="COMMIT:ADD",
            operation="ADD",
            confidence=0.9,
            final=0.9,
        ),
        _row(
            "add-false",
            aoi="aoi-b",
            target="REJECT",
            operation="ADD",
            confidence=0.8,
            final=0.2,
        ),
        _row(
            "delete-true",
            aoi="aoi-a",
            target="COMMIT:DELETE",
            operation="DELETE",
            confidence=0.6,
            final=0.9,
        ),
        _row(
            "delete-false",
            aoi="aoi-b",
            target="REJECT",
            operation="DELETE",
            confidence=0.5,
            final=0.2,
        ),
        _row(
            "reshape-true",
            aoi="aoi-a",
            target="COMMIT:RESHAPE",
            operation="RESHAPE",
            confidence=0.7,
            final=0.9,
        ),
        _row(
            "reshape-false",
            aoi="aoi-b",
            target="REJECT",
            operation="RESHAPE",
            confidence=0.6,
            final=0.2,
        ),
    ]
    return {(str(row["task_id"]), 1.5): row for row in rows}


def test_operation_calibration_learns_distinct_train_only_thresholds() -> None:
    result = choose_operation_thresholds(
        {"direct": _policy_rows(), "selected": _policy_rows()},
        replay_iou_threshold=0.99,
        require_topology=True,
        threshold_count=11,
        maximum_false_edit_rate=0.0,
    )
    thresholds = result["confidence_thresholds"]
    assert thresholds["ADD"] > thresholds["DELETE"]
    assert thresholds["ADD"] >= 0.8
    assert thresholds["DELETE"] >= 0.5


def test_operation_gate_uses_the_proposed_operation_threshold() -> None:
    row = _row(
        "delete-true",
        aoi="aoi-a",
        target="COMMIT:DELETE",
        operation="DELETE",
        confidence=0.65,
        final=0.9,
    )
    result = apply_operation_safe_commit(
        row,
        confidence_thresholds={"ADD": 0.9, "DELETE": 0.6, "RESHAPE": 0.8},
        replay_iou_threshold=0.99,
        require_topology=True,
    )
    assert result["safe_commit_accepted"] is True
    assert result["effective_operation"] == "DELETE"
    assert result["safe_commit_confidence_threshold"] == 0.6
    assert result["safe_commit_gate_mode"] == "operation_conditioned"


def test_vectorized_grid_matches_original_safe_commit_metrics() -> None:
    rows = [
        row
        for row in _policy_rows().values()
        if row["effective_operation"] == "ADD"
    ]
    thresholds = np.asarray([0.0, 0.8, 0.9, 1.0])
    vectorized = operation_threshold_candidates(
        rows,
        operation="ADD",
        thresholds=thresholds,
        replay_iou_threshold=0.99,
        require_topology=True,
    )
    for threshold, result in zip(thresholds, vectorized, strict=True):
        expected = macro_aoi_summary(
            [
                apply_safe_commit(
                    row,
                    confidence_threshold=float(threshold),
                    replay_iou_threshold=0.99,
                    require_topology=True,
                )
                for row in rows
            ]
        )
        assert result["metrics"] == expected
