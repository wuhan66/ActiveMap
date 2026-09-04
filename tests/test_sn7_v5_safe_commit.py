from __future__ import annotations

from scripts.calibrate_sn7_v5_safe_commit import apply_safe_commit, choose_threshold


def row(task: str, *, aoi: str, target: str, confidence: float, final: float) -> dict:
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


def test_safe_commit_reverts_an_unsafe_low_confidence_writeback():
    raw = row("keep", aoi="aoi-a", target="REJECT", confidence=0.1, final=0.2)

    safe = apply_safe_commit(
        raw,
        confidence_threshold=0.5,
        replay_iou_threshold=0.99,
        require_topology=True,
    )

    assert safe["safe_commit_accepted"] is False
    assert safe["effective_operation"] == "KEEP"
    assert safe["raster_iou"] == 0.5
    assert safe["false_edit"] is False


def test_train_only_calibration_selects_a_threshold_that_removes_false_edits():
    direct = {
        ("keep", 1.5): row("keep", aoi="aoi-a", target="REJECT", confidence=0.1, final=0.2),
        ("add", 1.5): row("add", aoi="aoi-b", target="COMMIT:ADD", confidence=0.9, final=0.9),
    }
    selected = {
        ("keep", 1.5): row("keep", aoi="aoi-a", target="REJECT", confidence=0.2, final=0.2),
        ("add", 1.5): row("add", aoi="aoi-b", target="COMMIT:ADD", confidence=0.8, final=0.9),
    }

    result = choose_threshold(
        {"direct": direct, "selected": selected},
        replay_iou_threshold=0.99,
        require_topology=True,
        threshold_count=11,
    )

    assert result["selected"]["confidence_threshold"] >= 0.1
    assert result["selected"]["metrics"]["false_edit_rate"] == 0.0
