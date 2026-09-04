import pytest

from scripts.audit_selection_safe_commit_2x2 import Gate, analyze, apply_gate


def row(task, *, target="COMMIT:ADD", changed=True, confidence=0.9, after=0.8):
    return {
        "task_id": task,
        "aoi_id": "aoi-1" if task == "one" else "aoi-2",
        "budget": 3.0,
        "target": target,
        "effective_operation": "ADD" if changed else "KEEP",
        "writeback_changed": changed,
        "fused_confidence": confidence,
        "vector_replay_iou": 1.0,
        "vector_delta_topology_valid": True,
        "prior_raster_iou": 0.5,
        "raster_iou": after,
        "spent_cost": 1.0,
        "semantic_tool_called": True,
        "split": "val",
        "test_assets_read": False,
    }


def test_safe_commit_reverts_to_prior_and_marks_missed_edit_when_rejected():
    raw = row("one", confidence=0.1)
    raw.update({"map_quality_before": raw["prior_raster_iou"], "map_quality_after": raw["raster_iou"]})
    gated = apply_gate(raw, Gate(0.7, 0.99))

    assert gated["gate_accepted"] is False
    assert gated["map_quality_after"] == pytest.approx(0.5)
    assert gated["missed_edit"] is True


def test_2x2_keeps_selection_and_commit_factors_matched():
    support = {("one", 3.0): row("one"), ("two", 3.0): row("two", changed=False)}
    learned = {("one", 3.0): row("one", after=0.9), ("two", 3.0): row("two", changed=False)}
    for rows in (support, learned):
        for value in rows.values():
            value.update({"map_quality_before": value["prior_raster_iou"], "map_quality_after": value["raster_iou"]})
    result = analyze(
        {
            "seed-a": {"notool": support, "benefit": learned},
            "seed-b": {"notool": support, "benefit": learned},
        },
        no_extra_policy="notool",
        learned_policy="benefit",
        gate=Gate(0.7, 0.99),
        repetitions=30,
        seed=7,
    )

    assert result["task_budget_count"] == 2
    assert result["cells"]["benefit__safe_commit"]["map_quality_after"]["mean"] == pytest.approx(0.85)
    assert result["paired_comparisons"]["selection_gain_safe_commit"]["map_quality_after"]["observed_delta"] == pytest.approx(0.05)
