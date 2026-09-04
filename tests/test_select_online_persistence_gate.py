import pytest

from scripts.select_online_persistence_gate import select


def summary(false_edit, quality, missed=0.1):
    branches = {
        "independent_reset": {},
        "carry_always_commit": {"false_edit_rate": 0.2},
        "carry_safe_commit": {
            "false_edit_rate": false_edit,
            "mean_step_raster_iou": quality,
            "mean_final_chain_raster_iou": quality,
            "missed_edit_rate": missed,
        },
    }
    return {
        "schema_version": "activemap-online-persistent-maintenance-v1",
        "split": "train",
        "test_assets_read": False,
        "protocol": {"safe_commit_gate": {"confidence_threshold": 0.5}},
        "metrics": {"branches": branches},
    }


def test_gate_selection_uses_train_safety_constraint_then_quality(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text("{}")
    second.write_text("{}")
    result = select(
        [(0.5, summary(0.15, 0.7), first), (0.7, summary(0.1, 0.68), second)],
        false_edit_fraction=0.8,
    )
    assert result["selected"]["threshold"] == pytest.approx(0.5)


def test_gate_selection_refuses_thresholds_without_required_safety_reduction(tmp_path):
    path = tmp_path / "a.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="no train threshold"):
        select([(0.5, summary(0.17, 0.7), path)], false_edit_fraction=0.8)
