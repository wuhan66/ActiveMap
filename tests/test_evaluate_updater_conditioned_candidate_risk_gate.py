import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_updater_conditioned_candidate_risk_gate.py"
SPEC = importlib.util.spec_from_file_location("candidate_risk_gate", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_policy_threshold_requires_safety_utility_and_iou_non_regression() -> None:
    rows = [
        {"risk": 0.9, "stale": {"utility": 0.0, "raster_iou": 0.5, "false_edit": 0.0, "missed_edit": 1.0, "cost": 0.0}, "refreshed": {"utility": 0.2, "raster_iou": 0.5, "false_edit": 1.0, "missed_edit": 0.0, "cost": 1.0}},
        {"risk": 0.1, "stale": {"utility": 0.0, "raster_iou": 0.5, "false_edit": 0.0, "missed_edit": 1.0, "cost": 0.0}, "refreshed": {"utility": 0.1, "raster_iou": 0.6, "false_edit": 0.0, "missed_edit": 0.0, "cost": 1.0}},
    ]
    result = MODULE.select_threshold(rows, max_false_edit_increase=0.0)
    assert result["status"] == "complete"
    assert result["selected"]["threshold"] > 0.1
    assert result["selected"]["metrics"]["false_edit"] == 0.0
