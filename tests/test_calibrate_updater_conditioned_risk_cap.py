import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "calibrate_updater_conditioned_risk_cap.py"
SPEC = importlib.util.spec_from_file_location("risk_cap_calibration", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _candidate(cap: float, *, false_edit: float, missed: float, utility: float) -> dict:
    return {
        "risk_cap": cap,
        "metrics": {
            "false_edit_rate": false_edit,
            "missed_edit_rate": missed,
            "quality_cost_utility": utility,
            "final_raster_iou": utility,
            "mean_additional_cost": 0.1,
        },
    }


def test_select_cap_enforces_stale_false_edit_ceiling_before_missed_edit_gain() -> None:
    selected = MODULE.select_cap(
        [
            _candidate(0.2, false_edit=0.10, missed=0.30, utility=0.10),
            _candidate(0.4, false_edit=0.13, missed=0.10, utility=0.20),
            _candidate(0.6, false_edit=0.10, missed=0.20, utility=0.15),
        ],
        stale_false_edit_rate=0.10,
        max_false_edit_increase=0.0,
    )
    assert selected["risk_cap"] == 0.6


def test_select_cap_allows_predeclared_false_edit_tolerance() -> None:
    selected = MODULE.select_cap(
        [
            _candidate(0.2, false_edit=0.10, missed=0.30, utility=0.10),
            _candidate(0.4, false_edit=0.12, missed=0.10, utility=0.20),
        ],
        stale_false_edit_rate=0.10,
        max_false_edit_increase=0.02,
    )
    assert selected["risk_cap"] == 0.4
