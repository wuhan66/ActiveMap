import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_updater_conditioned_disagreements.py"
SPEC = importlib.util.spec_from_file_location("updater_conditioned_disagreements", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summary_reports_safety_regressions_only_within_disagreements() -> None:
    rows = [
        {
            "edit_type": "DELETE",
            "stale_choice": {"stop": True},
            "refreshed_choice": {"stop": False},
            "utility_delta": 0.1,
            "raster_iou_delta": 0.0,
            "stale_false_edit": False,
            "refreshed_false_edit": True,
            "stale_missed_edit": True,
            "refreshed_missed_edit": False,
        },
        {
            "edit_type": "ADD",
            "stale_choice": {"stop": False},
            "refreshed_choice": {"stop": True},
            "utility_delta": -0.1,
            "raster_iou_delta": -0.01,
            "stale_false_edit": False,
            "refreshed_false_edit": False,
            "stale_missed_edit": False,
            "refreshed_missed_edit": True,
        },
    ]
    summary = MODULE._summary(rows, sample_count=10)
    assert summary["disagreement_count"] == 2
    assert summary["disagreement_rate"] == 0.2
    assert summary["by_action_pair"] == {"ACQUIRE->STOP": 1, "STOP->ACQUIRE": 1}
    assert summary["rates_within_disagreements"]["false_edit_regressed"] == 0.5
    assert summary["rates_within_disagreements"]["missed_edit_improved"] == 0.5
