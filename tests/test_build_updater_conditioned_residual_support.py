import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_updater_conditioned_residual_support.py"
SPEC = importlib.util.spec_from_file_location("residual_support", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summary_requires_group_disjoint_splits_and_counts_rare_target() -> None:
    rows = [
        {"split": "train", "source_episode": "a", "differs": True, "features": [1.0], "targets": {"false_edit_regression": True, "missed_edit_improvement": False}},
        {"split": "val", "source_episode": "b", "differs": True, "features": [1.0], "targets": {"false_edit_regression": False, "missed_edit_improvement": True}},
        {"split": "train", "source_episode": "c", "differs": False, "features": [1.0], "targets": {"false_edit_regression": False, "missed_edit_improvement": False}},
    ]
    summary = MODULE._summary(rows)
    assert summary["cross_split_group_overlap"] == 0
    assert summary["disagreement_count"] == 2
    assert summary["false_edit_regression_count"] == 1
    assert summary["missed_edit_improvement_count"] == 1
