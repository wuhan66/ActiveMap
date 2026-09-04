import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_updater_conditioned_candidate_risk_support.py"
SPEC = importlib.util.spec_from_file_location("candidate_risk_audit", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_candidate_risk_summary_counts_counterfactual_labels_by_grouped_split() -> None:
    rows = [
        {
            "sample_id": "a", "split": "train", "evidence_ids": ["x", "y"],
            "false_edit_risks": [0.01, 0.22],
            "metadata": {"source_episode": "ep-a", "executable_outcomes": {
                "x": {"false_edit": False, "missed_edit": True, "quality_gain": 0.0},
                "y": {"false_edit": True, "missed_edit": False, "quality_gain": 0.2},
            }},
        },
        {
            "sample_id": "b", "split": "val", "evidence_ids": ["z"],
            "false_edit_risks": [0.81],
            "metadata": {"source_episode": "ep-b", "executable_outcomes": {
                "z": {"false_edit": False, "missed_edit": False, "quality_gain": 0.0},
            }},
        },
    ]
    summary = MODULE.summarize(rows)
    assert summary["split_counts"]["train"] == {
        "candidates": 2, "false_edits": 1, "missed_edits": 1, "positive_quality": 1,
    }
    assert summary["cross_split_group_overlap"] == 0
