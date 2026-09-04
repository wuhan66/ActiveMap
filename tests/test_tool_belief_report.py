from scripts.render_tool_belief_report import intervention_rows


def test_intervention_rows_preserve_required_paper_metrics() -> None:
    metrics = {
        "accuracy": 0.7,
        "macro_f1": 0.6,
        "false_edit_rate": 0.1,
        "missed_edit_rate": 0.2,
        "mean_joint_utility": 0.3,
        "mean_tool_cost": 0.15,
        "expected_calibration_error": 0.08,
    }
    names = (
        "identity_one_step",
        "learned_one_step",
        "teacher_one_step",
        "learned_no_current_one_step",
        "learned_temporal_3",
        "learned_quality_3",
        "learned_quality_temporal_3",
    )
    rows = intervention_rows({"summaries": {name: metrics for name in names}})

    assert len(rows) == 7
    assert rows[1]["method"] == "learned_one_step"
    assert rows[1]["macro_f1"] == 0.6
    assert rows[1]["mean_joint_utility"] == 0.3


def test_intervention_rows_support_paired_sequence_report() -> None:
    metrics = {
        "accuracy": 0.7,
        "macro_f1": 0.6,
        "false_edit_rate": 0.1,
        "missed_edit_rate": 0.2,
        "mean_joint_utility": 0.3,
        "mean_tool_cost": 0.15,
        "expected_calibration_error": 0.08,
    }
    names = (
        "identity",
        "paired_1",
        "paired_2",
        "paired_3",
        "no_quality_3",
        "no_current_3",
        "teacher_3",
    )

    rows = intervention_rows({"summaries": {name: metrics for name in names}})

    assert [row["method"] for row in rows] == list(names)
