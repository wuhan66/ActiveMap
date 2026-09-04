from scripts.analyze_tool_belief_stopping import analyze_stopping


def _metrics(utility: float, cost: float = 0.0) -> dict[str, float]:
    return {
        "macro_f1": 0.5,
        "false_edit_rate": 0.1,
        "missed_edit_rate": 0.2,
        "mean_confidence": 0.8,
        "mean_tool_cost": cost,
        "mean_terminal_reward": utility + cost,
        "mean_joint_utility": utility,
    }


def test_stopping_oracle_selects_tools_only_when_they_improve_utility() -> None:
    report = {
        "summaries": {
            "identity": _metrics(0.5),
            "paired_1": _metrics(0.2, 0.1),
            "paired_2": _metrics(0.1, 0.2),
            "paired_3": _metrics(0.0, 0.3),
        }
    }
    details = []
    for sequence, target, baseline, predictions in (
        ("keep", "KEEP", "KEEP", ["ADD", "ADD", "ADD"]),
        ("add", "ADD", "KEEP", ["ADD", "ADD", "ADD"]),
    ):
        for step, prediction in enumerate(predictions, start=1):
            details.append(
                {
                    "sequence_id": sequence,
                    "step": step,
                    "target": target,
                    "baseline": baseline,
                    "paired": prediction,
                    "paired_confidence": 0.9,
                    "spent_cost": 0.1 * step,
                }
            )

    result = analyze_stopping(report, details)

    assert result["best_fixed_stage"] == "step_0"
    assert result["oracle_selected_step_counts"] == {"0": 1, "1": 1, "2": 0, "3": 0}
    assert result["oracle_utility_gain_over_identity"] > 0.0
