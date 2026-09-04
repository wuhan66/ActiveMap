from scripts.calibrate_tool_belief_decision import calibrate_decision


def _summary(macro_f1: float, false_edit: float, missed_edit: float) -> dict[str, float]:
    return {
        "macro_f1": macro_f1,
        "false_edit_rate": false_edit,
        "missed_edit_rate": missed_edit,
    }


def test_calibration_selects_safe_keep_boundary() -> None:
    report = {
        "protocol": {"sequence_count": 4},
        "summaries": {
            "identity": _summary(0.4, 0.0, 1.0),
            "paired_1": _summary(0.4, 0.0, 1.0),
            "paired_3": _summary(0.4, 0.5, 0.0),
        },
        "gates": {
            "checks": {
                "noop_probability_safety": True,
                "noop_confidence_safety": True,
                "noop_geometry_safety": True,
            }
        },
    }
    details = []
    examples = (
        ("k1", "KEEP", [0.8, 0.1, 0.05, 0.05]),
        ("k2", "KEEP", [0.7, 0.2, 0.05, 0.05]),
        ("a1", "ADD", [0.4, 0.5, 0.05, 0.05]),
        ("a2", "ADD", [0.3, 0.6, 0.05, 0.05]),
    )
    for sequence_id, target, probabilities in examples:
        details.append(
            {
                "sequence_id": sequence_id,
                "step": 3,
                "target": target,
                "paired": "ADD",
                "paired_probabilities": probabilities,
                "paired_confidence": 0.9,
                "spent_cost": 0.54,
            }
        )

    result = calibrate_decision(
        report,
        details,
        minimum_macro_f1_gain=0.0,
        max_false_edit_increase=0.0,
        max_missed_edit_increase=0.0,
    )

    assert result["calibrated"]["accuracy"] == 1.0
    assert result["calibrated"]["false_edit_rate"] == 0.0
    assert result["calibrated"]["missed_edit_rate"] == 0.0
    assert result["gates"]["passed"]
