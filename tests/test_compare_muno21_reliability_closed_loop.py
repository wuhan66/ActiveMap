from scripts.compare_muno21_reliability_closed_loop import compare


def _summary(belief_hash: str) -> dict:
    fingerprints = {
        key: {"sha256": key}
        for key in ("states", "episodes", "adapter", "tool_supervision", "tool_need_gate")
    }
    fingerprints["selector_checkpoints"] = [{"sha256": "selector"}]
    fingerprints["tool_belief_checkpoint"] = {"sha256": belief_hash}
    return {
        "protocol": {
            "split": "val",
            "test_assets_read": False,
            "budgets": [3.0],
            "evaluation_seed": 7,
            "tool_need_gate": {"threshold": 0.1},
            "source_fingerprints": fingerprints,
        }
    }


def _row(prediction: str, false_edit: bool, cost: float, utility: float) -> dict:
    return {
        "sample_id": "sample-b3",
        "task_id": "task-1",
        "split": "val",
        "budget": 3.0,
        "target": "REJECT",
        "evaluation_seed": 7,
        "prediction": prediction,
        "terminal_correct": not false_edit,
        "false_edit": false_edit,
        "missed_edit": False,
        "spent_cost": cost,
        "tool_calls": 2,
        "tool_successes": 2,
        "mean_tool_belief_l1_delta": 0.2,
        "quality_cost_utility": utility,
        "joint_utility": 1.0 if not false_edit else -1.0,
        "selected_evidence_ids": ["e0"] if false_edit else ["e0", "e1"],
    }


def test_safety_gain_with_higher_cost_does_not_promote() -> None:
    report = compare(
        _summary("ungated"),
        _summary("gated"),
        [_row("COMMIT:ADD", True, 0.18, -0.18)],
        [_row("REJECT", False, 1.68, -0.33)],
        repetitions=100,
        seed=1,
    )

    assert report["checks"]["false_edit_noninferior"] is True
    assert report["checks"]["utility_improves"] is False
    assert report["checks"]["cost_noninferior"] is False
    assert report["promotion_passed"] is False
    assert len(report["changed_rollouts"]) == 1
