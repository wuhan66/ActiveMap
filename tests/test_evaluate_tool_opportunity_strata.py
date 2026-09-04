import pytest

from scripts.evaluate_tool_opportunity_strata import evaluate_tool_opportunity_strata


def _manifest():
    return {
        "scope": "diagnostic_only",
        "split": "validation",
        "test_assets_read": False,
        "episode_id_sha256": "frozen",
        "records": [
            {"sequence_id": "positive", "stratum": "positive"},
            {"sequence_id": "boundary", "stratum": "near_boundary"},
            {"sequence_id": "harmful", "stratum": "harmful"},
        ],
    }


def _prediction(trajectory_id, target_class, prediction_class, *, exact=True):
    target = f"{target_class}:target"
    prediction = target if exact else f"{prediction_class}:wrong"
    return {
        "trajectory_id": trajectory_id,
        "target": target,
        "target_class": target_class,
        "prediction": prediction,
        "prediction_class": prediction_class,
        "schema_valid": True,
        "executable": True,
        "predicted_utility": 0.8 if exact else -0.2,
        "oracle_utility": 0.8,
    }


def test_static_strata_report_tool_recall_and_false_calls():
    predictions = [
        _prediction("positive", "USE_TOOL", "USE_TOOL"),
        _prediction("positive", "COMMIT", "COMMIT"),
        _prediction("boundary", "COMMIT", "USE_TOOL", exact=False),
        _prediction("harmful", "REJECT", "REJECT"),
        _prediction("acquisition-controller", "ACQUIRE", "ACQUIRE"),
    ]

    result = evaluate_tool_opportunity_strata(_manifest(), predictions)

    assert result["strata"]["positive"]["grounded_tool_exact_recall"] == 1.0
    assert result["strata"]["near_boundary"]["false_call_rate"] == 1.0
    assert result["strata"]["harmful"]["episode_any_tool_call_rate"] == 0.0
    assert result["excluded_non_opportunity_state_count"] == 1
    assert result["allowed_for_checkpoint_selection"] is False


def test_static_strata_reject_missing_and_test_records():
    predictions = [_prediction("positive", "USE_TOOL", "USE_TOOL")]
    with pytest.raises(ValueError, match="missing predictions"):
        evaluate_tool_opportunity_strata(_manifest(), predictions)

    manifest = _manifest()
    manifest["split"] = "test"
    with pytest.raises(ValueError, match="test opportunity"):
        evaluate_tool_opportunity_strata(manifest, predictions)
