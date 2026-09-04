from scripts.aggregate_hierarchical_semantic_vlm_seeds import aggregate


def _row(seed, delta, passed=True):
    direct_f1 = 0.5
    direct_utility = 0.4
    return {
        "model_training_seed": seed,
        "hierarchical": {
            "operation_metrics": {"macro_f1": direct_f1 + delta},
            "false_edit_rate": 0.08,
            "missed_edit_rate": 0.3,
        },
        "direct_vlm": {
            "operation_metrics": {"macro_f1": direct_f1},
            "false_edit_rate": 0.1,
            "missed_edit_rate": 0.35,
        },
        "mean_hierarchical_utility": direct_utility + delta,
        "mean_direct_utility": direct_utility,
        "hierarchical_minus_direct_utility": delta,
        "hierarchical_minus_direct_macro_f1": delta,
        "tool_metrics": {"call_rate": 0.1, "precision": 0.7, "recall": 0.4},
        "promotion_gate": {"passed": passed},
        "test_assets_read": False,
    }


def test_aggregate_requires_every_fixed_seed_to_pass():
    result = aggregate(
        [_row(20260716, 0.02), _row(20260719, 0.03, False), _row(20260722, 0.01)],
        [20260716, 20260719, 20260722],
    )
    assert result["metrics"]["macro_f1_delta"]["mean"] == 0.02
    assert result["metrics"]["utility_delta"]["mean"] == 0.02
    assert result["all_seed_gates_passed"] is False
    assert result["paper_claim_ready"] is False


def test_aggregate_rejects_missing_seed():
    try:
        aggregate([_row(1, 0.1)], [1, 2])
    except ValueError as error:
        assert "expected seeds" in str(error)
    else:
        raise AssertionError("missing fixed seeds must fail")
