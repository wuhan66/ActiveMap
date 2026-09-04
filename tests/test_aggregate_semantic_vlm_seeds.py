from scripts.aggregate_semantic_vlm_seeds import aggregate


def _summary(seed, gain=0.1):
    return {
        "model_training_seed": seed,
        "sample_count": 10,
        "validation_jsonl": "/data/val.jsonl",
        "schema_valid_rate": 1.0,
        "executable_valid_rate": 1.0,
        "operation_metrics": {"macro_f1": 0.6},
        "baseline_operation_metrics": {"macro_f1": 0.5},
        "forced_operation_metrics": {"macro_f1": 0.55},
        "false_edit_rate": 0.1,
        "baseline_false_edit_rate": 0.1,
        "forced_false_edit_rate": 0.2,
        "missed_edit_rate": 0.2,
        "baseline_missed_edit_rate": 0.3,
        "forced_missed_edit_rate": 0.25,
        "mean_policy_utility": 0.4 + gain,
        "mean_baseline_utility": 0.4,
        "mean_forced_utility": 0.2,
        "mean_policy_gain": gain,
        "tool_metrics": {
            "call_rate": 0.2,
            "precision": 0.7,
            "recall": 0.6,
            "f1": 0.646,
            "false_call_rate": 0.05,
            "mean_cost": 0.15,
        },
        "test_assets_read": False,
        "adapter": f"seed{seed}",
    }


def test_aggregate_requires_every_seed_to_pass():
    result = aggregate([_summary(16), _summary(19), _summary(22, gain=-0.01)], [16, 19, 22])
    assert result["all_static_gates_passed"] is False
    assert result["per_seed"][2]["failed_gates"] == ["cost_adjusted_utility"]


def test_aggregate_passes_three_consistent_seeds():
    result = aggregate([_summary(16), _summary(19), _summary(22)], [16, 19, 22])
    assert result["all_static_gates_passed"] is True
    assert result["paper_claim_ready"] is False
