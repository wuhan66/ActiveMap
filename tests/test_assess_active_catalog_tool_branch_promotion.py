from scripts.assess_active_catalog_tool_branch_promotion import assess


def _interval(delta, low=None, high=None):
    return {"observed_delta": delta, "ci95_low": delta if low is None else low,
            "ci95_high": delta if high is None else high}


def _payload():
    metrics = {
        "selective": {"mean_tool_calls": 0.8, "mean_tool_belief_l1_delta": 0.1},
        "forced": {"mean_tool_calls": 2.0, "mean_tool_belief_l1_delta": 0.2},
        "no_tool": {"mean_tool_calls": 0.0, "mean_tool_belief_l1_delta": 0.0},
    }
    comparisons = {}
    for reference in ("forced", "no_tool"):
        comparisons[f"selective_minus_{reference}"] = {"intervals": {
            "mean_quality_cost_utility": _interval(0.1, 0.03, 0.2),
            "mean_cost": _interval(-0.1, -0.2, -0.02) if reference == "forced" else _interval(0.05),
            "false_edit_rate": _interval(0.0, -0.01, 0.01),
            "terminal_accuracy": _interval(0.0, -0.005, 0.01),
        }}
    return {
        "schema_version": "active-catalog-closed-loop-paired-comparison-v1",
        "comparison_orientation": {"candidate": "selective", "references": ["forced", "no_tool"]},
        "metrics": metrics,
        "paired_aoi_comparisons": comparisons,
    }


def test_tool_branch_promotion_requires_sparse_safe_gain():
    result = assess(_payload())
    assert result["promote"] is True


def test_tool_branch_promotion_rejects_non_significant_utility():
    payload = _payload()
    payload["paired_aoi_comparisons"]["selective_minus_no_tool"]["intervals"]["mean_quality_cost_utility"]["ci95_low"] = -0.01
    result = assess(payload)
    assert result["promote"] is False
    assert result["checks"]["utility_gain_vs_no_tool"] is False


def test_three_seed_gate_requires_each_seed_to_use_tools():
    payload = _payload()
    payload["seed_count"] = 3
    payload["per_seed_metrics"] = {
        str(seed): payload["metrics"] for seed in (1, 2, 3)
    }
    assert assess(payload, min_seeds=3)["promote"] is True
    payload["per_seed_metrics"]["2"] = {
        **payload["metrics"],
        "selective": {"mean_tool_calls": 0.0, "mean_tool_belief_l1_delta": 0.0},
    }
    result = assess(payload, min_seeds=3)
    assert result["checks"]["every_seed_has_tool_calls"] is False


def test_frozen_prior_branch_adds_separate_causal_diagnostic():
    payload = _payload()
    payload["comparison_orientation"]["references"].append(
        "selective_frozen_prior"
    )
    payload["paired_aoi_comparisons"][
        "selective_minus_selective_frozen_prior"
    ] = {
        "intervals": {
            "mean_quality_cost_utility": _interval(0.04, 0.01, 0.07),
        }
    }
    result = assess(payload)
    assert result["promote"] is True
    assert result["causal_diagnostics"][
        "recurrent_belief_strict_gain_supported"
    ] is True
