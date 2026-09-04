from scripts.assess_sn7_step0_frontier_promotion import assess


def _payload():
    rows = []
    for seed in (1, 2, 3):
        rows.extend(
            [
                {"seed": seed, "variant": "notool", "mean_tool_calls": 0.0},
                {
                    "seed": seed,
                    "variant": "benefit",
                    "mean_tool_calls": 0.01,
                    "mean_tool_belief_l1_delta": 0.02,
                },
                {"seed": seed, "variant": "forced", "mean_tool_calls": 0.3},
            ]
        )
    return {
        "schema_version": "sn7-selective-tool-frontier-three-seed-v1",
        "seeds": [1, 2, 3],
        "per_seed": rows,
        "comparisons": {
            "benefit_vs_notool": {
                "hierarchical_seed_aoi_bootstrap": {
                    "terminal_correct": {"ci95_low": 0.01},
                    "false_edit": {"ci95_high": -0.01},
                }
            },
            "benefit_vs_forced": {
                "hierarchical_seed_aoi_bootstrap": {
                    "terminal_correct": {"ci95_low": 0.0, "ci95_high": 0.0},
                    "quality_cost_utility": {"ci95_low": 0.01},
                    "tool_calls": {"ci95_high": -0.1},
                }
            },
        },
        "protocol": {
            "gate_calibration_split": "train",
            "validation_used_for_gate_calibration": False,
            "matched_samples_and_budgets": True,
            "split": "val",
            "test_assets_read": False,
        },
    }


def test_promotes_complete_three_seed_frontier():
    assert assess(_payload())["promote"] is True


def test_rejects_frontier_without_forced_cost_reduction():
    payload = _payload()
    payload["comparisons"]["benefit_vs_forced"][
        "hierarchical_seed_aoi_bootstrap"
    ]["tool_calls"]["ci95_high"] = 0.01
    result = assess(payload)
    assert result["promote"] is False
    assert result["checks"]["cost_reduction_vs_forced"] is False


def test_promotes_consistent_frozen_test_provenance():
    payload = _payload()
    payload["protocol"]["split"] = "test"
    payload["protocol"]["test_assets_read"] = True
    result = assess(payload, expected_split="test")
    assert result["promote"] is True
    assert result["split"] == "test"
    assert result["test_assets_read"] is True


def test_rejects_mixed_split_provenance():
    payload = _payload()
    payload["protocol"]["split"] = "test"
    result = assess(payload, expected_split="test")
    assert result["promote"] is False
    assert result["checks"]["split_provenance_matches"] is False
