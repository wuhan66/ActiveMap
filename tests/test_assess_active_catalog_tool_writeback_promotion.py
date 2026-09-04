from scripts.assess_active_catalog_tool_writeback_promotion import assess


def _comparison(
    map_low: float = 0.01,
    *,
    cost_aware_low: float = 0.01,
    spent_cost_high: float = -0.01,
):
    return {
        "group_key": "aoi_id",
        "paired_delta": {
            "raster_iou_gain_auc": {"ci95_low": map_low},
            "vector_replay_iou_auc": {"ci95_low": -0.005},
            "vector_delta_topology_valid_auc": {"ci95_low": 0.0},
            "episode_utility_v2_balanced_auc": {"ci95_low": map_low},
            "episode_utility_v2_cost_aware_auc": {"ci95_low": cost_aware_low},
            "false_edit_auc": {"ci95_high": 0.01},
            "spent_cost_auc": {"ci95_high": spent_cost_high},
        },
    }


def test_promotes_with_gain_over_no_tool_and_cheaper_noninferior_forced():
    branch = {
        "schema_version": "active-catalog-tool-branch-promotion-v1",
        "promote": True,
    }
    forced = _comparison(map_low=-0.0005)
    assert assess(branch, _comparison(), forced)["promote"] is True
    assert assess(branch, _comparison(-0.001), _comparison())["promote"] is False


def test_rejects_forced_comparison_without_cost_reduction():
    branch = {
        "schema_version": "active-catalog-tool-branch-promotion-v1",
        "promote": True,
    }
    forced = _comparison(spent_cost_high=0.001)
    result = assess(branch, _comparison(), forced)
    assert result["promote"] is False
    assert result["checks"]["cost_reduction_vs_forced"] is False


def test_refuses_proxy_branch_that_failed_safety_gate():
    branch = {
        "schema_version": "active-catalog-tool-branch-promotion-v1",
        "promote": False,
    }
    try:
        assess(branch, _comparison(), _comparison())
    except ValueError as error:
        assert "did not pass" in str(error)
    else:
        raise AssertionError("failed closed-loop gate must not be promoted")


def test_frozen_recorder_preserves_negative_upstream_result():
    branch = {
        "schema_version": "active-catalog-tool-branch-promotion-v1",
        "promote": False,
    }
    result = assess(
        branch,
        _comparison(),
        _comparison(),
        allow_failed_upstream=True,
    )
    assert result["promote"] is False
    assert result["checks"]["closed_loop_tool_gate_passed"] is False


def test_three_seed_writeback_gate_rejects_single_seed_payloads():
    branch = {
        "schema_version": "active-catalog-tool-branch-promotion-v1",
        "promote": True,
    }
    result = assess(branch, _comparison(), _comparison(), min_seeds=3)
    assert result["promote"] is False
    assert result["checks"]["required_seed_count"] is False
