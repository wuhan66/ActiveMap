from scripts.aggregate_online_full_controller import REQUIRED_POLICIES, aggregate


def _row(*, policy, chain_id, aoi_id, step, quality, tool_calls, spent_cost):
    return {
        "policy": policy,
        "chain_id": chain_id,
        "aoi_id": aoi_id,
        "step": step,
        "split": "val",
        "test_assets_read": False,
        "final_raster_iou": quality,
        "false_edit": False,
        "missed_edit": False,
        "wrong_edit": False,
        "acquisitions": int(tool_calls > 0),
        "tool_calls": tool_calls,
        "spent_cost": spent_cost,
        "tool_cost": 0.18 * tool_calls,
        "commit_accepted": True,
        "safe_commit_rejected": False,
        "prior_input_sha256": f"{policy}-{chain_id}-{step}",
        "canonical_prior_sha256": f"canonical-{chain_id}",
    }


def _trace(seed_offset=0.0):
    rows = {}
    policy_quality = {
        "direct_current_hypothesis": 0.50,
        "direct_current_hypothesis_safe": 0.55,
        "active_selective_safe": 0.70,
        "active_forced_safe": 0.70,
    }
    policy_calls = {
        "direct_current_hypothesis": 0,
        "direct_current_hypothesis_safe": 0,
        "active_selective_safe": 1,
        "active_forced_safe": 2,
    }
    for chain_id, aoi_id in (("chain-0", "aoi-0"), ("chain-1", "aoi-1")):
        for policy in REQUIRED_POLICIES:
            for step in range(2):
                row = _row(
                    policy=policy,
                    chain_id=chain_id,
                    aoi_id=aoi_id,
                    step=step,
                    quality=policy_quality[policy] + seed_offset + 0.01 * step,
                    tool_calls=policy_calls[policy],
                    spent_cost=0.10 * policy_calls[policy],
                )
                rows[(chain_id, step, policy)] = row
    return rows


def test_aggregate_reports_horizon_prefixes_and_paired_cost_contrast():
    result = aggregate(
        {"seed-a": _trace(), "seed-b": _trace(0.01)},
        policies=REQUIRED_POLICIES,
        repetitions=40,
        seed=7,
    )

    assert set(result["horizons"]) == {"1", "2"}
    horizon_two = result["horizons"]["2"]
    assert horizon_two["policy_metrics"]["active_selective_safe"][
        "cumulative_tool_calls"
    ]["mean"] == 2.0
    assert horizon_two["paired_comparisons"]["active_minus_forced_safe"][
        "cumulative_spent_cost"
    ]["observed_delta"] < 0.0
    assert horizon_two["paired_comparisons"]["active_minus_direct_safe"][
        "final_map_quality"
    ]["ci95_low"] > 0.0
