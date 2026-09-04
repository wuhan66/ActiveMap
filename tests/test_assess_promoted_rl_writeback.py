import pytest

from scripts.assess_promoted_rl_writeback import assess


def _aggregate():
    return {
        "schema_version": "agent-writeback-seed-matched-aggregate-v1",
        "seed_count": 3,
        "seed_matched_references": True,
        "split": "val",
        "test_assets_read": False,
        "candidate_minus_seed_matched_sft": {
            "episode_utility_v2_safety_auc": {
                "observed_delta": 0.2,
                "ci95_low": 0.1,
                "ci95_high": 0.3,
            },
            "raster_iou_gain_auc": {
                "observed_delta": 0.1,
                "ci95_low": 0.0,
                "ci95_high": 0.2,
            },
            "false_edit_auc": {
                "observed_delta": -0.1,
                "ci95_low": -0.2,
                "ci95_high": 0.0,
            },
            "vector_delta_topology_valid_auc": {
                "observed_delta": 0.01,
                "ci95_low": -0.1,
                "ci95_high": 0.1,
            },
        },
    }


def test_promotes_only_when_all_executable_gates_pass():
    result = assess(_aggregate())

    assert result["decision"] == "promote_rl"
    assert all(result["checks"].values())


def test_retains_sft_when_false_edit_is_inferior():
    aggregate = _aggregate()
    aggregate["candidate_minus_seed_matched_sft"]["false_edit_auc"][
        "ci95_high"
    ] = 0.01

    result = assess(aggregate)

    assert result["decision"] == "retain_sft"
    assert result["checks"]["false_edit_ci_noninferior"] is False


def test_rejects_non_three_seed_or_unmatched_evidence():
    aggregate = _aggregate()
    aggregate["seed_count"] = 2

    with pytest.raises(ValueError, match="three-seed matched"):
        assess(aggregate)
