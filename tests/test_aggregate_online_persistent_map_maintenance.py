import pytest

from scripts.aggregate_online_persistent_map_maintenance import aggregate


def trace(offset=0.0):
    rows = {}
    for branch, value in (("independent_reset", 0.8), ("carry_always_commit", 0.7), ("carry_safe_commit", 0.75)):
        rows[("chain-0", 0, branch)] = {
            "chain_id": "chain-0", "step": 0, "branch": branch, "aoi_id": "aoi-0",
            "final_raster_iou": value + offset, "executed_raster_iou_gain": 0.0,
            "false_edit": False, "missed_edit": False, "wrong_edit": False,
            "commit_accepted": True, "safe_commit_rejected": False,
            "recovered_from_prior_error": False, "input_prior_matches_canonical": branch == "independent_reset",
            "test_assets_read": False,
        }
    return rows


def test_online_aggregate_reports_seed_and_aoi_bootstrap():
    result = aggregate({"a": trace(), "b": trace(0.01)}, repetitions=30, seed=7)
    assert result["seed_count"] == 2
    assert result["paired_comparisons"]["safe_minus_always"]["mean_step_raster_iou"]["observed_delta"] == pytest.approx(0.05)
