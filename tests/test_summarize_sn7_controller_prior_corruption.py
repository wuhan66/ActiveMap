import pytest

from scripts.summarize_sn7_controller_prior_corruption import (
    _controller_means,
    _writeback_variants,
)


def test_controller_means_averages_each_variant() -> None:
    keys = (
        "terminal_accuracy",
        "false_edit_rate",
        "missed_edit_rate",
        "mean_quality_gain",
        "mean_quality_cost_utility",
        "mean_tool_calls",
        "tool_call_episode_rate",
        "mean_tool_belief_l1_delta",
    )
    rows = []
    for seed, value in ((1, 0.2), (2, 0.4)):
        rows.append({"seed": seed, "variant": "benefit", **dict.fromkeys(keys, value)})

    result = _controller_means({"per_seed": rows})

    assert result["benefit"]["terminal_accuracy"] == pytest.approx(0.3)
    assert result["benefit"]["mean_tool_calls"] == pytest.approx(0.3)


def test_writeback_variants_preserves_paired_roles() -> None:
    candidate = {"raster_iou_auc": 0.8}
    result = _writeback_variants(
        {"baseline": {"raster_iou_auc": 0.7}, "candidate": candidate},
        {"baseline": {"raster_iou_auc": 0.6}, "candidate": candidate},
    )

    assert result["notool"]["raster_iou_auc"] == 0.7
    assert result["forced"]["raster_iou_auc"] == 0.6
    assert result["benefit"]["raster_iou_auc"] == 0.8
