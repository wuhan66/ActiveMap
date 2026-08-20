from __future__ import annotations

import pytest

from activemap.evaluation.episode_utility import (
    SCHEMA_VERSION,
    UTILITY_PROFILES,
    episode_utility,
    score_episode_profiles,
    utility_protocol,
)


def test_balanced_utility_uses_map_gain_and_counts_cost_once() -> None:
    score = episode_utility(
        final_map_quality=0.8,
        prior_map_quality=0.5,
        spent_cost=1.0,
        budget=2.0,
        false_edit=False,
        missed_edit=False,
        wrong_edit=False,
        profile=UTILITY_PROFILES["balanced"],
    )

    assert score["quality_gain"] == pytest.approx(0.3)
    assert score["normalized_cost"] == pytest.approx(0.5)
    assert score["cost_penalty"] == pytest.approx(0.05)
    assert score["value"] == pytest.approx(0.25)


def test_false_edit_is_more_costly_under_safety_profile() -> None:
    scores = score_episode_profiles(
        final_map_quality=0.4,
        prior_map_quality=0.9,
        spent_cost=0.0,
        budget=1.0,
        false_edit=True,
        missed_edit=False,
        wrong_edit=False,
    )

    assert scores["balanced"]["value"] == pytest.approx(-1.0)
    assert scores["safety"]["value"] == pytest.approx(-1.5)
    assert scores["cost_aware"]["value"] == pytest.approx(-1.0)


def test_utility_rejects_overlapping_error_categories() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        episode_utility(
            final_map_quality=0.0,
            prior_map_quality=0.0,
            spent_cost=0.0,
            budget=1.0,
            false_edit=True,
            missed_edit=True,
            wrong_edit=False,
            profile=UTILITY_PROFILES["balanced"],
        )


def test_protocol_distinguishes_proxy_from_paper_primary() -> None:
    proxy = utility_protocol(
        quality_source="terminal_operation_exactness_proxy", paper_primary=False
    )
    writeback = utility_protocol(
        quality_source="executable_writeback_raster_iou", paper_primary=True
    )

    assert proxy["schema_version"] == SCHEMA_VERSION
    assert proxy["paper_primary"] is False
    assert writeback["paper_primary"] is True
    assert set(writeback["profiles"]) == {"balanced", "safety", "cost_aware"}
