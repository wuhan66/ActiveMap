import pytest

from scripts.cache_policy_relative_vlm_branches import policy_relative_metrics, select_shard


def _row(static: bool, direct: float, post: float) -> dict:
    return {
        "static_use_tool": static,
        "policy_relative_use_tool": post > direct,
        "direct_utility": direct,
        "post_tool_utility": post,
    }


def test_policy_relative_metrics_reports_churn_and_regret():
    rows = [
        _row(True, 0.0, 1.0),
        _row(True, 1.0, 0.0),
        _row(False, 0.0, 0.5),
        _row(False, 1.0, 0.0),
    ]
    result = policy_relative_metrics(rows)

    assert result["static_positive_count"] == 2
    assert result["policy_relative_positive_count"] == 2
    assert result["label_agreement"] == 0.5
    assert result["positive_jaccard"] == pytest.approx(1.0 / 3.0)
    assert result["static_label_precision"] == 0.5
    assert result["static_label_recall"] == 0.5
    assert result["mean_direct_utility"] == 0.5
    assert result["mean_static_label_utility"] == 0.5
    assert result["mean_policy_relative_oracle_utility"] == 0.875
    assert result["static_label_acquisition_regret"] == 0.375


def test_policy_relative_metrics_rejects_empty_cache():
    with pytest.raises(ValueError, match="empty"):
        policy_relative_metrics([])


def test_select_shard_is_deterministic_and_complete():
    items = list(range(7))
    shards = [select_shard(items, 2, index) for index in range(2)]

    assert shards == [[0, 2, 4, 6], [1, 3, 5]]
    assert sorted(shards[0] + shards[1]) == items


def test_select_shard_rejects_invalid_index():
    with pytest.raises(ValueError, match="shard_index"):
        select_shard([1], 2, 2)
