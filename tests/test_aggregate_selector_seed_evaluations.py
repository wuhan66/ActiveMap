import pytest

from scripts.aggregate_selector_seed_evaluations import aggregate_seeds


def _rows(offset):
    return [
        {
            "sample_id": "one",
            "aoi_id": "a",
            "called": True,
            "target_acquire": True,
            "false_call": False,
            "harmful_call": False,
            "exact_acquire": True,
            "utility": 0.2 + offset,
            "regret": 0.0,
        },
        {
            "sample_id": "two",
            "aoi_id": "b",
            "called": False,
            "target_acquire": False,
            "false_call": False,
            "harmful_call": False,
            "exact_acquire": False,
            "utility": 0.0,
            "regret": 0.1,
        },
    ]


def test_paired_seed_aggregation_is_reproducible() -> None:
    first = aggregate_seeds([_rows(0.0), _rows(0.1), _rows(-0.1)], draws=50, seed=4)
    second = aggregate_seeds([_rows(0.0), _rows(0.1), _rows(-0.1)], draws=50, seed=4)
    assert first == second
    assert first["mean"]["mean_utility"] == pytest.approx(0.1)


def test_seed_aggregation_rejects_mismatched_samples() -> None:
    mismatched = _rows(0.0)
    mismatched[0]["sample_id"] = "different"
    with pytest.raises(ValueError, match="identical ordered samples"):
        aggregate_seeds([_rows(0.0), mismatched], draws=10, seed=1)
