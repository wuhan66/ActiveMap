import pytest

from scripts.launch_sn7_two_stage_writeback_matrix import jobs


def test_jobs_assigns_matched_policies_to_distinct_gpus() -> None:
    assert jobs([0, 2, 4, 5]) == [
        {"policy": "always_stop", "gpu": 0},
        {"policy": "edit_utility_s1", "gpu": 2},
        {"policy": "edit_utility_s2", "gpu": 4},
        {"policy": "edit_utility_s3", "gpu": 5},
    ]


def test_jobs_accepts_four_custom_policy_names() -> None:
    assert jobs([1, 2, 3, 4], ("stop", "seed1", "seed2", "seed3")) == [
        {"policy": "stop", "gpu": 1},
        {"policy": "seed1", "gpu": 2},
        {"policy": "seed2", "gpu": 3},
        {"policy": "seed3", "gpu": 4},
    ]


@pytest.mark.parametrize("gpus", [[0, 2], [0, 0, 4, 5]])
def test_jobs_rejects_incomplete_or_duplicate_gpus(gpus) -> None:
    with pytest.raises(ValueError, match="four distinct GPUs"):
        jobs(gpus)
