import pytest

from scripts.launch_sn7_selector_utility_sweep import jobs


def test_jobs_assigns_one_variant_per_gpu() -> None:
    assert jobs(7, [0, 2, 4, 5]) == [
        {"variant": "generic_rank", "seed": 7, "gpu": 0},
        {"variant": "generic_utility", "seed": 7, "gpu": 2},
        {"variant": "edit_rank", "seed": 7, "gpu": 4},
        {"variant": "edit_utility", "seed": 7, "gpu": 5},
    ]


def test_jobs_requires_four_distinct_gpus() -> None:
    with pytest.raises(ValueError, match="four distinct GPUs"):
        jobs(7, [0, 1])
