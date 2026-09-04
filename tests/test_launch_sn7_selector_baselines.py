import pytest

from scripts.launch_sn7_selector_baselines import jobs


def test_jobs_pair_methods_per_seed():
    assert jobs([11, 12], [0, 1, 2, 3]) == [
        {"method": "generic", "seed": 11, "gpu": 0},
        {"method": "edit_conditioned", "seed": 11, "gpu": 1},
        {"method": "generic", "seed": 12, "gpu": 2},
        {"method": "edit_conditioned", "seed": 12, "gpu": 3},
    ]


def test_jobs_refuses_oversubscription():
    with pytest.raises(ValueError, match="distinct GPU"):
        jobs([11, 12], [0, 1])
