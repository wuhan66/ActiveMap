import argparse

import pytest

from scripts.launch_sn7_writeback_safety_sweep import _candidate, jobs


def test_safety_sweep_assigns_four_candidates_to_distinct_gpus():
    candidates = [
        _candidate("raw:0:0"),
        _candidate("m05:0.05:0"),
        _candidate("m10:0.10:0"),
        _candidate("m15:0.15:0"),
    ]

    plan = jobs(candidates, [0, 2, 4, 5])

    assert [row["gpu"] for row in plan] == [0, 2, 4, 5]
    assert plan[-1]["delta_margin"] == 0.15
    assert plan[-1]["min_delta_component_pixels"] == 0


def test_safety_sweep_accepts_minimum_component_pixels():
    candidate = _candidate("m20_p32:0.20:0:32")

    assert candidate["delta_margin"] == 0.20
    assert candidate["confidence_floor"] == 0.0
    assert candidate["min_delta_component_pixels"] == 32


@pytest.mark.parametrize(
    "value", ["bad", "Bad:0:0", "bad:0.6:0", "bad:0.1:0:-1"]
)
def test_candidate_rejects_invalid_specification(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _candidate(value)
