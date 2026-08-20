import numpy as np
import pytest

from activemap.safe_commit import (
    apply_commit_threshold,
    apply_keep_preserving_guard,
    select_safety_candidate,
)


def test_apply_commit_threshold_suppresses_only_low_confidence_updates():
    probabilities = np.asarray(
        [
            [0.2, 0.6, 0.1, 0.1],
            [0.3, 0.35, 0.2, 0.15],
            [0.7, 0.1, 0.1, 0.1],
        ]
    )
    assert apply_commit_threshold(probabilities, keep_index=0, threshold=0.5).tolist() == [1, 0, 0]


def test_select_safety_candidate_enforces_cap_before_quality():
    candidates = [
        {
            "C": 1.0,
            "commit_threshold": 0.4,
            "metrics": {
                "macro_f1": 0.8,
                "false_edit_rate": 0.2,
                "missed_edit_rate": 0.1,
            },
        },
        {
            "C": 0.1,
            "commit_threshold": 0.6,
            "metrics": {
                "macro_f1": 0.7,
                "false_edit_rate": 0.1,
                "missed_edit_rate": 0.2,
            },
        },
    ]
    selected = select_safety_candidate(candidates, false_edit_cap=0.1)
    assert selected["C"] == 0.1
    with pytest.raises(ValueError, match="no candidate"):
        select_safety_candidate(candidates, false_edit_cap=0.05)


def test_keep_preserving_guard_never_turns_baseline_keep_into_edit():
    baseline = np.asarray([0, 1, 0, 2])
    candidate = np.asarray([3, 2, 1, 0])
    guarded = apply_keep_preserving_guard(baseline, candidate, keep_index=0)
    assert guarded.tolist() == [0, 2, 0, 0]
