from __future__ import annotations

import numpy as np

from scripts.diagnose_selector_stop_frontier import (
    MarginObservations,
    summarize_frontier,
)


def test_stop_frontier_surfaces_feasible_noncollapsed_margin() -> None:
    observations = MarginObservations(
        margins=np.asarray([-1.0, 0.1, 0.5, 1.0]),
        chosen_candidate_utilities=np.asarray([0.0, 0.2, -0.1, 0.4]),
        stop_utilities=np.asarray([0.0, 0.0, 0.0, 0.0]),
        oracle_utilities=np.asarray([0.0, 0.2, 0.3, 0.4]),
        target_acquire=np.asarray([False, True, True, True]),
        exact_candidate=np.asarray([True, True, False, True]),
    )

    result = summarize_frontier(
        observations,
        current_stop_margin=10.0,
        points=5,
        max_false_call_rate=0.25,
        max_harmful_call_fraction=0.5,
        min_acquire_recall=0.5,
    )

    assert result["current_checkpoint"]["acquire_rate"] == 0.0
    assert result["zero_margin"]["acquire_rate"] == 0.75
    assert result["feasible_point_count"] > 0
    assert result["selected_feasible"] is not None
    assert result["selected_feasible"]["acquire_recall"] >= 0.5
