from __future__ import annotations

import pytest

from scripts.audit_sn7_counterfactual_runtime_alignment import summarize


def _row(expected: float, actual: float, *, exact: bool = False) -> dict:
    return {
        "source_episode": f"episode-{expected}-{actual}",
        "aoi_id": "aoi-a",
        "selected_expected_raster_gain": expected,
        "raw_raster_iou_gain": actual,
        "exact_oracle_candidate": exact,
    }


def test_alignment_summary_reports_sign_failure_and_exact_oracle() -> None:
    result = summarize(
        [_row(0.3, 0.2, exact=True), _row(0.2, -0.4), _row(-0.1, -0.2)],
        epsilon=1e-6,
    )
    assert result["count"] == 3
    assert result["sign_agreement_rate"] == pytest.approx(2 / 3)
    assert result["expected_positive_runtime_positive_rate"] == pytest.approx(0.5)
    assert result["expected_positive_runtime_harmful_rate"] == pytest.approx(0.5)
    assert result["exact_oracle_runtime_positive_rate"] == 1.0


def test_alignment_summary_handles_constant_series() -> None:
    result = summarize([_row(0.1, 0.0), _row(0.1, 0.0)], epsilon=1e-6)
    assert result["pearson_correlation"] is None
