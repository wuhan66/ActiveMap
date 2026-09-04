from __future__ import annotations

import pytest

from scripts.aggregate_sn7_v6_evidence_value_audits import METRICS, aggregate


def _record(seed: int, value: float) -> dict:
    return {
        "seed": seed,
        "summary": f"seed{seed}.json",
        "checkpoint": f"seed{seed}.pt",
        "checkpoint_sha256": "a" * 64,
        "state_count": 10.0,
        "overall": {metric: value for metric in METRICS},
    }


def test_aggregate_reports_per_seed_mean_and_sample_standard_deviation():
    result = aggregate(
        [_record(1, 0.1), _record(2, 0.2), _record(3, 0.3)]
    )

    assert result["seeds"] == [1, 2, 3]
    assert result["metrics"]["acquire_rate"]["mean"] == pytest.approx(0.2)
    assert result["metrics"]["acquire_rate"]["sample_std"] == pytest.approx(0.1)
    assert result["test_assets_read"] is False
