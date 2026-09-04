import pytest

from scripts.aggregate_chronological_map_maintenance import aggregate


def rows(offset=0.0):
    return [
        {
            "task_id": f"task-{index}",
            "budget": 3.0,
            "aoi_id": f"aoi-{index}",
            "independent_iou": 0.8,
            "carry_iou": 0.7 + offset,
            "risk_gated_iou": 0.75 + offset,
            "risk_gate_intervened": index == 0,
            "requested_intervention": True,
            "test_assets_read": False,
        }
        for index in range(2)
    ]


def test_three_seed_aggregate_reports_error_propagation_and_gate_gain():
    result = aggregate(
        {"1": rows(), "2": rows(0.01), "3": rows(-0.01)},
        repetitions=20,
        seed=7,
    )
    assert result["seed_count"] == 3
    assert result["aggregate"]["carry_minus_independent"]["mean"] == pytest.approx(-0.1)
    assert result["aggregate"]["risk_gated_minus_carry"]["mean"] == pytest.approx(0.05)
    assert result["aggregate"]["risk_gated_minus_independent"]["mean"] == pytest.approx(
        -0.05
    )
