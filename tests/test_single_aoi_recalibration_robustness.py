from scripts.evaluate_sn7_single_aoi_recalibration_robustness import (
    distribution,
    eligible_aoi_rows,
)


def test_distribution_and_aoi_filter():
    values = distribution([1.0, 2.0, 3.0])
    assert values["mean"] == 2.0
    assert values["min"] == 1.0
    rows = [
        {"aoi_id": "mixed", "target_edit": "KEEP"},
        {"aoi_id": "mixed", "target_edit": "ADD"},
        {"aoi_id": "stable", "target_edit": "KEEP"},
    ]
    assert set(eligible_aoi_rows(rows)) == {"mixed"}
