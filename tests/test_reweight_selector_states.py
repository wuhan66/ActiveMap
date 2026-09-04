import pytest

from scripts.reweight_selector_states import reweight_record


def test_reweight_record_preserves_content_and_updates_utility() -> None:
    source = {
        "sample_id": "sample",
        "evidence_costs": [1.0, 1.5],
        "oracle_utilities": [-0.1, 0.2],
        "metadata": {"cost_weight": 0.18, "aoi_id": "aoi"},
    }
    adjusted = reweight_record(source, 0.10)
    assert adjusted["oracle_utilities"] == pytest.approx([-0.02, 0.32])
    assert adjusted["metadata"]["source_cost_weight"] == 0.18
    assert adjusted["metadata"]["cost_weight"] == 0.10
    assert adjusted["metadata"]["aoi_id"] == "aoi"
    assert source["oracle_utilities"] == [-0.1, 0.2]
