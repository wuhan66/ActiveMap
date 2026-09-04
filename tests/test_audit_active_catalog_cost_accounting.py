import pytest

from activemap.models import (
    CandidateHypothesis,
    EditOperation,
    EditRecord,
    EpisodeRecord,
    EvidenceItem,
    GeoJSONGeometry,
)
from scripts.audit_active_catalog_cost_accounting import analyze


def episode(identifier: str) -> EpisodeRecord:
    geometry = GeoJSONGeometry(type="Polygon", coordinates=[[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]])
    evidence = [
        EvidenceItem(evidence_id=f"ev-{index}", timestamp="2018_01", region=(0, 0, 2, 2), scale=1, image_path="image.tif", clear_fraction=1.0, cost=1.0)
        for index in range(3)
    ]
    return EpisodeRecord(
        episode_id=identifier, aoi_id="aoi", anchor_timestamp="2018_01", split="val",
        source_dataset="fixture", map_before="before", target_map="after", prior_geometry=geometry,
        target_geometry=geometry, hypothesis=CandidateHypothesis(op=EditOperation.KEEP, object_id="obj", source="fixture"),
        evidence_catalog=evidence, gt_edit=EditRecord(op=EditOperation.KEEP, object_id="obj"),
        is_synthetic=False, derivation_version="fixture",
    )


def rollout(acquisitions: int, spent: float) -> dict:
    return {
        "source_episode": "one", "budget": 3.0, "split": "val", "test_assets_read": False,
        "acquisitions": acquisitions, "tool_calls": acquisitions, "spent_cost": spent, "tool_cost": 0.0,
    }


def test_audit_keeps_shared_updater_work_separate_from_actual_rollout_acquisitions():
    result = analyze(
        {"one": episode("one")},
        {
            "s1": {"notool": [rollout(0, 0.0)], "benefit": [rollout(1, 0.5)]},
            "s2": {"notool": [rollout(0, 0.0)], "benefit": [rollout(1, 0.5)]},
        },
        updater_latency_ms=4.0,
    )
    assert result["policies"]["notool"]["shared_preacquisition_updater_calls"]["mean"] == pytest.approx(3.0)
    assert result["policies"]["benefit"]["incremental_acquisitions"]["mean"] == pytest.approx(1.0)
    assert result["policies"]["notool"]["incremental_evidence_budget"]["mean"] == pytest.approx(0.0)
