import pytest

from activemap.models import (
    CandidateHypothesis,
    EditOperation,
    EditRecord,
    EpisodeRecord,
    EvidenceItem,
    GeoJSONGeometry,
)
from scripts.audit_preacquisition_cost import analyze


def episode(identifier):
    geometry = GeoJSONGeometry(type="Polygon", coordinates=[[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]])
    evidence = [
        EvidenceItem(evidence_id=f"e-{index}", timestamp="2018_01", region=(0, 0, 2, 2), scale=1, image_path="image.tif", clear_fraction=1.0, cost=1.0)
        for index in range(3)
    ]
    return EpisodeRecord(
        episode_id=identifier, aoi_id="aoi", anchor_timestamp="2018_01", split="val",
        source_dataset="fixture", map_before="before", target_map="after", prior_geometry=geometry,
        target_geometry=geometry, hypothesis=CandidateHypothesis(op=EditOperation.KEEP, object_id="obj", source="fixture"),
        evidence_catalog=evidence, gt_edit=EditRecord(op=EditOperation.KEEP, object_id="obj"),
        is_synthetic=False, derivation_version="fixture",
    )


def row(task_id, selected, cost):
    return {
        "task_id": task_id, "budget": 3.0, "split": "val", "test_assets_read": False,
        "selected_evidence_ids": selected, "spent_cost": cost,
    }


def test_cost_audit_exposes_shared_perception_without_erasing_incremental_savings(monkeypatch):
    import activemap.agent.identifiers as identifiers

    monkeypatch.setattr(identifiers, "public_task_id", lambda value: value)
    episodes = {"one": episode("one")}
    source = [row("one", ["e-0"], 0.5)]
    result = analyze(
        episodes,
        {"s1": {"notool": source, "benefit": source}, "s2": {"notool": source, "benefit": source}},
        updater_latency_ms=4.0,
    )

    cell = result["policies"]["benefit"]
    assert cell["shared_preacquisition_updater_calls"]["mean"] == pytest.approx(3.0)
    assert cell["shared_preacquisition_perception_ms"]["mean"] == pytest.approx(12.0)
    assert cell["incremental_selected_evidence_cost"]["mean"] == pytest.approx(0.5)
