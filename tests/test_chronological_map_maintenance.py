import pytest

pytest.importorskip("shapely")

from activemap.agent.identifiers import public_task_id
from activemap.models import (
    CandidateHypothesis,
    EditOperation,
    EditRecord,
    EpisodeRecord,
    GeoJSONGeometry,
)
from scripts.evaluate_chronological_map_maintenance import (
    build_contiguous_chains,
    evaluate,
    summarize,
)


def polygon(x0, x1):
    return GeoJSONGeometry(
        type="Polygon",
        coordinates=[[[x0, 0], [x1, 0], [x1, 1], [x0, 1], [x0, 0]]],
    )


def episode(identifier, timestamp, prior, target):
    return EpisodeRecord(
        episode_id=identifier,
        aoi_id="aoi",
        anchor_timestamp=timestamp,
        split="val",
        source_dataset="fixture",
        map_before="before.geojson",
        target_map="after.geojson",
        prior_geometry=prior,
        target_geometry=target,
        hypothesis=CandidateHypothesis(
            op=EditOperation.RESHAPE,
            object_id="object-1",
            geometry=target,
            source="fixture",
        ),
        evidence_catalog=[],
        gt_edit=EditRecord(
            op=EditOperation.RESHAPE,
            object_id="object-1",
            geometry=target,
        ),
        is_synthetic=False,
        derivation_version="fixture",
    )


def row(current_episode, add, remove=None, confidence=0.9):
    return {
        "task_id": public_task_id(current_episode.episode_id),
        "budget": 3.0,
        "predicted_add_geometry": add.model_dump() if add else None,
        "predicted_remove_geometry": remove.model_dump() if remove else None,
        "writeback_changed": bool(add or remove),
        "fused_confidence": confidence,
        "vector_replay_iou": 1.0,
        "vector_delta_topology_valid": True,
        "split": "val",
        "test_assets_read": False,
    }


def test_contiguous_chain_carries_state_and_risk_gate_blocks_low_confidence():
    first = episode("one", "2018_02", polygon(0, 1), polygon(0, 2))
    second = episode("two", "2018_03", polygon(0, 2), polygon(0, 3))
    chains = build_contiguous_chains(
        [
            (second, row(second, polygon(2, 3), confidence=0.1)),
            (first, row(first, polygon(1, 2))),
        ],
        continuity_tolerance=1e-6,
        minimum_length=2,
    )
    records = evaluate(chains, confidence_threshold=0.7, replay_iou_threshold=0.99)

    assert len(records) == 2
    assert records[0]["risk_gate_intervened"] is True
    assert records[1]["risk_gate_intervened"] is False
    assert records[1]["carry_iou"] == 1.0
    assert records[1]["risk_gated_iou"] < 1.0
    assert summarize(records)["risk_gate_rejections"] == 1


def test_discontinuous_ground_truth_breaks_chain():
    first = episode("one", "2018_02", polygon(0, 1), polygon(0, 2))
    second = episode("two", "2018_03", polygon(10, 11), polygon(10, 12))
    chains = build_contiguous_chains(
        [(first, row(first, polygon(1, 2))), (second, row(second, polygon(11, 12)))],
        continuity_tolerance=1e-6,
        minimum_length=2,
    )
    assert chains == []
