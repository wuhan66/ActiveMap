from __future__ import annotations

import json
from pathlib import Path

from activemap.data.argotweak_native import (
    build_argotweak_native_episodes,
    operation_set_utility,
    proposal_city_geometry,
    proposal_operation_belief,
)
from activemap.data.argotweak_native_export import (
    argotweak_frame_operation_counts,
    assign_argotweak_proposal_objects,
)


def _proposal(proposal_id: str, operation: str, confidence: float, object_id=None):
    return {
        "proposal_id": proposal_id,
        "object_id": object_id,
        "operation": operation,
        "geometry": [[0.0, 0.0], [1.0, 1.0]],
        "confidence": {"joint": confidence},
    }


def test_frame_supervision_uses_official_change_codes() -> None:
    frame = {
        "annotation": {
            "lane_segment": [
                {"change_score": [0]},
                {"change_score": [1]},
                {"change_score": [3, 5]},
            ],
            "area": [
                {"category": 1, "change_score": [2]},
                {"category": 2, "change_score": [2]},
            ],
        }
    }
    assert argotweak_frame_operation_counts(frame) == {
        "KEEP": 1,
        "ADD": 1,
        "DELETE": 1,
        "RESHAPE": 1,
    }


def test_proposals_are_matched_one_to_one_and_fail_closed() -> None:
    frame = {
        "annotation": {
            "lane_segment": [
                {
                    "id": "lane-7",
                    "centerline": [[0.0, 0.0], [1.0, 0.0]],
                    "left_laneline": [[0.0, 1.0], [1.0, 1.0]],
                    "right_laneline": [[0.0, -1.0], [1.0, -1.0]],
                    "change_score": [3],
                }
            ],
            "area": [],
        }
    }
    proposals = [
        {
            **_proposal("near", "RESHAPE", 0.9),
            "feature_class": "lane_segment",
            "geometry": [[0.0, 0.0], [1.0, 0.0]],
        },
        {
            **_proposal("far", "DELETE", 0.9),
            "feature_class": "lane_segment",
            "geometry": [[20.0, 20.0], [21.0, 20.0]],
        },
    ]

    summary = assign_argotweak_proposal_objects(proposals, frame, max_distance=1.5)

    assert summary["matched"] == 1
    assert proposals[0]["object_id"] == "lane-7"
    assert proposals[0]["matched_target_operation"] == "RESHAPE"
    assert proposals[1]["object_id"] is None
    assert proposals[1]["object_match_status"] == "unmatched_fail_closed"


def test_belief_and_utility_reward_correct_sparse_evidence() -> None:
    proposals = [_proposal("a", "ADD", 0.8), _proposal("k", "KEEP", 0.2)]
    belief = proposal_operation_belief(proposals)
    assert belief == [0.2, 0.8, 0.0, 0.0]
    target = {"KEEP": 2, "ADD": 1, "DELETE": 0, "RESHAPE": 0}
    good = operation_set_utility(
        proposals,
        target,
        evidence_cost=1.0,
        false_edit_weight=0.5,
        missed_edit_weight=0.5,
        cost_weight=0.05,
    )
    bad = operation_set_utility(
        [_proposal("d", "DELETE", 0.8)],
        target,
        evidence_cost=1.0,
        false_edit_weight=0.5,
        missed_edit_weight=0.5,
        cost_weight=0.05,
    )
    assert good > bad


def test_proposal_geometry_is_transformed_to_city_frame() -> None:
    proposal = {
        "geometry": [[float(index), 0.0] for index in range(30)],
        "feature_class": "lane_segment",
    }
    geometry = proposal_city_geometry(
        proposal,
        {
            "rotation": [[0.0, -1.0], [1.0, 0.0]],
            "translation": [100.0, 200.0],
        },
    )
    ring = geometry.coordinates[0]
    assert ring[0] == [100.0, 210.0]
    assert ring[0] == ring[-1]


def test_native_episode_alignment_and_fail_closed_commit(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}", encoding="utf-8")
    prior = tmp_path / "prior.geojson"
    target = tmp_path / "target.geojson"
    prior.write_text("{}", encoding="utf-8")
    target.write_text("{}", encoding="utf-8")
    scenes = tmp_path / "scenes.jsonl"
    scenes.write_text(
        json.dumps(
            {
                "schema_version": "activemap-structured-map-scene-v1",
                "sample_id": "argotweak:segment-a",
                "dataset": "argotweak",
                "split": "val",
                "aoi_id": "segment-a",
                "native_sample_id": "segment-a",
                "prior_map_path": str(prior),
                "target_map_path": str(target),
                "observations": [
                    {
                        "observation_id": "bundle:1000",
                        "timestamp": "1000",
                        "modality": "camera_bundle",
                        "path": str(bundle),
                        "cost": 1.0,
                        "metadata": {},
                    }
                ],
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    proposals = tmp_path / "proposals.jsonl"
    proposals.write_text(
        json.dumps(
            {
                "split": "val",
                "segment_id": "segment-a",
                "timestamp": "1000",
                "gt_operation_counts": {"KEEP": 2, "ADD": 1, "DELETE": 1, "RESHAPE": 0},
                "proposals": [
                    _proposal("add", "ADD", 0.9),
                    _proposal("delete", "DELETE", 0.9),
                ],
                "city_se3_egovehicle": {
                    "rotation": [[1.0, 0.0], [0.0, 1.0]],
                    "translation": [10.0, 20.0],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "native.jsonl"

    summary = build_argotweak_native_episodes(proposals, scenes, output)
    row = json.loads(output.read_text(encoding="utf-8"))

    assert summary["episodes"] == 1
    assert row["frozen_perception"] is True
    assert row["evidence"][0]["commit_ready_proposal_ids"] == ["add"]
    assert row["evidence"][0]["blocked_proposals"]["delete"] == ("OBJECT_ASSIGNMENT_REQUIRED")
