import json

from scripts.build_active_catalog_robustness_slices import (
    build_slices,
    geometry_area,
)


def _trace(sample, episode, aoi, budget, target, prediction, confidence, uncertainty):
    return {
        "sample_id": sample,
        "source_episode": episode,
        "aoi_id": aoi,
        "split": "val",
        "budget": budget,
        "target_edit": target,
        "predicted_edit": prediction,
        "terminal_correct": target == prediction,
        "false_edit": target == "KEEP" and prediction != "KEEP",
        "missed_edit": target != "KEEP" and prediction == "KEEP",
        "wrong_edit": target != "KEEP" and prediction not in {target, "KEEP"},
        "acquisitions": 1,
        "steps": 2,
        "spent_cost": 1.0,
        "tool_calls": 0,
        "tool_cost": 0.0,
        "tool_belief_l1_delta": 0.0,
        "quality_gain": float(target == prediction),
        "quality_cost_utility": float(target == prediction) - 0.1,
        "episode_utility_v2_proxy_balanced": 0.0,
        "episode_utility_v2_proxy_safety": 0.0,
        "episode_utility_v2_proxy_cost_aware": 0.0,
        "selected_evidence_ids": ["e0"],
        "model_action_count": 1,
        "valid_action_count": 1,
        "fallback_count": 0,
        "events": [
            {
                "observable_state": {
                    "direct_draft": {"confidence": confidence},
                    "belief": {"uncertainty": uncertainty},
                    "candidate_evidence": [{"clear_fraction": 0.8}],
                }
            }
        ],
        "test_assets_read": False,
    }


def _write(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_polygon_area_accounts_for_holes():
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
            [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]],
        ],
    }
    assert geometry_area(geometry) == 15.0


def test_builds_aligned_observed_slices(tmp_path):
    episodes = tmp_path / "episodes.jsonl"
    episode_rows = []
    reference = []
    candidate = []
    for index, (aoi, target) in enumerate((("a", "KEEP"), ("b", "ADD"), ("c", "RESHAPE"))):
        episode = f"e{index}"
        episode_rows.append(
            {
                "episode_id": episode,
                "split": "val",
                "prior_geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [index + 1, 0], [index + 1, 1], [0, 1], [0, 0]]],
                },
            }
        )
        reference.append(
            _trace(f"s{index}", episode, aoi, 1.5, target, "KEEP", 0.2 + index * 0.2, 0.1 + index * 0.2)
        )
        candidate.append(
            _trace(f"s{index}", episode, aoi, 1.5, target, target, 0.2 + index * 0.2, 0.1 + index * 0.2)
        )
    _write(episodes, episode_rows)
    reference_path = tmp_path / "reference.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    _write(reference_path, reference)
    _write(candidate_path, candidate)
    result = build_slices(
        {"reference": reference_path, "candidate": candidate_path},
        episodes,
        reference="reference",
        candidate="candidate",
        expected_records=3,
        repetitions=0,
        seed=7,
    )
    assert set(result["dimensions"]["edit_type"]) == {"KEEP", "ADD", "RESHAPE"}
    assert result["dimensions"]["aoi"]["a"]["candidate_minus_reference_aoi_bootstrap"] is None
    assert result["test_assets_read"] is False
