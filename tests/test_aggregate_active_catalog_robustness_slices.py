import json

from scripts.aggregate_active_catalog_robustness_slices import aggregate


def _write(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
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
        "wrong_edit": False,
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


def test_aggregates_matched_seed_slices_without_test_access(tmp_path):
    episodes = tmp_path / "episodes.jsonl"
    _write(
        episodes,
        [
            {
                "episode_id": "e1",
                "split": "val",
                "prior_geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            },
            {
                "episode_id": "e2",
                "split": "val",
                "prior_geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [2, 0], [2, 1], [0, 1], [0, 0]]],
                },
            },
        ],
    )
    records = []
    for seed in (1, 2):
        reference = tmp_path / f"reference-{seed}.jsonl"
        candidate = tmp_path / f"candidate-{seed}.jsonl"
        _write(
            reference,
            [
                _trace("s1", "e1", "a", 1.5, "KEEP", "KEEP", 0.2, 0.2),
                _trace("s2", "e2", "b", 1.5, "ADD", "KEEP", 0.8, 0.8),
            ],
        )
        _write(
            candidate,
            [
                _trace("s1", "e1", "a", 1.5, "KEEP", "KEEP", 0.2, 0.2),
                _trace("s2", "e2", "b", 1.5, "ADD", "ADD", 0.8, 0.8),
            ],
        )
        records.append((seed, reference, candidate))
    result = aggregate(
        records,
        episodes,
        expected_records=2,
        repetitions=0,
        seed=7,
    )
    assert result["seed_count"] == 2
    assert result["dimensions"]["edit_type"]["ADD"][
        "candidate_minus_reference_observed_mean"
    ]["terminal_accuracy"] > 0.0
    assert result["test_assets_read"] is False
