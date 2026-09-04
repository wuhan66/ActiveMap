import json

import pytest

from scripts.aggregate_active_catalog_paired_policy_seeds import aggregate


def _row(episode, aoi, *, correct, utility):
    return {
        "source_episode": episode,
        "aoi_id": aoi,
        "split": "val",
        "budget": 2.0,
        "target_edit": "ADD",
        "predicted_edit": "ADD" if correct else "KEEP",
        "terminal_correct": correct,
        "false_edit": False,
        "missed_edit": not correct,
        "wrong_edit": False,
        "acquisitions": int(correct),
        "steps": 1 + int(correct),
        "spent_cost": 0.1 * int(correct),
        "quality_gain": utility + 0.1,
        "quality_cost_utility": utility,
        "episode_utility_v2_proxy_balanced": utility,
        "episode_utility_v2_proxy_safety": utility,
        "episode_utility_v2_proxy_cost_aware": utility,
        "model_action_count": 1,
        "valid_action_count": 1,
        "fallback_count": 0,
        "test_assets_read": False,
    }


def _trace(tmp_path, name, rows):
    path = tmp_path / f"{name}.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def test_seed_matched_aggregate_uses_independent_references(tmp_path):
    pairs = []
    for seed in (1, 2, 3):
        reference = _trace(
            tmp_path,
            f"reference-{seed}",
            [
                _row("e1", "a", correct=False, utility=0.0),
                _row("e2", "b", correct=False, utility=0.0),
            ],
        )
        candidate = _trace(
            tmp_path,
            f"candidate-{seed}",
            [
                _row("e1", "a", correct=True, utility=0.2),
                _row("e2", "b", correct=True, utility=0.3),
            ],
        )
        pairs.append((seed, candidate, reference))

    result = aggregate(pairs, repetitions=100, seed=9)

    assert result["seed_matched_references"] is True
    assert result["seed_count"] == 3
    assert result["strict_utility_gain_ci95"] is True
    assert (
        result["seed_variation"]["terminal_accuracy"]["sample_std_delta"] == 0.0
    )
    assert set(result["per_seed_candidate_minus_sft"]) == {1, 2, 3}
    assert (
        result["candidate_minus_seed_matched_sft"]["terminal_accuracy"][
            "observed_delta"
        ]
        == 1.0
    )


def test_seed_matched_aggregate_rejects_cross_seed_protocol_drift(tmp_path):
    reference1 = _trace(
        tmp_path, "r1", [_row("e1", "a", correct=False, utility=0.0)]
    )
    candidate1 = _trace(
        tmp_path, "c1", [_row("e1", "a", correct=True, utility=0.2)]
    )
    reference2 = _trace(
        tmp_path, "r2", [_row("e2", "b", correct=False, utility=0.0)]
    )
    candidate2 = _trace(
        tmp_path, "c2", [_row("e2", "b", correct=True, utility=0.2)]
    )

    with pytest.raises(ValueError, match="cross-seed protocol mismatch"):
        aggregate(
            [(1, candidate1, reference1), (2, candidate2, reference2)],
            repetitions=10,
            seed=3,
        )
