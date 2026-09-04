import json
from pathlib import Path

from scripts.aggregate_active_catalog_policy_seeds import aggregate


def _write(path: Path, quality_gain: float) -> None:
    rows = []
    for aoi in range(3):
        for index in range(2):
            rows.append(
                {
                    "source_episode": f"episode-{aoi}-{index}",
                    "budget": 1.0,
                    "aoi_id": f"aoi-{aoi}",
                    "target_edit": "ADD",
                    "terminal_correct": quality_gain > 0,
                    "false_edit": False,
                    "missed_edit": False,
                    "wrong_edit": False,
                    "spent_cost": 0.1 if quality_gain > 0 else 0.0,
                    "quality_gain": quality_gain,
                    "quality_cost_utility": quality_gain - 0.01,
                    "episode_utility_v2_proxy_balanced": quality_gain,
                    "episode_utility_v2_proxy_safety": quality_gain,
                    "episode_utility_v2_proxy_cost_aware": quality_gain,
                    "tool_calls": 0,
                    "tool_belief_l1_delta": 0.0,
                    "steps": 1,
                    "acquisitions": 0,
                    "valid_action": True,
                    "fallback_used": False,
                    "split": "val",
                    "test_assets_read": False,
                }
            )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_aggregate_uses_shared_aoi_bootstrap(tmp_path: Path):
    reference = tmp_path / "reference.jsonl"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write(reference, 0.0)
    _write(first, 0.2)
    _write(second, 0.1)

    result = aggregate(
        [(1, first), (2, second)],
        reference,
        repetitions=100,
        seed=7,
    )

    assert result["seed_count"] == 2
    assert result["shared_aoi_resampling_across_model_seeds"] is True
    delta = result["candidate_minus_reference"]["mean_quality_gain"]
    assert delta["observed_delta"] == 0.15
    assert delta["ci95_low"] > 0.0
