import json

from scripts.aggregate_active_catalog_tool_branches import aggregate


def _row(episode, aoi, utility, *, cost, calls, belief_delta):
    return {
        "split": "val",
        "test_assets_read": False,
        "source_episode": episode,
        "budget": 2.0,
        "aoi_id": aoi,
        "target_edit": "KEEP",
        "terminal_correct": True,
        "false_edit": False,
        "missed_edit": False,
        "acquisitions": 1,
        "steps": 2,
        "spent_cost": cost,
        "tool_calls": calls,
        "tool_belief_l1_delta": belief_delta,
        "quality_gain": utility + cost,
        "quality_cost_utility": utility,
        "model_action_count": 1,
        "valid_action_count": 1,
        "fallback_count": 0,
    }


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_shared_aoi_bootstrap_aggregates_complete_tool_seeds(tmp_path):
    records = []
    for seed in (1, 2):
        for label, utility, cost, calls in (
            ("no_tool", 0.10, 0.5, 0),
            ("forced", 0.12, 1.5, 2),
            ("selective", 0.30, 0.8, 1),
        ):
            path = tmp_path / f"{seed}-{label}.jsonl"
            _write(
                path,
                [
                    _row("e1", "a", utility, cost=cost, calls=calls, belief_delta=0.1 * calls),
                    _row("e2", "b", utility, cost=cost, calls=calls, belief_delta=0.1 * calls),
                ],
            )
            records.append((seed, label, path))
    result = aggregate(records, repetitions=50, seed=9)
    assert result["seed_count"] == 2
    assert result["shared_aoi_resampling_across_seeds"] is True
    assert result["paired_aoi_comparisons"]["selective_minus_no_tool"]["intervals"][
        "mean_quality_cost_utility"
    ]["ci95_low"] > 0.0
    assert result["metrics"]["selective"]["mean_tool_calls"] == 1.0


def test_optional_frozen_prior_branch_reports_recurrent_comparison(tmp_path):
    records = []
    for seed in (1, 2):
        for label, utility, cost, calls in (
            ("no_tool", 0.10, 0.5, 0),
            ("forced", 0.12, 1.5, 2),
            ("selective_frozen_prior", 0.20, 0.8, 1),
            ("selective", 0.30, 0.8, 1),
        ):
            path = tmp_path / f"{seed}-{label}.jsonl"
            _write(
                path,
                [
                    _row("e1", "a", utility, cost=cost, calls=calls, belief_delta=0.1 * calls),
                    _row("e2", "b", utility, cost=cost, calls=calls, belief_delta=0.1 * calls),
                ],
            )
            records.append((seed, label, path))
    result = aggregate(records, repetitions=50, seed=9)
    interval = result["paired_aoi_comparisons"][
        "selective_minus_selective_frozen_prior"
    ]["intervals"]["mean_quality_cost_utility"]
    assert interval["ci95_low"] > 0.0
