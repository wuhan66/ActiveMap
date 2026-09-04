from scripts.summarize_muno21_candidate_grpo_formal import summarize


def _metric(deltas, low, high):
    return {
        "seed_deltas": deltas,
        "seed_mean_delta": sum(deltas) / len(deltas),
        "seed_mean_ci95_low": low,
        "seed_mean_ci95_high": high,
    }


def test_formal_summary_promotes_only_non_dominated_result():
    aggregate = {
        "protocol": {"test_assets_read": False, "model_seeds": [1, 2, 3]},
        "candidate_minus_seed_matched_sft": {
            "episode_utility_v2_balanced_auc": _metric([0.02, 0.03, 0.04], 0.01, 0.05),
            "false_edit_auc": _metric([-0.01, -0.02, 0.0], -0.02, 0.0),
        },
    }
    result = summarize(aggregate)
    assert result["promote_to_main"] is True
    aggregate["candidate_minus_seed_matched_sft"][
        "episode_utility_v2_balanced_auc"
    ] = _metric([0.02, -0.01, 0.04], -0.01, 0.05)
    assert summarize(aggregate)["decision"] == "appendix_diagnostic"
