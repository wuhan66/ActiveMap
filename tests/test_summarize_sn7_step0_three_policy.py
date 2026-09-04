from scripts.summarize_sn7_step0_three_policy import summarize


def _row(seed: int, index: int, variant: str):
    benefit = variant == "benefit"
    forced = variant == "forced"
    return {
        "sample_id": f"s{index}",
        "aoi_id": f"a{index}",
        "split": "val",
        "test_assets_read": False,
        "predicted_edit": "KEEP" if benefit else "DELETE",
        "terminal_correct": benefit,
        "false_edit": not benefit,
        "missed_edit": False,
        "wrong_edit": False,
        "quality_gain": 0.0,
        "quality_cost_utility": 0.1 if benefit else 0.0,
        "tool_calls": 1 if benefit else 2 if forced else 0,
        "tool_belief_l1_delta": 0.1 if benefit or forced else 0.0,
    }


def test_summarizes_three_policy_seed_matrix():
    traces = {
        seed: {
            variant: [_row(seed, index, variant) for index in range(2)]
            for variant in ("notool", "benefit", "forced")
        }
        for seed in (1, 2, 3)
    }
    result = summarize(traces, split="val", repetitions=20, bootstrap_seed=7)
    assert result["comparisons"]["benefit_vs_notool"][
        "hierarchical_seed_aoi_bootstrap"
    ]["terminal_correct"]["ci95_low"] > 0
    assert result["protocol"]["test_assets_read"] is False
