from scripts.summarize_muno21_generated_oof_critic import METRICS


def test_required_promotion_metrics_are_present() -> None:
    assert set(METRICS) == {
        "terminal_accuracy",
        "false_edit_rate",
        "missed_edit_rate",
        "balanced_utility",
        "safety_utility",
    }
