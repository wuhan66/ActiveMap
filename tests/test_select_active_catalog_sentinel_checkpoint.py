from scripts.select_active_catalog_sentinel_checkpoint import choose_candidate


def _candidate(label: str, utility: float, *, eligible: bool = True) -> dict:
    return {
        "label": label,
        "eligible": eligible,
        "metrics": {
            "realized_utility_mean": utility,
            "selection_macro_f1": 0.5,
            "exact_evidence_recall": 0.2,
            "false_call_rate": 0.1,
        },
    }


def test_checkpoint_selection_rejects_all_stop_candidates() -> None:
    assert choose_candidate([_candidate("all-stop", 0.0, eligible=False)]) is None


def test_checkpoint_selection_prioritizes_realized_utility() -> None:
    selected = choose_candidate([_candidate("low", 0.01), _candidate("high", 0.03)])
    assert selected is not None
    assert selected["label"] == "high"
