from scripts.evaluate_selector_checkpoint import aggregate_rows, cluster_bootstrap


def _row(aoi, called, target, harmful, exact, utility):
    return {
        "aoi_id": aoi,
        "called": called,
        "target_acquire": target,
        "false_call": called and not target,
        "harmful_call": harmful,
        "exact_acquire": exact,
        "utility": utility,
        "regret": 0.2 - utility,
    }


def test_aggregate_rows_reports_conditional_call_metrics() -> None:
    rows = [
        _row("a", True, True, False, True, 0.2),
        _row("a", True, False, True, False, -0.1),
        _row("b", False, True, False, False, 0.0),
        _row("b", False, False, False, False, 0.0),
    ]
    metrics = aggregate_rows(rows)
    assert metrics["call_rate"] == 0.5
    assert metrics["false_call_rate"] == 0.25
    assert metrics["harmful_call_fraction"] == 0.5
    assert metrics["acquire_recall"] == 0.5
    assert metrics["exact_acquire_recall"] == 0.5


def test_cluster_bootstrap_is_reproducible() -> None:
    rows = [
        _row("a", True, True, False, True, 0.2),
        _row("b", False, True, False, False, 0.0),
    ]
    assert cluster_bootstrap(rows, draws=50, seed=3) == cluster_bootstrap(
        rows, draws=50, seed=3
    )
