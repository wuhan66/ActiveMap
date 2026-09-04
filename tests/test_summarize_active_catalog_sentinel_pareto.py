from scripts.summarize_active_catalog_sentinel_pareto import dominates, mark_pareto


def _row(utility: float, harmful: float, *, passed: bool = True) -> dict:
    return {
        "gate_passed": passed,
        "metrics": {
            "realized_utility_mean": utility,
            "selection_macro_f1": 0.8,
            "exact_evidence_recall": 0.2,
            "false_call_rate": 0.05,
            "harmful_call_rate_all_states": harmful,
            "mean_cost_per_state": 0.5,
            "mean_regret": 0.1,
        },
    }


def test_dominance_requires_no_safety_tradeoff() -> None:
    safe = _row(0.03, 0.05)
    risky = _row(0.04, 0.10)
    assert not dominates(safe, risky)
    assert not dominates(risky, safe)


def test_pareto_excludes_strictly_dominated_and_failed_rows() -> None:
    best = _row(0.04, 0.05)
    dominated = _row(0.03, 0.05)
    failed = _row(0.05, 0.01, passed=False)
    rows = [best, dominated, failed]
    mark_pareto(rows)
    assert [row["pareto"] for row in rows] == [True, False, False]
