"""Cross-seed summaries for validation-gated updater experiments."""

from __future__ import annotations

from statistics import fmean, stdev
from typing import Any


def _reported_candidate(decision: dict[str, Any]) -> dict[str, Any]:
    candidates = list(decision.get("candidates", []))
    if not candidates:
        raise ValueError("promotion decision has no candidates")
    promoted_name = decision.get("promoted_checkpoint")
    if promoted_name is not None:
        for candidate in candidates:
            if candidate["checkpoint_name"] == promoted_name:
                return candidate
        raise ValueError(f"promoted checkpoint {promoted_name!r} is not a candidate")
    feasible = [candidate for candidate in candidates if candidate["constraint_satisfied"]]
    if not feasible:
        raise ValueError("promotion decision has no safety-feasible candidate")
    return max(
        feasible,
        key=lambda candidate: (
            float(candidate["selected"]["macro_f1"]),
            float(candidate["selected"]["delete_f1"]),
        ),
    )


def aggregate_seed_decisions(decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate selected validation points while enforcing single test access."""

    if not decisions:
        raise ValueError("at least one seed decision is required")
    test_seeds = [
        seed for seed, decision in decisions.items() if decision.get("test_evaluation") is not None
    ]
    if len(test_seeds) > 1:
        raise ValueError(f"test evaluation appears in multiple seeds: {test_seeds}")

    rows = []
    for seed, decision in decisions.items():
        candidate = _reported_candidate(decision)
        selected = candidate["selected"]
        rows.append(
            {
                "seed": seed,
                "status": decision["status"],
                "checkpoint_name": candidate["checkpoint_name"],
                "macro_f1": float(selected["macro_f1"]),
                "delete_f1": float(selected["delete_f1"]),
                "false_edit_rate": float(selected["false_edit_rate"]),
                "edit_accuracy": float(selected["edit_accuracy"]),
                "test_accessed": decision.get("test_evaluation") is not None,
            }
        )

    aggregate = {}
    for metric in ("macro_f1", "delete_f1", "false_edit_rate", "edit_accuracy"):
        values = [float(row[metric]) for row in rows]
        aggregate[metric] = {
            "mean": fmean(values),
            "sample_std": stdev(values) if len(values) > 1 else 0.0,
            "minimum": min(values),
            "maximum": max(values),
        }
    return {
        "seed_count": len(rows),
        "promotion_count": sum(str(row["status"]).startswith("promoted") for row in rows),
        "test_evaluation_count": len(test_seeds),
        "test_seed": test_seeds[0] if test_seeds else None,
        "seeds": rows,
        "aggregate": aggregate,
    }
