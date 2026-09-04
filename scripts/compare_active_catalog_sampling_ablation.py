#!/usr/bin/env python3
"""Paired AOI comparison for weighted versus natural SFT sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_active_catalog_selector import active_catalog_metrics
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from evaluate_active_catalog_selector import active_catalog_metrics

INVARIANT_FIELDS = (
    "task_id",
    "aoi_id",
    "source_episode",
    "split",
    "budget",
    "oracle_step",
    "gt_edit",
    "draft_edit",
    "candidate_count",
    "target_selection",
    "target_evidence_id",
    "stop_utility",
    "oracle_utility",
)
HIGHER_IS_BETTER = (
    "selection_macro_f1",
    "realized_utility_mean",
    "exact_evidence_recall",
)
LOWER_IS_BETTER = (
    "mean_regret",
    "false_call_rate",
    "harmful_call_rate_all_states",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty trace: {path}")
    ids = [str(row["example_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate example_id in {path}")
    return rows


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _improvements(
    weighted: dict[str, float], unweighted: dict[str, float]
) -> dict[str, float]:
    result = {
        key: float(weighted[key]) - float(unweighted[key])
        for key in HIGHER_IS_BETTER
    }
    result.update(
        {
            key: float(unweighted[key]) - float(weighted[key])
            for key in LOWER_IS_BETTER
        }
    )
    return result


def compare_traces(
    weighted_rows: list[dict[str, Any]],
    unweighted_rows: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    weighted = {str(row["example_id"]): row for row in weighted_rows}
    unweighted = {str(row["example_id"]): row for row in unweighted_rows}
    if set(weighted) != set(unweighted):
        raise ValueError("weighted and unweighted traces have different example keys")
    for example_id in sorted(weighted):
        for field in INVARIANT_FIELDS:
            if weighted[example_id].get(field) != unweighted[example_id].get(field):
                raise ValueError(f"protocol mismatch at {example_id}.{field}")
    ordered_ids = sorted(weighted)
    weighted_ordered = [weighted[key] for key in ordered_ids]
    unweighted_ordered = [unweighted[key] for key in ordered_ids]
    weighted_metrics = active_catalog_metrics(weighted_ordered)
    unweighted_metrics = active_catalog_metrics(unweighted_ordered)
    observed = _improvements(weighted_metrics, unweighted_metrics)

    groups: dict[str, list[str]] = defaultdict(list)
    for example_id in ordered_ids:
        groups[str(weighted[example_id]["aoi_id"])].append(example_id)
    group_ids = sorted(groups)
    if len(group_ids) < 2:
        raise ValueError("paired AOI bootstrap requires at least two AOIs")
    distributions = {key: [] for key in observed}
    rng = random.Random(seed)
    for _ in range(repetitions):
        sampled_groups = rng.choices(group_ids, k=len(group_ids))
        sampled_ids = [key for group in sampled_groups for key in groups[group]]
        improvements = _improvements(
            active_catalog_metrics([weighted[key] for key in sampled_ids]),
            active_catalog_metrics([unweighted[key] for key in sampled_ids]),
        )
        for key, value in improvements.items():
            distributions[key].append(value)
    intervals = {
        key: {
            "observed_weighted_improvement": observed[key],
            "ci95_low": _quantile(values, 0.025),
            "ci95_high": _quantile(values, 0.975),
        }
        for key, values in distributions.items()
    }
    weighted_gate = {
        "valid_actions": all(bool(row.get("valid_action")) for row in weighted_ordered),
        "nonzero_calls": weighted_metrics["predicted_call_rate"] > 0.0,
        "utility_ci_above_zero": intervals["realized_utility_mean"]["ci95_low"] > 0.0,
        "macro_f1_observed_improves": observed["selection_macro_f1"] > 0.0,
        "false_calls_noninferior": intervals["false_call_rate"]["ci95_low"] >= -0.02,
        "harmful_calls_noninferior": intervals["harmful_call_rate_all_states"]["ci95_low"] >= -0.01,
        "exact_recall_observed_improves": observed["exact_evidence_recall"] > 0.0,
    }
    unweighted_gate = {
        "valid_actions": all(bool(row.get("valid_action")) for row in unweighted_ordered),
        "nonzero_calls": unweighted_metrics["predicted_call_rate"] > 0.0,
        "utility_ci_above_zero": intervals["realized_utility_mean"]["ci95_high"] < 0.0,
        "macro_f1_observed_improves": observed["selection_macro_f1"] < 0.0,
        "false_calls_noninferior": intervals["false_call_rate"]["ci95_high"] <= 0.02,
        "harmful_calls_noninferior": intervals["harmful_call_rate_all_states"]["ci95_high"] <= 0.01,
        "exact_recall_observed_improves": observed["exact_evidence_recall"] < 0.0,
    }
    weighted_passed = all(weighted_gate.values())
    unweighted_passed = all(unweighted_gate.values())
    if weighted_passed:
        decision = "weighted"
    elif unweighted_passed:
        decision = "unweighted"
    else:
        decision = "inconclusive"
    return {
        "sample_count": len(ordered_ids),
        "aoi_count": len(group_ids),
        "weighted_metrics": weighted_metrics,
        "unweighted_metrics": unweighted_metrics,
        "paired_aoi_bootstrap": {
            "repetitions": repetitions,
            "seed": seed,
            "intervals": intervals,
        },
        "weighted_gate": {**weighted_gate, "passed": weighted_passed},
        "unweighted_gate": {**unweighted_gate, "passed": unweighted_passed},
        "decision": decision,
    }


def _validate_summary(trace: Path) -> dict[str, Any]:
    summary_path = trace.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "active-catalog-selector-evaluation-v1":
        raise ValueError(f"unexpected evaluation summary: {summary_path}")
    if summary.get("test_assets_read") is not False:
        raise ValueError("sampling comparison only permits validation artifacts")
    if summary.get("sources", {}).get("trace_sha256") != _sha256(trace):
        raise ValueError(f"trace checksum mismatch: {trace}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weighted_trace", type=Path)
    parser.add_argument("unweighted_trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    weighted_summary = _validate_summary(args.weighted_trace)
    unweighted_summary = _validate_summary(args.unweighted_trace)
    for key in ("validation", "evaluation_index"):
        left = weighted_summary["sources"][key]["sha256"]
        right = unweighted_summary["sources"][key]["sha256"]
        if left != right:
            raise ValueError(f"weighted/unweighted {key} checksum mismatch")
    report = compare_traces(
        _load_jsonl(args.weighted_trace),
        _load_jsonl(args.unweighted_trace),
        repetitions=args.repetitions,
        seed=args.seed,
    )
    report.update(
        {
            "schema_version": "active-catalog-sampling-ablation-v1",
            "weighted_trace": str(args.weighted_trace.resolve()),
            "unweighted_trace": str(args.unweighted_trace.resolve()),
            "test_assets_read": False,
            "checkpoint_step": args.checkpoint_step,
            "diagnostic_only": args.diagnostic_only,
            "promotion_eligible": not args.diagnostic_only,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
