#!/usr/bin/env python3
"""Evaluate a train-only-OOF calibrated agreement rule for SET_SELECT heads.

The rule is intentionally simple and auditable: acquire only when two frozen
action heads both clear their own train-only OOF STOP margins *and* select the
same public evidence ID.  Neither validation score is used to choose a margin,
weight, or fallback action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.train_sequential_set_utility_ranker import action_metrics, task_bootstrap
except ModuleNotFoundError as error:
    if error.name is None or not error.name.startswith("scripts"):
        raise
    from train_sequential_set_utility_ranker import action_metrics, task_bootstrap


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_summary(run: Path) -> dict[str, Any]:
    summary_path = run / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("test_assets_read") is not False:
        raise ValueError("source run violates frozen-test isolation")
    if payload.get("schema_version") != "structured-vla-set-action-head-v1":
        raise ValueError("source run is not a structured SET_SELECT head")
    return payload


def _load_scores(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty scores: {path}")
    seen = set()
    for row in rows:
        task_id = str(row.get("task_id"))
        if task_id in seen:
            raise ValueError("duplicate task score row")
        seen.add(task_id)
        candidates = list(row.get("candidate_ids", []))
        advantages = list(row.get("advantages", []))
        scores = list(row.get("relative_action_scores", []))
        if not candidates or len(candidates) != len(advantages) or len(candidates) != len(scores):
            raise ValueError("malformed candidate scores")
        if row.get("test_assets_read") is not False:
            raise ValueError("score row violates frozen-test isolation")
    return rows


def agreement_traces(
    primary_rows: list[dict[str, Any]],
    safety_rows: list[dict[str, Any]],
    *,
    primary_margin: float,
    safety_margin: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if len(primary_rows) != len(safety_rows):
        raise ValueError("source policies have different task support")
    traces = []
    primary_calls = safety_calls = same_candidate = agreed_calls = 0
    for primary, safety in zip(primary_rows, safety_rows, strict=True):
        if (
            str(primary["task_id"]) != str(safety["task_id"])
            or list(primary["candidate_ids"]) != list(safety["candidate_ids"])
            or list(primary["advantages"]) != list(safety["advantages"])
        ):
            raise ValueError("source score rows are not task/candidate aligned")
        candidates = [str(value) for value in primary["candidate_ids"]]
        advantages = [float(value) for value in primary["advantages"]]
        primary_scores = [float(value) for value in primary["relative_action_scores"]]
        safety_scores = [float(value) for value in safety["relative_action_scores"]]
        primary_index = max(range(len(candidates)), key=lambda index: (primary_scores[index], candidates[index]))
        safety_index = max(range(len(candidates)), key=lambda index: (safety_scores[index], candidates[index]))
        primary_call = primary_scores[primary_index] >= primary_margin
        safety_call = safety_scores[safety_index] >= safety_margin
        primary_calls += int(primary_call)
        safety_calls += int(safety_call)
        same_candidate += int(primary_index == safety_index)
        acquire = primary_call and safety_call and primary_index == safety_index
        agreed_calls += int(acquire)
        target_index = max(range(len(candidates)), key=lambda index: (advantages[index], candidates[index]))
        target_call = advantages[target_index] > 0.0
        predicted_index = primary_index if acquire else None
        traces.append(
            {
                "task_id": str(primary["task_id"]),
                "candidate_count": len(candidates),
                "target_selection": "ACQUIRE" if target_call else "STOP",
                "target_evidence_id": candidates[target_index] if target_call else None,
                "predicted_selection": "ACQUIRE" if acquire else "STOP",
                "predicted_evidence_id": candidates[predicted_index] if predicted_index is not None else None,
                "realized_utility": advantages[predicted_index] if predicted_index is not None else 0.0,
                "oracle_utility": max(max(advantages), 0.0),
                "primary_best_score": primary_scores[primary_index],
                "safety_best_score": safety_scores[safety_index],
                "same_candidate": primary_index == safety_index,
                "test_assets_read": False,
            }
        )
    count = len(traces)
    return traces, {
        "primary_call_rate": primary_calls / count,
        "safety_call_rate": safety_calls / count,
        "same_candidate_rate": same_candidate / count,
        "agreement_call_rate": agreed_calls / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("primary_run", type=Path)
    parser.add_argument("safety_run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    primary_summary = _load_summary(args.primary_run)
    safety_summary = _load_summary(args.safety_run)
    primary_margin = float(primary_summary["oof"]["safety_margin"])
    safety_margin = float(safety_summary["oof"]["safety_margin"])
    primary_oof = _load_scores(args.primary_run / "oof_raw_scores.jsonl")
    safety_oof = _load_scores(args.safety_run / "oof_raw_scores.jsonl")
    primary_val = _load_scores(args.primary_run / "validation_raw_scores.jsonl")
    safety_val = _load_scores(args.safety_run / "validation_raw_scores.jsonl")
    oof_traces, oof_coverage = agreement_traces(
        primary_oof, safety_oof, primary_margin=primary_margin, safety_margin=safety_margin
    )
    validation_traces, validation_coverage = agreement_traces(
        primary_val, safety_val, primary_margin=primary_margin, safety_margin=safety_margin
    )
    oof_metrics = action_metrics(oof_traces)
    validation_metrics = action_metrics(validation_traces)
    bootstrap = task_bootstrap(
        validation_traces, repetitions=args.bootstrap_repetitions, seed=args.seed
    )
    promotion = {
        "nonzero_calls": validation_metrics["call_rate"] > 0.0,
        "bounded_call_rate": validation_metrics["call_rate"] <= 0.50,
        "positive_utility": validation_metrics["utility_mean"] > 0.0,
        "task_utility_ci_above_zero": bootstrap["utility_mean"]["ci95_low"] > 0.0,
        "false_call_rate_at_most_0_10": validation_metrics["false_call_rate"] <= 0.10,
        "exact_candidate_recall_above_zero": validation_metrics["exact_candidate_recall"] > 0.0,
    }
    args.output.mkdir(parents=True)
    with (args.output / "oof_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in oof_traces:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (args.output / "validation_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in validation_traces:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "structured-vla-set-action-agreement-v1",
        "role": "diagnostic-only-train-only-oof-agreement-risk-policy",
        "controller_interface": "STOP-or-one-public-evidence-id",
        "rule": "ACQUIRE only if both source heads clear frozen OOF margins and choose the same candidate",
        "primary_margin": primary_margin,
        "safety_margin": safety_margin,
        "oof": {"coverage": oof_coverage, "metrics": oof_metrics},
        "validation": {"coverage": validation_coverage, "metrics": validation_metrics},
        "validation_task_bootstrap": bootstrap,
        "promotion_gate": {**promotion, "passed": all(promotion.values())},
        "sources": {
            "primary_summary": _sha256(args.primary_run / "summary.json"),
            "primary_oof_scores": _sha256(args.primary_run / "oof_raw_scores.jsonl"),
            "primary_validation_scores": _sha256(args.primary_run / "validation_raw_scores.jsonl"),
            "safety_summary": _sha256(args.safety_run / "summary.json"),
            "safety_oof_scores": _sha256(args.safety_run / "oof_raw_scores.jsonl"),
            "safety_validation_scores": _sha256(args.safety_run / "validation_raw_scores.jsonl"),
        },
        "test_assets_read": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
