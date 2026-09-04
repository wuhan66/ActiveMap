#!/usr/bin/env python3
"""Paired validation audit for ungated and reliability-gated belief updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METRICS = (
    "terminal_correct",
    "false_edit",
    "missed_edit",
    "spent_cost",
    "tool_calls",
    "tool_successes",
    "mean_tool_belief_l1_delta",
    "quality_cost_utility",
    "joint_utility",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _mean(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return statistics.fmean(values) if values else 0.0


def _fingerprint(protocol: dict[str, Any], key: str) -> str | None:
    value = protocol.get("source_fingerprints", {}).get(key)
    if isinstance(value, list):
        return hashlib.sha256(
            json.dumps([item.get("sha256") for item in value], sort_keys=True).encode()
        ).hexdigest()
    return value.get("sha256") if isinstance(value, dict) else None


def _protocol_checks(left: dict[str, Any], right: dict[str, Any]) -> dict[str, bool]:
    left_protocol = left.get("protocol", {})
    right_protocol = right.get("protocol", {})
    checks = {
        "validation_only": left_protocol.get("test_assets_read") is False
        and right_protocol.get("test_assets_read") is False,
        "same_split": left_protocol.get("split") == right_protocol.get("split") == "val",
        "same_budgets": left_protocol.get("budgets") == right_protocol.get("budgets"),
        "same_seed": left_protocol.get("evaluation_seed") == right_protocol.get("evaluation_seed"),
        "same_gate_threshold": left_protocol.get("tool_need_gate", {}).get("threshold")
        == right_protocol.get("tool_need_gate", {}).get("threshold"),
    }
    for key in ("states", "episodes", "adapter", "selector_checkpoints", "tool_supervision", "tool_need_gate"):
        checks[f"same_{key}_fingerprint"] = _fingerprint(left_protocol, key) == _fingerprint(
            right_protocol, key
        )
    left_belief = _fingerprint(left_protocol, "tool_belief_checkpoint")
    right_belief = _fingerprint(right_protocol, "tool_belief_checkpoint")
    checks["belief_checkpoint_differs"] = bool(left_belief and right_belief and left_belief != right_belief)
    return checks


def _bootstrap_task_delta(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    metric: str,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    by_task: dict[str, list[float]] = defaultdict(list)
    for ungated, gated in pairs:
        by_task[str(ungated["task_id"])].append(float(gated[metric]) - float(ungated[metric]))
    task_deltas = [statistics.fmean(values) for values in by_task.values()]
    rng = random.Random(seed)
    samples = []
    for _ in range(repetitions):
        samples.append(statistics.fmean(rng.choice(task_deltas) for _ in task_deltas))
    samples.sort()
    lo = samples[int(0.025 * (repetitions - 1))]
    hi = samples[int(0.975 * (repetitions - 1))]
    return {"mean_delta": statistics.fmean(task_deltas), "ci95_low": lo, "ci95_high": hi}


def compare(
    ungated_summary: dict[str, Any],
    gated_summary: dict[str, Any],
    ungated_rows: list[dict[str, Any]],
    gated_rows: list[dict[str, Any]],
    repetitions: int = 5000,
    seed: int = 20260718,
) -> dict[str, Any]:
    ungated = {row["sample_id"]: row for row in ungated_rows}
    gated = {row["sample_id"]: row for row in gated_rows}
    if len(ungated) != len(ungated_rows) or len(gated) != len(gated_rows):
        raise ValueError("sample_id must be unique within each rollout")
    if set(ungated) != set(gated):
        raise ValueError("paired rollouts do not contain identical sample_ids")
    pairs = [(ungated[sample_id], gated[sample_id]) for sample_id in sorted(ungated)]
    for left, right in pairs:
        for key in ("task_id", "split", "budget", "target", "evaluation_seed"):
            if left[key] != right[key]:
                raise ValueError(f"pair mismatch for {left['sample_id']}: {key}")

    called_ungated = {row["sample_id"] for row in ungated_rows if int(row["tool_calls"]) > 0}
    called_gated = {row["sample_id"] for row in gated_rows if int(row["tool_calls"]) > 0}
    changed = []
    for left, right in pairs:
        if left["prediction"] != right["prediction"] or left["selected_evidence_ids"] != right["selected_evidence_ids"]:
            changed.append(
                {
                    "sample_id": left["sample_id"],
                    "task_id": left["task_id"],
                    "budget": left["budget"],
                    "target": left["target"],
                    "ungated_prediction": left["prediction"],
                    "gated_prediction": right["prediction"],
                    "ungated_false_edit": bool(left["false_edit"]),
                    "gated_false_edit": bool(right["false_edit"]),
                    "ungated_cost": float(left["spent_cost"]),
                    "gated_cost": float(right["spent_cost"]),
                    "ungated_utility": float(left["quality_cost_utility"]),
                    "gated_utility": float(right["quality_cost_utility"]),
                    "ungated_evidence": left["selected_evidence_ids"],
                    "gated_evidence": right["selected_evidence_ids"],
                }
            )

    means = {
        metric: {
            "ungated": _mean(ungated_rows, metric),
            "gated": _mean(gated_rows, metric),
            "delta_gated_minus_ungated": _mean(gated_rows, metric) - _mean(ungated_rows, metric),
        }
        for metric in METRICS
    }
    bootstrap = {
        metric: _bootstrap_task_delta(pairs, metric, repetitions, seed + index)
        for index, metric in enumerate(("quality_cost_utility", "false_edit", "terminal_correct", "spent_cost"))
    }
    protocol_checks = _protocol_checks(ungated_summary, gated_summary)
    checks = {
        **protocol_checks,
        "identical_sample_pairs": len(pairs) == len(ungated_rows) == len(gated_rows),
        "identical_called_samples": called_ungated == called_gated,
        "nonzero_successful_tool_calls": sum(int(row["tool_successes"]) for row in gated_rows) > 0,
        "utility_improves": means["quality_cost_utility"]["delta_gated_minus_ungated"] > 0.0,
        "false_edit_noninferior": means["false_edit"]["delta_gated_minus_ungated"] <= 0.0,
        "terminal_accuracy_noninferior": means["terminal_correct"]["delta_gated_minus_ungated"] >= 0.0,
        "cost_noninferior": means["spent_cost"]["delta_gated_minus_ungated"] <= 0.0,
        "no_false_edit_after_called_tools": not any(
            bool(row["false_edit"]) for row in gated_rows if int(row["tool_calls"]) > 0
        ),
        "utility_ci_excludes_zero": bootstrap["quality_cost_utility"]["ci95_low"] > 0.0,
    }
    checks["passed"] = all(checks.values())
    return {
        "schema_version": "muno21-reliability-closed-loop-paired-v1",
        "sample_count": len(pairs),
        "task_count": len({left["task_id"] for left, _ in pairs}),
        "means": means,
        "task_paired_bootstrap": bootstrap,
        "called_samples": {
            "ungated": sorted(called_ungated),
            "gated": sorted(called_gated),
        },
        "changed_rollouts": changed,
        "checks": checks,
        "promotion_passed": checks["passed"],
        "claim_boundary": "Single-seed validation-only diagnostic; test remains frozen.",
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ungated_summary", type=Path)
    parser.add_argument("ungated_rollouts", type=Path)
    parser.add_argument("gated_summary", type=Path)
    parser.add_argument("gated_rollouts", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = compare(
        json.loads(args.ungated_summary.read_text(encoding="utf-8")),
        json.loads(args.gated_summary.read_text(encoding="utf-8")),
        _read_jsonl(args.ungated_rollouts),
        _read_jsonl(args.gated_rollouts),
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
