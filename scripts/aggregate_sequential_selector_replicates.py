#!/usr/bin/env python3
"""Aggregate fixed-seed sequential SELECT evaluations without test access.

The unit of inference is not an individual SELECT row.  A model seed is
sampled together with a validation task, preserving the three direct-draft
states belonging to that task.  This prevents a single favorable LoRA seed or
correlated within-task states from being reported as independent evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("runs must use LABEL=EVALUATION_DIRECTORY")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not label or not path.is_dir():
        raise argparse.ArgumentTypeError(f"invalid run: {value}")
    return label, path


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def selector_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("empty selector traces")
    target = [row["target_selection"] == "ACQUIRE" for row in rows]
    predicted = [row["predicted_selection"] == "ACQUIRE" for row in rows]
    true_positive = sum(left and right for left, right in zip(target, predicted, strict=True))
    false_positive = sum(
        not left and right for left, right in zip(target, predicted, strict=True)
    )
    false_negative = sum(
        left and not right for left, right in zip(target, predicted, strict=True)
    )
    true_negative = len(rows) - true_positive - false_positive - false_negative
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    stop_precision = true_negative / max(true_negative + false_negative, 1)
    stop_recall = true_negative / max(true_negative + false_positive, 1)
    acquire_f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    stop_f1 = 2.0 * stop_precision * stop_recall / max(stop_precision + stop_recall, 1e-12)
    utility = [
        float(row["policy_relative_advantage"]) if call else 0.0
        for row, call in zip(rows, predicted, strict=True)
    ]
    return {
        "accuracy": (true_positive + true_negative) / len(rows),
        "macro_f1": (acquire_f1 + stop_f1) / 2.0,
        "acquire_precision": precision,
        "acquire_recall": recall,
        "predicted_call_rate": sum(predicted) / len(rows),
        "false_call_rate": false_positive / max(sum(not value for value in target), 1),
        "missed_call_rate": false_negative / max(sum(target), 1),
        "realized_utility_mean": sum(utility) / len(rows),
    }


def load_run(label: str, directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = directory / "summary.json"
    traces_path = directory / "traces.jsonl"
    if not summary_path.is_file() or not traces_path.is_file():
        raise FileNotFoundError(f"incomplete evaluation directory: {directory}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("test_assets_read"):
        raise ValueError(f"test access in {label}")
    traces = [json.loads(line) for line in traces_path.read_text(encoding="utf-8").splitlines()]
    if len(traces) != int(summary["sample_count"]):
        raise ValueError(f"trace count mismatch in {label}")
    if not all(row["split"] == "val" for row in traces):
        raise ValueError(f"non-validation trace in {label}")
    return summary, traces


def trace_validity_rate(
    traces: list[dict[str, Any]], field: str, *, fallback: str = "valid_action"
) -> float:
    """Read v2 validity fields while retaining immutable v1 receipts."""

    return sum(bool(row.get(field, row.get(fallback, False))) for row in traces) / len(traces)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260865)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if len(args.run) < 3:
        raise ValueError("replicate promotion requires at least three model seeds")
    if args.bootstrap_repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")

    labels: list[str] = []
    summaries: dict[str, dict[str, Any]] = {}
    traces_by_seed: dict[str, list[dict[str, Any]]] = {}
    reference_keys: list[tuple[str, str]] | None = None
    reference_targets: dict[tuple[str, str], tuple[str, float]] | None = None
    for label, directory in args.run:
        if label in summaries:
            raise ValueError(f"duplicate label: {label}")
        summary, traces = load_run(label, directory)
        keys = [(str(row["task_id"]), str(row["trajectory_id"])) for row in traces]
        targets = {
            key: (str(row["target_selection"]), float(row["policy_relative_advantage"]))
            for key, row in zip(keys, traces, strict=True)
        }
        if len(targets) != len(traces):
            raise ValueError(f"duplicate trajectory in {label}")
        if reference_keys is None:
            reference_keys, reference_targets = keys, targets
        elif keys != reference_keys or targets != reference_targets:
            raise ValueError("replicate evaluations do not share identical validation states")
        labels.append(label)
        summaries[label] = summary
        traces_by_seed[label] = traces

    assert reference_keys is not None
    task_ids = sorted({task_id for task_id, _ in reference_keys})
    by_seed_task = {
        label: {
            task_id: [row for row in traces if str(row["task_id"]) == task_id]
            for task_id in task_ids
        }
        for label, traces in traces_by_seed.items()
    }
    if any(not rows for by_task in by_seed_task.values() for rows in by_task.values()):
        raise ValueError("missing task state in replicate traces")

    per_seed = {
        label: {
            "metrics": selector_metrics(traces_by_seed[label]),
            "raw_json_valid_rate": trace_validity_rate(
                traces_by_seed[label], "raw_json_valid"
            ),
            "raw_schema_valid_rate": trace_validity_rate(
                traces_by_seed[label], "raw_schema_valid"
            ),
            "executable_action_valid_rate": trace_validity_rate(
                traces_by_seed[label], "executable_action_valid"
            ),
            # Backwards-compatible alias used by older tables.
            "valid_action_rate": trace_validity_rate(
                traces_by_seed[label], "executable_action_valid"
            ),
            "summary_sha256": hashlib.sha256(
                (directory / "summary.json").read_bytes()
            ).hexdigest(),
            "trace_sha256": hashlib.sha256(
                (directory / "traces.jsonl").read_bytes()
            ).hexdigest(),
        }
        for label, directory in args.run
    }
    pooled = [row for label in labels for row in traces_by_seed[label]]
    observed = selector_metrics(pooled)
    rng = random.Random(args.seed)
    distributions = {"realized_utility_mean": [], "macro_f1": []}
    for _ in range(args.bootstrap_repetitions):
        sampled_seeds = rng.choices(labels, k=len(labels))
        sampled_tasks = rng.choices(task_ids, k=len(task_ids))
        sampled = [
            row
            for label in sampled_seeds
            for task_id in sampled_tasks
            for row in by_seed_task[label][task_id]
        ]
        metrics = selector_metrics(sampled)
        for key in distributions:
            distributions[key].append(metrics[key])
    intervals = {
        key: {
            "observed": observed[key],
            "ci95_low": quantile(values, 0.025),
            "ci95_high": quantile(values, 0.975),
            "bootstrap_probability_gt_zero": sum(value > 0.0 for value in values)
            / len(values),
        }
        for key, values in distributions.items()
    }
    always_stop = summaries[labels[0]]["always_stop_macro_f1"]
    gate = {
        "three_or_more_model_seeds": len(labels) >= 3,
        "all_actions_executable": all(
            record["executable_action_valid_rate"] == 1.0
            for record in per_seed.values()
        ),
        "all_seeds_make_nonzero_calls": all(
            record["metrics"]["predicted_call_rate"] > 0.0 for record in per_seed.values()
        ),
        "all_seeds_bound_call_rate": all(
            record["metrics"]["predicted_call_rate"] <= 0.50
            for record in per_seed.values()
        ),
        "all_seeds_positive_utility": all(
            record["metrics"]["realized_utility_mean"] > 0.0
            for record in per_seed.values()
        ),
        "hierarchical_utility_ci_above_zero": intervals["realized_utility_mean"]
        ["ci95_low"]
        > 0.0,
        "mean_macro_f1_above_always_stop": observed["macro_f1"] > always_stop,
    }
    report = {
        "schema_version": "sequential-selector-replicate-aggregate-v1",
        "evaluation_role": "validation-only-model-seed-by-task-hierarchical-bootstrap",
        "model_seed_count": len(labels),
        "task_count": len(task_ids),
        "states_per_seed": len(reference_keys),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.seed,
        "per_seed": per_seed,
        "pooled_metrics": observed,
        "hierarchical_bootstrap": {"grouping": "model_seed x task_id", "intervals": intervals},
        "always_stop_macro_f1": always_stop,
        "promotion_gate": {**gate, "passed": all(gate.values())},
        "test_assets_read": False,
        "next_action": (
            "manual joint-SFT approval only after this gate passes"
            if all(gate.values())
            else "diagnose SELECT objective; do not launch joint SFT or GRPO"
        ),
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Sequential SELECT Replication",
        "",
        "| Seed | Utility | Macro-F1 | Call rate | False-call rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in labels:
        metric = per_seed[label]["metrics"]
        lines.append(
            f"| {label} | {metric['realized_utility_mean']:.6f} | "
            f"{metric['macro_f1']:.6f} | {metric['predicted_call_rate']:.6f} | "
            f"{metric['false_call_rate']:.6f} |"
        )
    interval = intervals["realized_utility_mean"]
    lines.extend(
        [
            "",
            "Hierarchical utility (model seed x task): "
            f"{interval['observed']:.6f} "
            f"[95% CI {interval['ci95_low']:.6f}, {interval['ci95_high']:.6f}]",
            "",
            f"Promotion gate: **{'PASS' if all(gate.values()) else 'FAIL'}**",
        ]
    )
    (args.output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
