#!/usr/bin/env python3
"""Aggregate frozen validation traces from independent agreement-policy seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

try:
    from scripts.train_sequential_set_utility_ranker import action_metrics
except ModuleNotFoundError as error:
    if error.name is None or not error.name.startswith("scripts"):
        raise
    from train_sequential_set_utility_ranker import action_metrics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("runs must use LABEL=RUN_DIRECTORY")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not label or not (path / "summary.json").is_file() or not (path / "validation_traces.jsonl").is_file():
        raise argparse.ArgumentTypeError(f"invalid agreement run: {value}")
    return label, path


def load_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if summary.get("test_assets_read") is not False:
        raise ValueError("test access detected")
    traces = [json.loads(line) for line in (path / "validation_traces.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if not traces or any(row.get("test_assets_read") is not False for row in traces):
        raise ValueError("invalid validation traces")
    task_ids = [str(row["task_id"]) for row in traces]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("agreement traces must contain one row per task")
    return summary, traces


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def hierarchical_utility_interval(traces_by_seed: dict[str, list[dict[str, Any]]], repetitions: int, seed: int) -> dict[str, float]:
    labels = sorted(traces_by_seed)
    task_ids = [str(row["task_id"]) for row in traces_by_seed[labels[0]]]
    expected = {
        task_id: (str(row["target_selection"]), str(row["target_evidence_id"]))
        for task_id, row in zip(task_ids, traces_by_seed[labels[0]], strict=True)
    }
    utilities: dict[str, dict[str, float]] = {}
    for label in labels:
        indexed = {str(row["task_id"]): row for row in traces_by_seed[label]}
        if set(indexed) != set(task_ids):
            raise ValueError("seed task support differs")
        if any(
            (str(indexed[task_id]["target_selection"]), str(indexed[task_id]["target_evidence_id"])) != expected[task_id]
            for task_id in task_ids
        ):
            raise ValueError("seed target support differs")
        utilities[label] = {task_id: float(indexed[task_id]["realized_utility"]) for task_id in task_ids}
    observed = sum(utilities[label][task_id] for label in labels for task_id in task_ids) / (len(labels) * len(task_ids))
    rng = random.Random(seed)
    samples = []
    for _ in range(repetitions):
        sampled_labels = rng.choices(labels, k=len(labels))
        sampled_tasks = rng.choices(task_ids, k=len(task_ids))
        samples.append(
            sum(utilities[label][task_id] for label in sampled_labels for task_id in sampled_tasks)
            / (len(sampled_labels) * len(sampled_tasks))
        )
    return {
        "observed": observed,
        "ci95_low": quantile(samples, 0.025),
        "ci95_high": quantile(samples, 0.975),
        "bootstrap_probability_gt_zero": sum(value > 0.0 for value in samples) / repetitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if len(args.run) < 3:
        raise ValueError("agreement replication requires at least three independent seeds")
    labels = [label for label, _ in args.run]
    if len(labels) != len(set(labels)):
        raise ValueError("duplicate run label")
    summaries, traces_by_seed = {}, {}
    for label, path in args.run:
        summary, traces = load_run(path)
        summaries[label] = summary
        traces_by_seed[label] = traces
    per_seed = {
        label: {
            "metrics": action_metrics(traces),
            "summary_sha256": _sha256(dict(args.run)[label] / "summary.json"),
            "traces_sha256": _sha256(dict(args.run)[label] / "validation_traces.jsonl"),
        }
        for label, traces in traces_by_seed.items()
    }
    flattened = [row for traces in traces_by_seed.values() for row in traces]
    pooled = action_metrics(flattened)
    interval = hierarchical_utility_interval(traces_by_seed, args.bootstrap_repetitions, args.seed)
    always_stop = float(summaries[labels[0]].get("always_stop_macro_f1", 0.0))
    if always_stop <= 0.0:
        target_rate = sum(row["target_selection"] == "ACQUIRE" for row in traces_by_seed[labels[0]]) / len(traces_by_seed[labels[0]])
        always_stop = (2.0 * (1.0 - target_rate)) / (2.0 - target_rate) / 2.0
    promotion = {
        "three_or_more_model_seeds": len(labels) >= 3,
        "all_seeds_nonzero_calls": all(value["metrics"]["call_rate"] > 0.0 for value in per_seed.values()),
        "all_seeds_bounded_call_rate": all(value["metrics"]["call_rate"] <= 0.50 for value in per_seed.values()),
        "all_seeds_positive_utility": all(value["metrics"]["utility_mean"] > 0.0 for value in per_seed.values()),
        "all_seeds_false_call_rate_at_most_0_10": all(value["metrics"]["false_call_rate"] <= 0.10 for value in per_seed.values()),
        "hierarchical_utility_ci_above_zero": interval["ci95_low"] > 0.0,
        "mean_macro_f1_above_always_stop": pooled["macro_f1"] > always_stop,
    }
    args.output.mkdir(parents=True)
    summary = {
        "schema_version": "structured-vla-set-action-agreement-replicate-aggregate-v1",
        "role": "validation-only-three-seed-agreement-policy-replication",
        "model_seed_count": len(labels),
        "task_count": len(traces_by_seed[labels[0]]),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.seed,
        "per_seed": per_seed,
        "pooled_metrics": pooled,
        "hierarchical_bootstrap": {"grouping": "model_seed x task_id", "realized_utility_mean": interval},
        "always_stop_macro_f1": always_stop,
        "promotion_gate": {**promotion, "passed": all(promotion.values())},
        "test_assets_read": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
