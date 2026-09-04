#!/usr/bin/env python3
"""Audit label churn and task-bootstrap utility in a policy-relative branch cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scripts.cache_policy_relative_vlm_branches import policy_relative_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_bootstrap(
    rows: list[dict[str, Any]], *, repetitions: int, seed: int
) -> dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["task_id"]), []).append(row)
    tasks = sorted(grouped)
    if len(tasks) < 2:
        raise ValueError("task bootstrap requires at least two tasks")
    rng = np.random.default_rng(seed)
    keys = (
        "policy_relative_oracle_gain_over_direct",
        "static_label_gain_over_direct",
        "static_label_acquisition_regret",
        "static_label_precision",
        "static_label_recall",
        "policy_relative_positive_rate",
    )
    samples = {key: [] for key in keys}
    for _ in range(repetitions):
        selected = rng.choice(tasks, size=len(tasks), replace=True)
        replicate = [row for task_id in selected for row in grouped[str(task_id)]]
        metrics = policy_relative_metrics(replicate)
        for key in keys:
            samples[key].append(float(metrics[key]))
    return {
        "unit": "task_id",
        "task_count": len(tasks),
        "repetitions": repetitions,
        "seed": seed,
        "intervals": {
            key: {
                "mean": float(np.mean(values)),
                "low_95": float(np.quantile(values, 0.025)),
                "high_95": float(np.quantile(values, 0.975)),
            }
            for key, values in samples.items()
        },
    }


def _category(row: dict[str, Any]) -> str:
    static = bool(row["static_use_tool"])
    relative = bool(row["policy_relative_use_tool"])
    if static and relative:
        return "retained_positive"
    if static:
        return "stale_positive"
    if relative:
        return "new_positive"
    return "retained_stop"


def analyze(
    branch_root: Path, *, repetitions: int, seed: int
) -> dict[str, Any]:
    summary_path = branch_root / "summary.json"
    trace_path = branch_root / "traces.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("test_assets_read") is not False:
        raise ValueError("branch cache violates the frozen-test protocol")
    if summary.get("trace_sha256") != _sha256(trace_path):
        raise ValueError("branch cache trace hash mismatch")
    rows = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line
    ]
    categories = Counter(_category(row) for row in rows)
    target_by_category: dict[str, Counter[str]] = {}
    transition_by_category: dict[str, Counter[str]] = {}
    for row in rows:
        category = _category(row)
        target_by_category.setdefault(category, Counter())[str(row["target_operation"])] += 1
        transition = f'{row["direct_operation"]}->{row["post_tool_operation"]}'
        transition_by_category.setdefault(category, Counter())[transition] += 1
    return {
        "schema_version": "policy-relative-branch-cache-audit-v1",
        "source_summary": str(summary_path.resolve()),
        "source_summary_sha256": _sha256(summary_path),
        "metrics": policy_relative_metrics(rows),
        "label_categories": dict(sorted(categories.items())),
        "target_operation_by_category": {
            key: dict(sorted(value.items())) for key, value in sorted(target_by_category.items())
        },
        "direct_to_post_transition_by_category": {
            key: dict(sorted(value.items()))
            for key, value in sorted(transition_by_category.items())
        },
        "advantage_histogram": dict(
            sorted(
                Counter(str(float(row["policy_relative_advantage"])) for row in rows).items(),
                key=lambda item: float(item[0]),
            )
        ),
        "task_bootstrap": task_bootstrap(rows, repetitions=repetitions, seed=seed),
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("branch_root", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    result = analyze(args.branch_root, repetitions=args.repetitions, seed=args.seed)
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
