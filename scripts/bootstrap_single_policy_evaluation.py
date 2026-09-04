#!/usr/bin/env python3
"""Task-grouped bootstrap for one frozen policy-versus-direct evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from scripts.bootstrap_semantic_vlm_results import _metrics, _quantile


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def single_seed_bootstrap(
    rows: list[dict[str, Any]], *, repetitions: int, seed: int
) -> dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    normalized = [
        {
            **row,
            "baseline_operation": row.get("baseline_operation", row["direct_operation"]),
            "baseline_utility": row.get("baseline_utility", row["direct_utility"]),
        }
        for row in rows
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in normalized:
        if row.get("split") != "val":
            raise ValueError("single-seed bootstrap only permits validation rows")
        grouped.setdefault(str(row["task_id"]), []).append(row)
    tasks = sorted(grouped)
    if len(tasks) < 2:
        raise ValueError("task bootstrap requires at least two tasks")
    observed = _metrics(normalized)
    distributions = {name: [] for name in observed}
    rng = random.Random(seed)
    for _ in range(repetitions):
        sampled = rng.choices(tasks, k=len(tasks))
        metrics = _metrics([row for task_id in sampled for row in grouped[task_id]])
        for name, value in metrics.items():
            distributions[name].append(value)
    intervals = {
        name: {
            "observed": observed[name],
            "ci95_low": _quantile(values, 0.025),
            "ci95_high": _quantile(values, 0.975),
        }
        for name, values in distributions.items()
    }
    for name in ("operation_macro_f1_delta", "mean_utility_delta"):
        finite = [value for value in distributions[name] if math.isfinite(value)]
        intervals[name]["bootstrap_probability_gt_zero"] = sum(
            value > 0.0 for value in finite
        ) / len(finite)
    finite_false = [
        value for value in distributions["false_edit_rate_delta"] if math.isfinite(value)
    ]
    intervals["false_edit_rate_delta"]["bootstrap_probability_le_0p02"] = (
        sum(value <= 0.02 for value in finite_false) / len(finite_false)
        if finite_false
        else math.nan
    )
    return {
        "schema_version": "single-policy-task-bootstrap-v1",
        "sample_count": len(normalized),
        "task_count": len(tasks),
        "repetitions": repetitions,
        "bootstrap_seed": seed,
        "grouping_unit": "task_id",
        "intervals": intervals,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in args.traces.read_text(encoding="utf-8").splitlines() if line
    ]
    result = single_seed_bootstrap(rows, repetitions=args.repetitions, seed=args.seed)
    result["source"] = {"path": str(args.traces.resolve()), "sha256": _sha256(args.traces)}
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
