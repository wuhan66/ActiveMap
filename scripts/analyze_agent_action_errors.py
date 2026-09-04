#!/usr/bin/env python3
"""Summarize structured-action errors for a validation checkpoint."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _rate(correct: int, total: int) -> float:
    return correct / max(total, 1)


def analyze(path: Path, *, worst_count: int = 20) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    confusion: Counter[tuple[str, str]] = Counter()
    target_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    by_step: dict[int, Counter[str]] = defaultdict(Counter)
    acquire_false_positive_targets: Counter[str] = Counter()
    utility_by_target: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        target = str(row["target_class"])
        prediction = str(row["prediction_class"])
        exact = bool(row["executable"] and row["target"] == row["prediction"])
        confusion[(target, prediction)] += 1
        target_counts[target] += 1
        prediction_counts[prediction] += 1
        step = int(row["step"])
        by_step[step]["total"] += 1
        by_step[step]["exact"] += int(exact)
        by_step[step]["executable"] += int(row["executable"])
        if prediction == "ACQUIRE" and target != "ACQUIRE":
            acquire_false_positive_targets[target] += 1
        if row.get("predicted_utility") is not None:
            utility_by_target[target].append(float(row["predicted_utility"]))

    labels = sorted(set(target_counts) | set(prediction_counts))
    worst = sorted(
        (row for row in rows if row.get("predicted_utility") is not None),
        key=lambda row: float(row["predicted_utility"]),
    )[:worst_count]
    return {
        "sample_count": len(rows),
        "labels": labels,
        "target_counts": dict(target_counts),
        "prediction_counts": dict(prediction_counts),
        "confusion": {
            target: {prediction: confusion[(target, prediction)] for prediction in labels}
            for target in labels
        },
        "by_step": {
            str(step): {
                "sample_count": counts["total"],
                "exact_accuracy": _rate(counts["exact"], counts["total"]),
                "executable_valid_rate": _rate(counts["executable"], counts["total"]),
            }
            for step, counts in sorted(by_step.items())
        },
        "acquire_false_positive_targets": dict(acquire_false_positive_targets),
        "mean_predicted_utility_by_target": {
            target: sum(values) / len(values) for target, values in utility_by_target.items()
        },
        "worst_utility_examples": [
            {
                "trajectory_id": row["trajectory_id"],
                "step": row["step"],
                "target": row["target"],
                "prediction": row["prediction"],
                "predicted_utility": row["predicted_utility"],
                "oracle_utility": row["oracle_utility"],
                "schema_valid": row["schema_valid"],
                "executable": row["executable"],
            }
            for row in worst
        ],
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worst-count", type=int, default=20)
    args = parser.parse_args()
    result = analyze(args.predictions, worst_count=args.worst_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
