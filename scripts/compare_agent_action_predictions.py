#!/usr/bin/env python3
"""Grouped paired bootstrap comparison for two Agent action prediction files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        key = (str(row["trajectory_id"]), int(row["step"]))
        if key in rows:
            raise ValueError(f"duplicate prediction key in {path}: {key}")
        rows[key] = row
    return rows


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    confusion: Counter[tuple[str, str]] = Counter()
    exact = 0
    regrets = []
    for row in rows:
        target = str(row["target_class"])
        prediction = str(row["prediction_class"])
        confusion[(target, prediction)] += 1
        exact += int(bool(row["executable"]) and row["target"] == row["prediction"])
        if row.get("oracle_utility") is not None and row.get("predicted_utility") is not None:
            regrets.append(float(row["oracle_utility"]) - float(row["predicted_utility"]))
    labels = sorted({target for target, _ in confusion})
    f1_values = []
    recalls = {}
    for label in labels:
        true_positive = confusion[(label, label)]
        false_positive = sum(
            count
            for (target, prediction), count in confusion.items()
            if prediction == label and target != label
        )
        false_negative = sum(
            count
            for (target, prediction), count in confusion.items()
            if target == label and prediction != label
        )
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1_values.append(2 * precision * recall / max(precision + recall, 1e-12))
        recalls[label] = recall
    return {
        "exact_action_accuracy": exact / max(len(rows), 1),
        "macro_f1": sum(f1_values) / max(len(f1_values), 1),
        "acquire_recall": recalls.get("ACQUIRE", 0.0),
        "mean_regret": sum(regrets) / max(len(regrets), 1),
    }


def _group_id(trajectory_id: str) -> str:
    return trajectory_id.split("__b", maxsplit=1)[0]


def compare(
    baseline_path: Path,
    candidate_path: Path,
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    if set(baseline) != set(candidate):
        raise ValueError("prediction files do not contain identical trajectory-step keys")
    ordered_keys = sorted(baseline)
    baseline_rows = [baseline[key] for key in ordered_keys]
    candidate_rows = [candidate[key] for key in ordered_keys]
    baseline_metrics = _metrics(baseline_rows)
    candidate_metrics = _metrics(candidate_rows)
    observed = {
        name: candidate_metrics[name] - baseline_metrics[name]
        for name in baseline_metrics
    }

    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in ordered_keys:
        groups[_group_id(key[0])].append(key)
    group_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in observed}
    for _ in range(bootstrap):
        sampled_groups = rng.choice(group_ids, size=len(group_ids), replace=True)
        sampled_keys = [key for group in sampled_groups for key in groups[str(group)]]
        left = _metrics([baseline[key] for key in sampled_keys])
        right = _metrics([candidate[key] for key in sampled_keys])
        for name in observed:
            samples[name].append(right[name] - left[name])
    intervals = {
        name: {
            "delta": observed[name],
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for name, values in samples.items()
    }
    baseline_correct = {
        key: bool(row["executable"] and row["target"] == row["prediction"])
        for key, row in baseline.items()
    }
    candidate_correct = {
        key: bool(row["executable"] and row["target"] == row["prediction"])
        for key, row in candidate.items()
    }
    transitions = {
        "both_correct": sum(
            baseline_correct[key] and candidate_correct[key] for key in ordered_keys
        ),
        "baseline_only": sum(
            baseline_correct[key] and not candidate_correct[key] for key in ordered_keys
        ),
        "candidate_only": sum(
            not baseline_correct[key] and candidate_correct[key] for key in ordered_keys
        ),
        "both_wrong": sum(
            not baseline_correct[key] and not candidate_correct[key] for key in ordered_keys
        ),
    }
    return {
        "protocol": {
            "paired": True,
            "group": "opaque source task id across budgets and steps",
            "bootstrap": bootstrap,
            "seed": seed,
            "test_assets_read": False,
        },
        "sample_count": len(ordered_keys),
        "group_count": len(group_ids),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "paired_delta": intervals,
        "correctness_transitions": transitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    result = compare(
        args.baseline,
        args.candidate,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
