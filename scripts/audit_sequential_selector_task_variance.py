#!/usr/bin/env python3
"""Audit task-level SELECT utility variance without changing the policy.

The report is diagnostic-only. It reads validation traces produced by fixed
policies and describes where realized SELECT utility is concentrated; it never
selects a threshold, a checkpoint, or a new training target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("runs must use LABEL=EVALUATION_DIRECTORY")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not label or not (path / "summary.json").is_file() or not (path / "traces.jsonl").is_file():
        raise argparse.ArgumentTypeError(f"invalid evaluation run: {value}")
    return label, path


def _load_traces(directory: Path) -> list[dict[str, Any]]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary.get("test_assets_read") is not False:
        raise ValueError(f"test access in {directory}")
    rows = [
        json.loads(line)
        for line in (directory / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != int(summary["sample_count"]):
        raise ValueError(f"trace count mismatch in {directory}")
    if not all(row.get("split") == "val" for row in rows):
        raise ValueError(f"non-validation trace in {directory}")
    return rows


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_task_variance(
    traces_by_seed: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a seed-by-task diagnosis on identical frozen validation traces."""

    if len(traces_by_seed) < 2:
        raise ValueError("task variance audit requires multiple model seeds")
    labels = sorted(traces_by_seed)
    reference = traces_by_seed[labels[0]]
    reference_keys = [(str(row["task_id"]), str(row["trajectory_id"])) for row in reference]
    reference_targets = {
        key: (str(row["target_selection"]), float(row["policy_relative_advantage"]))
        for key, row in zip(reference_keys, reference, strict=True)
    }
    if len(reference_targets) != len(reference):
        raise ValueError("duplicate validation trajectory")
    for label in labels[1:]:
        rows = traces_by_seed[label]
        keys = [(str(row["task_id"]), str(row["trajectory_id"])) for row in rows]
        targets = {
            key: (str(row["target_selection"]), float(row["policy_relative_advantage"]))
            for key, row in zip(keys, rows, strict=True)
        }
        if keys != reference_keys or targets != reference_targets:
            raise ValueError("runs do not share identical validation states")

    by_seed_task: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for label, rows in traces_by_seed.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["task_id"])].append(row)
        by_seed_task[label] = grouped
    task_ids = sorted(by_seed_task[labels[0]])
    if any(sorted(by_seed_task[label]) != task_ids for label in labels[1:]):
        raise ValueError("runs do not share identical task support")

    task_rows: list[dict[str, Any]] = []
    state_disagreement_count = 0
    for task_id in task_ids:
        per_seed_rows = {label: by_seed_task[label][task_id] for label in labels}
        expected_trajectories = [str(row["trajectory_id"]) for row in per_seed_rows[labels[0]]]
        if any(
            [str(row["trajectory_id"]) for row in per_seed_rows[label]]
            != expected_trajectories
            for label in labels[1:]
        ):
            raise ValueError(f"trajectory order differs in task {task_id}")
        per_seed_utility = {
            label: sum(
                float(row["policy_relative_advantage"])
                for row in rows
                if row["predicted_selection"] == "ACQUIRE"
            )
            for label, rows in per_seed_rows.items()
        }
        per_seed_calls = {
            label: sum(row["predicted_selection"] == "ACQUIRE" for row in rows)
            for label, rows in per_seed_rows.items()
        }
        mean_task_utility = statistics.fmean(per_seed_utility.values())
        oracle_utility = sum(
            max(float(row["policy_relative_advantage"]), 0.0)
            for row in per_seed_rows[labels[0]]
        )
        false_calls = {
            label: sum(
                row["predicted_selection"] == "ACQUIRE"
                and row["target_selection"] != "ACQUIRE"
                for row in rows
            )
            for label, rows in per_seed_rows.items()
        }
        missed_calls = {
            label: sum(
                row["predicted_selection"] != "ACQUIRE"
                and row["target_selection"] == "ACQUIRE"
                for row in rows
            )
            for label, rows in per_seed_rows.items()
        }
        disagreement = sum(
            len(
                {
                    per_seed_rows[label][index]["predicted_selection"]
                    for label in labels
                }
            )
            > 1
            for index in range(len(expected_trajectories))
        )
        state_disagreement_count += disagreement
        task_rows.append(
            {
                "task_id": task_id,
                "state_count": len(expected_trajectories),
                "oracle_utility_sum": oracle_utility,
                "mean_realized_utility_sum": mean_task_utility,
                "mean_utility_gap_to_oracle": oracle_utility - mean_task_utility,
                "per_seed_utility_sum": per_seed_utility,
                "per_seed_calls": per_seed_calls,
                "per_seed_false_calls": false_calls,
                "per_seed_missed_calls": missed_calls,
                "state_prediction_disagreements": disagreement,
            }
        )

    means = [float(row["mean_realized_utility_sum"]) for row in task_rows]
    signs = Counter(
        "positive" if value > 0.0 else "negative" if value < 0.0 else "zero"
        for value in means
    )
    report = {
        "model_seed_count": len(labels),
        "seed_labels": labels,
        "task_count": len(task_rows),
        "state_count_per_seed": len(reference),
        "task_utility_distribution": {
            "mean": statistics.fmean(means),
            "sample_sd": statistics.stdev(means) if len(means) > 1 else 0.0,
            "q05": _quantile(means, 0.05),
            "q25": _quantile(means, 0.25),
            "median": _quantile(means, 0.50),
            "q75": _quantile(means, 0.75),
            "q95": _quantile(means, 0.95),
            "sign_counts": dict(sorted(signs.items())),
        },
        "mean_state_prediction_disagreements_per_task": state_disagreement_count
        / len(task_rows),
        "total_state_prediction_disagreements": state_disagreement_count,
        "most_negative_tasks": sorted(
            task_rows, key=lambda row: (row["mean_realized_utility_sum"], row["task_id"])
        )[:10],
        "largest_oracle_gaps": sorted(
            task_rows,
            key=lambda row: (-row["mean_utility_gap_to_oracle"], row["task_id"]),
        )[:10],
    }
    return report, task_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if len(args.run) < 2:
        raise ValueError("provide at least two seed evaluations")

    labels = [label for label, _ in args.run]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate seed label")
    traces_by_seed = {label: _load_traces(directory) for label, directory in args.run}
    report, task_rows = summarize_task_variance(traces_by_seed)
    args.output.mkdir(parents=True)
    task_path = args.output / "per_task.jsonl"
    with task_path.open("w", encoding="utf-8") as handle:
        for row in task_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    result = {
        "schema_version": "sequential-selector-task-variance-audit-v1",
        "audit_role": "diagnostic-only-no-threshold-or-checkpoint-selection",
        **report,
        "sources": {
            label: {
                "summary_sha256": _sha256(directory / "summary.json"),
                "traces_sha256": _sha256(directory / "traces.jsonl"),
            }
            for label, directory in args.run
        },
        "per_task_sha256": _sha256(task_path),
        "test_assets_read": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# SELECT Task-Variance Audit",
        "",
        "Diagnostic-only: this report does not select a threshold, checkpoint, or target.",
        "",
        "| Tasks | Mean task utility | Task utility SD | Negative / zero / positive tasks |",
        "| ---: | ---: | ---: | ---: |",
        (
            f"| {report['task_count']} | {report['task_utility_distribution']['mean']:.6f} "
            f"| {report['task_utility_distribution']['sample_sd']:.6f} | "
            f"{report['task_utility_distribution']['sign_counts'].get('negative', 0)} / "
            f"{report['task_utility_distribution']['sign_counts'].get('zero', 0)} / "
            f"{report['task_utility_distribution']['sign_counts'].get('positive', 0)} |"
        ),
        "",
        "## Most Negative Tasks",
        "",
        "| Task | Mean utility | Oracle utility | Gap | State disagreements |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["most_negative_tasks"]:
        lines.append(
            f"| {row['task_id']} | {row['mean_realized_utility_sum']:.6f} | "
            f"{row['oracle_utility_sum']:.6f} | {row['mean_utility_gap_to_oracle']:.6f} "
            f"| {row['state_prediction_disagreements']} |"
        )
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
