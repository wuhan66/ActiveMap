#!/usr/bin/env python3
"""Aggregate supervised Direct-VLM seeds into validation-only paper evidence."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


ACTION_METRICS = (
    "accuracy",
    "macro_f1",
)
SAFETY_METRICS = (
    "false_edit_rate",
    "missed_edit_rate",
    "schema_valid_rate",
)
WRITEBACK_METRICS = (
    "mean_raster_iou",
    "mean_raster_iou_gain",
    "mean_vector_replay_iou",
    "vector_delta_topology_valid_rate",
    "mean_episode_utility_v2_balanced",
    "mean_episode_utility_v2_safety",
    "mean_episode_utility_v2_cost_aware",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _budget(summary: dict[str, Any], value: float) -> dict[str, Any]:
    return next(row for row in summary["budgets"] if float(row["budget"]) == value)


def _run(seed: int, root: Path, budget: float) -> tuple[dict[str, Any], set[tuple[str, str]]]:
    complete = _json(root / "COMPLETE.json")
    evaluation = _json(root / "evaluation/summary.json")
    traces = _jsonl(root / "evaluation/traces.jsonl")
    writeback = _budget(
        _json(root / "writeback/safe_delta/summary.json"), budget
    )
    protocol = evaluation["protocol"]
    if complete.get("status") != "complete":
        raise ValueError(f"seed {seed} is not complete")
    if (
        protocol.get("split") != "val"
        or evaluation.get("test_assets_read")
        or complete.get("test_assets_read")
    ):
        raise ValueError(f"seed {seed} is not validation-only")
    if evaluation.get("adapter") is None or protocol.get("image_mode") != "composite":
        raise ValueError(f"seed {seed} is not the supervised composite baseline")
    identity = {
        (str(row["example_id"]), str(row["target_operation"])) for row in traces
    }
    operation = evaluation["operation_metrics"]
    row = {
        "seed": seed,
        "sample_count": evaluation["sample_count"],
        "accuracy": operation["accuracy"],
        "macro_f1": operation["macro_f1"],
        "false_edit_rate": evaluation["false_edit_rate"],
        "missed_edit_rate": evaluation["missed_edit_rate"],
        "schema_valid_rate": evaluation["schema_valid_rate"],
        **{metric: writeback[metric] for metric in WRITEBACK_METRICS},
        "run_root": str(root.resolve()),
        "test_assets_read": False,
    }
    return row, identity


def aggregate(
    runs: dict[int, Path], *, budget: float
) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("at least two supervised Direct-VLM seeds are required")
    per_seed = []
    identities = []
    for seed, root in sorted(runs.items()):
        row, identity = _run(seed, root, budget)
        per_seed.append(row)
        identities.append(identity)
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Direct-VLM seeds do not use identical paired validation support")
    metrics = ACTION_METRICS + SAFETY_METRICS + WRITEBACK_METRICS
    aggregate_metrics = {}
    for metric in metrics:
        values = [float(row[metric]) for row in per_seed]
        aggregate_metrics[metric] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
        }
    return {
        "schema_version": "muno21-supervised-direct-vlm-seed-aggregate-v1",
        "split": "val",
        "budget": budget,
        "seed_count": len(per_seed),
        "paired_example_count": len(identities[0]),
        "per_seed": per_seed,
        "aggregate_metrics": aggregate_metrics,
        "claim_scope": "strong supervised baseline stability; not primary-method promotion",
        "test_assets_read": False,
    }


def _write_table(path: Path, result: dict[str, Any]) -> None:
    columns = (
        "seed",
        "accuracy",
        "macro_f1",
        "false_edit_rate",
        "missed_edit_rate",
        "mean_raster_iou_gain",
        "mean_episode_utility_v2_balanced",
    )
    labels = (
        "Seed",
        "Acc.",
        "Macro-F1",
        "False edit",
        "Missed edit",
        "Raster-IoU gain",
        "Balanced utility",
    )
    lines = [
        "| " + " | ".join(labels) + " |",
        "|" + "|".join("---" for _ in labels) + "|",
    ]
    for row in result["per_seed"]:
        lines.append(
            "| "
            + " | ".join(
                str(row[column])
                if column == "seed"
                else f"{float(row[column]):.4f}"
                for column in columns
            )
            + " |"
        )
    means = result["aggregate_metrics"]
    lines.append(
        "| Mean +/- std | "
        + " | ".join(
            f"{means[column]['mean']:.4f} +/- {means[column]['sample_std']:.4f}"
            for column in columns[1:]
        )
        + " |"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--run", action="append", required=True, metavar="SEED=ROOT")
    parser.add_argument("--budget", type=float, default=3.0)
    args = parser.parse_args()
    runs = {}
    for value in args.run:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in runs:
            raise ValueError(f"duplicate seed: {seed}")
        runs[seed] = Path(path_text)
    result = aggregate(runs, budget=args.budget)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_seed.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["per_seed"][0]))
        writer.writeheader()
        writer.writerows(result["per_seed"])
    _write_table(args.output_dir / "table.md", result)
    print(json.dumps({"seeds": len(runs), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
