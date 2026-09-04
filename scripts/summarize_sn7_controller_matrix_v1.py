#!/usr/bin/env python3
"""Summarize the frozen SN7 controller matrix and paired AOI bootstrap deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from scripts.compare_agent_writebacks import _load, _metrics, compare


DEFAULT_ROOT = Path("/home/wh/ActiveMap/runs")


def paths(root: Path) -> dict[str, list[Path]]:
    heuristic = root / "baselines/sn7_terminal_evidence_heuristics_v1/writebacks"
    learned = root / "selector/sn7_value_aware_h18_closed_loop_writeback_v1/writebacks"
    gated = root / "selector/sn7_value_aware_h18_add_commit_gate_writeback_v1/writebacks"
    return {
        "Always STOP": [gated / "always_stop/evaluation/writeback.jsonl"],
        "Cheapest": [heuristic / "cheapest/evaluation/writeback.jsonl"],
        "Clear-per-cost": [heuristic / "clear_per_cost/evaluation/writeback.jsonl"],
        "Uncertainty gate": [heuristic / "uncertainty_gate/evaluation/writeback.jsonl"],
        "Random": [heuristic / "random/evaluation/writeback.jsonl"],
        "Generic learned h18": [
            learned / f"generic_value_h18_s{seed}/evaluation/writeback.jsonl"
            for seed in (1, 2, 3)
        ],
        "ActiveMap ADD gate": [
            gated / f"add_gate_h18_s{seed}_v2/evaluation/writeback.jsonl"
            for seed in (1, 2, 3)
        ],
    }


METRICS = {
    "raster_iou_gain_auc": "IoU gain AUC",
    "episode_utility_v2_balanced_auc": "Balanced utility",
    "episode_utility_v2_safety_auc": "Safety utility",
    "episode_utility_v2_cost_aware_auc": "Cost-aware utility",
    "false_edit_auc": "False edit",
    "missed_edit_auc": "Missed edit",
    "wrong_edit_auc": "Wrong edit",
    "spent_cost_auc": "Cost",
    "component_count_absolute_error_auc": "Component error",
}


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "runs": len(values),
    }


def format_cell(row: dict[str, float]) -> str:
    if int(row["runs"]) == 1:
        return f'{row["mean"]:.4f}'
    return f'{row["mean"]:.4f} +/- {row["std"]:.4f}'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    matrix = paths(args.run_root)
    missing = [str(path) for runs in matrix.values() for path in runs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing controller results: {missing}")

    raw: dict[str, list[dict[str, float]]] = {
        method: [_metrics(list(_load(path).values())) for path in runs]
        for method, runs in matrix.items()
    }
    table = {
        method: {
            metric: summarize([run[metric] for run in runs])
            for metric in METRICS
        }
        for method, runs in raw.items()
    }

    active_paths = matrix["ActiveMap ADD gate"]
    comparisons: dict[str, list[dict[str, Any]]] = {}
    for baseline_name in ("Always STOP", "Uncertainty gate", "Generic learned h18"):
        baseline_paths = matrix[baseline_name]
        pairs = zip(
            baseline_paths if len(baseline_paths) > 1 else baseline_paths * len(active_paths),
            active_paths,
            strict=True,
        )
        comparisons[baseline_name] = [
            compare(
                baseline,
                candidate,
                bootstrap=args.bootstrap,
                seed=args.seed + index,
                group_key="aoi_id",
            )
            for index, (baseline, candidate) in enumerate(pairs)
        ]

    payload = {
        "schema_version": "sn7-controller-matrix-v1",
        "split": "val",
        "test_assets_read": False,
        "methods": {name: [str(path) for path in runs] for name, runs in matrix.items()},
        "metrics": table,
        "active_map_paired_comparisons": comparisons,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "controller_matrix.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    headers = ["Method", *METRICS.values()]
    lines = [
        "# SN7 Controller Matrix",
        "",
        "Validation-only results. Values are budget AUC; multi-seed rows are mean +/- sample std.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(METRICS)) + " |",
    ]
    for method, metrics in table.items():
        lines.append(
            "| " + " | ".join([method, *[format_cell(metrics[key]) for key in METRICS]]) + " |"
        )
    lines.extend(["", "Test assets were not read.", ""])
    (args.output / "controller_matrix.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
