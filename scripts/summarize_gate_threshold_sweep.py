#!/usr/bin/env python3
"""Summarize a gate-threshold sweep and render quality/safety/cost figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.summarize_muno21_grpo_v2 import metrics, read_rows


DISPLAY_METRICS = (
    "macro_f1",
    "false_edit_rate",
    "missed_edit_rate",
    "mean_tool_calls",
    "episode_utility_v2_proxy_balanced_auc",
)
DISPLAY_NAMES = ("Macro-F1", "False edit", "Missed edit", "Tool calls", "Balanced utility")


def parse_run(value: str) -> tuple[float, str, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--run requires THRESHOLD=SEED=JSONL")
    return float(parts[0]), parts[1], Path(parts[2])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    baseline = metrics(read_rows(args.baseline))
    per_run: list[dict[str, Any]] = []
    grouped: dict[float, list[dict[str, float]]] = defaultdict(list)
    for threshold, seed, path in args.run:
        values = metrics(read_rows(path))
        grouped[threshold].append(values)
        per_run.append({"threshold": threshold, "seed": seed, **values})

    aggregate: list[dict[str, Any]] = []
    for threshold in sorted(grouped):
        runs = grouped[threshold]
        row: dict[str, Any] = {"threshold": threshold, "seeds": len(runs)}
        for metric in DISPLAY_METRICS:
            values = np.asarray([run[metric] for run in runs], dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregate.append(row)

    write_csv(args.output / "per_run.csv", per_run)
    write_csv(args.output / "aggregate.csv", aggregate)
    payload = {
        "schema_version": "activemap-gate-threshold-sweep-v1",
        "protocol": {
            "split": "val",
            "test_assets_read": False,
            "selection": "threshold 0.09 frozen before this sweep",
            "diagnostic_thresholds": [row["threshold"] for row in aggregate],
            "aggregation": "mean and sample standard deviation across policy seeds",
        },
        "baseline_sft_gate009": baseline,
        "per_run": per_run,
        "aggregate": aggregate,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "| Gate threshold | Seeds | Macro-F1 | False edit | Missed edit | Tool calls | Balanced utility |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        cells = [f'{row["threshold"]:.2f}', str(row["seeds"])]
        for metric in DISPLAY_METRICS:
            cells.append(f'{row[f"{metric}_mean"]:.4f} +/- {row[f"{metric}_std"]:.4f}')
        lines.append("| " + " | ".join(cells) + " |")
    (args.output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt

    raw = np.asarray(
        [[row[f"{metric}_mean"] for metric in DISPLAY_METRICS] for row in aggregate],
        dtype=float,
    ).T
    normalized = np.zeros_like(raw)
    for index, values in enumerate(raw):
        span = float(values.max() - values.min())
        normalized[index] = 0.0 if span == 0 else (values - values.mean()) / span
    # Lower false edits, missed edits, and tool calls are preferable.
    normalized[1:4] *= -1

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    image = ax.imshow(normalized, cmap="RdBu", vmin=-0.65, vmax=0.65, aspect="auto")
    ax.set_xticks(range(len(aggregate)), [f'{row["threshold"]:.2f}' for row in aggregate])
    ax.set_yticks(range(len(DISPLAY_NAMES)), DISPLAY_NAMES)
    ax.set_xlabel("Reliability-gate threshold")
    ax.set_title("Gate operating boundary (annotations show absolute validation metrics)")
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            ax.text(j, i, f"{raw[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="within-metric direction-corrected score")
    fig.tight_layout()
    fig.savefig(args.output / "gate_threshold_heatmap.png", dpi=240)
    fig.savefig(args.output / "gate_threshold_heatmap.pdf")
    plt.close(fig)

    tools = raw[3]
    utility = raw[4]
    safety = raw[1]
    point_groups: dict[tuple[float, float, float], list[float]] = defaultdict(list)
    for row, x, y, false_edit in zip(aggregate, tools, utility, safety):
        point_groups[(round(float(x), 10), round(float(y), 10), round(float(false_edit), 10))].append(
            float(row["threshold"])
        )
    point_items = list(point_groups.items())
    plot_x = np.asarray([point[0][0] for point in point_items])
    plot_y = np.asarray([point[0][1] for point in point_items])
    plot_safety = np.asarray([point[0][2] for point in point_items])
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    points = ax.scatter(plot_x, plot_y, c=plot_safety, cmap="viridis_r", s=100, edgecolor="black")
    for (x, y, _), thresholds in point_items:
        if len(thresholds) == 1:
            label = f"{thresholds[0]:.2f}"
        else:
            label = f"{min(thresholds):.2f}-{max(thresholds):.2f}"
        threshold = min(thresholds)
        if 0.105 <= threshold <= 0.12:
            offset = (8, 19)
        elif 0.095 <= threshold < 0.105:
            offset = (8, -19)
        elif 0.085 <= threshold < 0.095:
            offset = (8, 3)
        else:
            offset = (5, 5)
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points")
    ax.set_xlabel("Mean tool calls per episode")
    ax.set_ylabel("Balanced utility AUC")
    ax.set_title("Quality-safety-cost operating frontier")
    ax.grid(alpha=0.25)
    fig.colorbar(points, ax=ax, label="False-edit rate (lower is safer)")
    fig.tight_layout()
    fig.savefig(args.output / "gate_quality_safety_cost_frontier.png", dpi=240)
    fig.savefig(args.output / "gate_quality_safety_cost_frontier.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
