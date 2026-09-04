#!/usr/bin/env python3
"""Render paper-ready Safe Commit confidence-interval figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

METRICS = (
    ("terminal_accuracy", "Terminal accuracy", 1.0),
    ("false_edit_rate", "False-edit safety", -1.0),
    ("missed_edit_rate", "Missed-edit safety", -1.0),
    ("balanced_utility", "Balanced utility", 1.0),
    ("safety_utility", "Safety utility", 1.0),
    ("cost_aware_utility", "Cost-aware utility", 1.0),
    ("mean_cost", "Acquisition cost", -1.0),
)
METHODS = (
    ("Strict stump", "#d97706"),
    ("Logistic risk", "#087f8c"),
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _oriented(interval: dict[str, float], direction: float) -> tuple[float, float, float]:
    value = direction * float(interval["observed_delta"])
    bounds = sorted(
        [direction * float(interval["ci95_low"]), direction * float(interval["ci95_high"])]
    )
    return value * 100.0, bounds[0] * 100.0, bounds[1] * 100.0


def _forest(summaries: list[dict[str, Any]], comparator: str, output: Path) -> list[dict[str, Any]]:
    key = "safe_minus_candidate" if comparator == "Hybrid 8K" else "safe_minus_reference"
    y = np.arange(len(METRICS), dtype=np.float64)
    offsets = (-0.13, 0.13)
    records: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for method_index, ((method, color), summary) in enumerate(zip(METHODS, summaries, strict=True)):
        values, low_errors, high_errors = [], [], []
        for metric, label, direction in METRICS:
            value, low, high = _oriented(summary[key][metric], direction)
            values.append(value)
            low_errors.append(value - low)
            high_errors.append(high - value)
            records.append(
                {
                    "method": method,
                    "comparator": comparator,
                    "metric": metric,
                    "display_metric": label,
                    "oriented_delta_pp": value,
                    "ci95_low_pp": low,
                    "ci95_high_pp": high,
                }
            )
        ax.errorbar(
            values,
            y + offsets[method_index],
            xerr=np.asarray([low_errors, high_errors]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            linewidth=1.5,
            markersize=5,
            label=method,
        )
    ax.axvline(0.0, color="#4b5563", linewidth=1.0, linestyle="--")
    ax.set_yticks(y, [label for _, label, _ in METRICS])
    ax.invert_yaxis()
    ax.set_xlabel("Oriented paired improvement (percentage points)")
    ax.set_title(f"Terminal-only Safe Commit vs {comparator}")
    ax.grid(axis="x", color="#d1d5db", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    return records


def _tradeoff(summaries: list[dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.7), constrained_layout=True)
    for (method, color), summary in zip(METHODS, summaries, strict=True):
        false = summary["safe_minus_candidate"]["false_edit_rate"]
        missed = summary["safe_minus_candidate"]["missed_edit_rate"]
        x = -float(false["observed_delta"]) * 100.0
        y = float(missed["observed_delta"]) * 100.0
        x_bounds = sorted([-float(false["ci95_low"]) * 100.0, -float(false["ci95_high"]) * 100.0])
        y_bounds = sorted([float(missed["ci95_low"]) * 100.0, float(missed["ci95_high"]) * 100.0])
        ax.errorbar(
            x,
            y,
            xerr=[[x - x_bounds[0]], [x_bounds[1] - x]],
            yerr=[[y - y_bounds[0]], [y_bounds[1] - y]],
            fmt="o",
            color=color,
            capsize=4,
            markersize=7,
            label=method,
        )
    ax.axhline(0.0, color="#4b5563", linewidth=1.0, linestyle="--")
    ax.set_xlabel("False-edit reduction (percentage points)")
    ax.set_ylabel("Missed-edit increase (percentage points)")
    ax.set_title("Safe Commit safety tradeoff vs Hybrid 8K")
    ax.grid(color="#d1d5db", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stump_summary", type=Path)
    parser.add_argument("logistic_summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [_read(args.stump_summary), _read(args.logistic_summary)]
    records = []
    records.extend(
        _forest(summaries, "Hybrid 8K", args.output_dir / "safe_commit_vs_hybrid_forest.png")
    )
    records.extend(
        _forest(summaries, "Old VLA", args.output_dir / "safe_commit_vs_old_vla_forest.png")
    )
    _tradeoff(summaries, args.output_dir / "safe_commit_safety_tradeoff.png")
    with (args.output_dir / "safe_commit_oriented_deltas.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
