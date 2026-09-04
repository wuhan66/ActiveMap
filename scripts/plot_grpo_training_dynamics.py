#!/usr/bin/env python3
"""Render paper-ready training dynamics for a recurrent GRPO matrix."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PALETTE = {
    "constrained": "#087E8B",
    "unconstrained": "#D1495B",
    "lambda025": "#F4A261",
    "lambda100": "#355070",
}


def variant_from_name(name: str) -> str:
    if name.startswith("constrained_"):
        return "constrained"
    if name.startswith("unconstrained_"):
        return "unconstrained"
    if name.startswith("lambda025_"):
        return "lambda025"
    if name.startswith("lambda100_"):
        return "lambda100"
    return name.rsplit("_seed", 1)[0]


def read_history(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty history: {path}")
    required = {"step", "policy_loss", "entropy", "advantage", "sequence_ratio"}
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    return rows


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    result = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = np.median(values[start : index + 1])
    return result


def aligned_statistics(
    runs: list[list[dict[str, Any]]], field: str, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    final_step = min(int(run[-1]["step"]) for run in runs)
    grid = np.arange(1, final_step + 1, dtype=float)
    aligned = []
    for run in runs:
        steps = np.asarray([row["step"] for row in run], dtype=float)
        values = smooth(np.asarray([row[field] for row in run], dtype=float), window)
        aligned.append(np.interp(grid, steps, values))
    matrix = np.stack(aligned)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1) if len(matrix) > 1 else np.zeros_like(mean)
    return grid, mean, std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--clip-low", type=float, default=0.8)
    parser.add_argument("--clip-high", type=float, default=1.28)
    args = parser.parse_args()

    histories: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    raw_rows: list[dict[str, Any]] = []
    for path in sorted(args.matrix_root.glob("*/history.jsonl")):
        run_name = path.parent.name
        variant = variant_from_name(run_name)
        rows = read_history(path)
        histories[variant].append(rows)
        raw_rows.extend({"variant": variant, "run": run_name, **row} for row in rows)
    if not histories:
        raise ValueError(f"no */history.jsonl files under {args.matrix_root}")

    args.output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in raw_rows for key in row})
    with (args.output / "grpo_training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    audit = {
        "matrix_root": str(args.matrix_root),
        "smooth_window": args.smooth_window,
        "variants": {
            variant: {
                "runs": len(runs),
                "steps": [int(run[-1]["step"]) for run in runs],
            }
            for variant, runs in histories.items()
        },
    }
    (args.output / "grpo_training_dynamics.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.7), constrained_layout=True)
    panels = (
        ("policy_loss", "Policy objective", "Smoothed value"),
        ("sequence_ratio", "Sequence probability ratio", "Ratio"),
        ("entropy", "Policy entropy", "Entropy"),
        ("advantage", "Group-relative advantage", "Advantage"),
    )
    display = {
        "constrained": "Constrained (lambda=0.5)",
        "unconstrained": "Unconstrained",
        "lambda025": "Lambda=0.25",
        "lambda100": "Lambda=1.0",
    }
    for axis, (field, title, ylabel) in zip(axes.flat, panels):
        plotted_low: list[float] = []
        plotted_high: list[float] = []
        for variant, runs in histories.items():
            steps, mean, std = aligned_statistics(runs, field, args.smooth_window)
            color = PALETTE.get(variant, "#555555")
            axis.plot(steps, mean, color=color, linewidth=1.7, label=display.get(variant, variant))
            plotted_low.append(float(np.min(mean - std)))
            plotted_high.append(float(np.max(mean + std)))
            if len(runs) > 1:
                axis.fill_between(steps, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
        if field == "sequence_ratio":
            axis.axhline(1.0, color="#444444", linewidth=0.8, linestyle="--")
            low = min(min(plotted_low), 1.0)
            high = max(max(plotted_high), 1.0)
            padding = max((high - low) * 0.16, 0.0005)
            axis.set_ylim(low - padding, high + padding)
            axis.text(
                0.98,
                0.06,
                f"clip limits [{args.clip_low:.2f}, {args.clip_high:.2f}]",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                color="#555555",
                fontsize=8,
            )
        if field == "advantage":
            axis.axhline(0.0, color="#444444", linewidth=0.8, linestyle="--")
        axis.set_title(title)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=min(4, len(labels)),
        frameon=False,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"grpo_training_dynamics.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
