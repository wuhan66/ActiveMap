#!/usr/bin/env python3
"""Plot structured-action safety trajectories across SFT checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")


def _load_run(run_dir: Path) -> tuple[int, list[dict[str, Any]], str | None]:
    seed_match = re.search(r"seed(\d+)$", run_dir.name)
    if seed_match is None:
        raise ValueError(f"run directory has no seed suffix: {run_dir}")
    seed = int(seed_match.group(1))
    rows = []
    for summary in sorted(
        (run_dir / "evaluation").glob("checkpoint-*/actions/summary.json"),
        key=lambda path: int(path.parents[1].name.split("-")[-1]),
    ):
        match = CHECKPOINT.match(summary.parents[1].name)
        if match is None:
            continue
        data = json.loads(summary.read_text(encoding="utf-8"))
        if data.get("test_assets_read") is not False:
            raise ValueError(f"checkpoint summary is not validation-only: {summary}")
        tool = data["tool_metrics"]
        rows.append(
            {
                "seed": seed,
                "checkpoint": summary.parents[1].name,
                "step": int(match.group(1)),
                "sample_count": int(data["sample_count"]),
                "false_call_rate": float(tool["false_call_rate"]),
                "grounded_call_recall": float(tool["grounded_call_accuracy"]),
                "tool_positive_exact": float(
                    tool["tool_positive_exact_accuracy"]
                ),
                "macro_f1": float(data["macro_f1"]),
                "exact_action_accuracy": float(data["exact_action_accuracy"]),
            }
        )
    if not rows:
        raise ValueError(f"no checkpoint action summaries in {run_dir}")
    decision_path = run_dir / "evaluation/selection/static_checkpoint_decision.json"
    selected = None
    if decision_path.is_file():
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("protocol", {}).get("test_assets_read") is not False:
            raise ValueError(f"selection is not validation-only: {decision_path}")
        selected = decision.get("selected_checkpoint")
    return seed, rows, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--max-false-call-rate", type=float, default=0.02)
    args = parser.parse_args()
    if len(args.run_dirs) < 2:
        raise ValueError("at least two model-seed runs are required")

    all_rows = []
    selected_by_seed = {}
    for run_dir in args.run_dirs:
        seed, rows, selected = _load_run(run_dir)
        if seed in selected_by_seed:
            raise ValueError(f"duplicate seed {seed}")
        selected_by_seed[seed] = selected
        all_rows.extend(rows)
    sample_counts = {row["sample_count"] for row in all_rows}
    if len(sample_counts) != 1:
        raise ValueError(f"checkpoint summaries use different supports: {sample_counts}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = tuple(all_rows[0])
    with (args.output_dir / "checkpoint_safety_trajectory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    metrics = (
        ("false_call_rate", "False-call rate"),
        ("grounded_call_recall", "Grounded tool recall"),
        ("macro_f1", "Action Macro-F1"),
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(12.2, 3.5), constrained_layout=True
    )
    palette = ("#126E82", "#D95F43", "#408A5C", "#6A5AA3", "#B17A19")
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        for color, seed in zip(palette, sorted(selected_by_seed), strict=False):
            rows = sorted(
                (row for row in all_rows if row["seed"] == seed),
                key=lambda row: row["step"],
            )
            axis.plot(
                [row["step"] for row in rows],
                [row[metric] for row in rows],
                marker="o",
                markersize=4,
                linewidth=1.6,
                color=color,
                label=str(seed),
            )
            selected = selected_by_seed[seed]
            selected_row = next(
                (row for row in rows if row["checkpoint"] == selected), None
            )
            if selected_row is not None:
                axis.scatter(
                    [selected_row["step"]],
                    [selected_row[metric]],
                    s=62,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.6,
                    zorder=4,
                )
        if metric == "false_call_rate":
            axis.axhline(
                args.max_false_call_rate,
                color="#B53A3A",
                linestyle="--",
                linewidth=1.2,
                label="Safety gate",
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Checkpoint step", fontsize=9)
        axis.grid(color="#E2E6E9", linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Validation metric", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=min(6, len(labels)),
        frameon=False,
        fontsize=8,
    )
    for suffix, kwargs in (
        (".png", {"dpi": 240}),
        (".pdf", {}),
    ):
        fig.savefig(
            args.output_dir / f"checkpoint_safety_trajectory{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)

    manifest = {
        "schema_version": "muno21-checkpoint-safety-trajectory-v1",
        "model_seeds": sorted(selected_by_seed),
        "selected_checkpoints": {
            str(seed): selected for seed, selected in sorted(selected_by_seed.items())
        },
        "sample_count_per_checkpoint": sample_counts.pop(),
        "max_false_call_rate": args.max_false_call_rate,
        "record_count": len(all_rows),
        "split": "val",
        "test_assets_read": False,
    }
    (args.output_dir / "checkpoint_safety_trajectory.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
