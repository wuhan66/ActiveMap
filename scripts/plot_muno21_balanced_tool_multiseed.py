#!/usr/bin/env python3
"""Render direction-corrected multi-seed controller and writeback forests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONTROLLER_METRICS = (
    "episode_utility_v2_balanced_auc",
    "terminal_accuracy",
    "false_edit_rate",
    "mean_cost",
)
WRITEBACK_METRICS = (
    "raster_iou_gain_auc",
    "episode_utility_v2_balanced_auc",
    "false_edit_auc",
    "missed_edit_auc",
)
LABELS = {
    "episode_utility_v2_balanced_auc": "Balanced utility AUC",
    "terminal_accuracy": "Terminal accuracy",
    "false_edit_rate": "False-edit rate",
    "mean_cost": "Evidence cost",
    "raster_iou_gain_auc": "Raster-IoU gain AUC",
    "false_edit_auc": "False-edit AUC",
    "missed_edit_auc": "Missed-edit AUC",
}


def _short_comparison(value: str) -> str:
    return (
        value.replace("qwen3_4b_sft_tool_to_belief", "Tool-to-Belief")
        .replace("qwen3_4b_sft_tools_no_belief", "No recurrent belief")
        .replace("qwen3_4b_sft", "SFT")
        .replace("edit_conditioned_selector", "Edit-conditioned selector")
        .replace("generic_selector", "Generic selector")
        .replace("forced_tools", "Forced tools")
    )


def _plot(
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...],
    output_stem: Path,
) -> None:
    present = [metric for metric in metrics if any(row["metric"] == metric for row in rows)]
    if not present:
        raise ValueError("no requested metrics are present")
    fig, axes = plt.subplots(
        1,
        len(present),
        figsize=(4.1 * len(present), 3.7),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, metric in zip(axes[0], present, strict=True):
        selected = [row for row in rows if row["metric"] == metric]
        labels = [_short_comparison(row["comparison"]) for row in selected]
        signs = [-1.0 if row["direction"] == "lower" else 1.0 for row in selected]
        centers = [sign * float(row["delta"]) for sign, row in zip(signs, selected)]
        lows = [
            sign
            * float(row["ci95_high"] if sign < 0 else row["ci95_low"])
            for sign, row in zip(signs, selected)
        ]
        highs = [
            sign
            * float(row["ci95_low"] if sign < 0 else row["ci95_high"])
            for sign, row in zip(signs, selected)
        ]
        positions = list(range(len(selected)))
        colors = ["#178A68" if row["strictly_better"] else "#53616D" for row in selected]
        for position, center, low, high, color in zip(
            positions, centers, lows, highs, colors, strict=True
        ):
            axis.errorbar(
                center,
                position,
                xerr=[[center - low], [high - center]],
                fmt="o",
                color=color,
                ecolor=color,
                capsize=3,
                markersize=5,
                linewidth=1.5,
            )
        axis.axvline(0, color="#A8AFB5", linewidth=1, linestyle="--")
        axis.set_yticks(positions, labels if axis is axes[0][0] else [])
        axis.invert_yaxis()
        axis.set_title(LABELS.get(metric, metric), fontsize=10)
        axis.set_xlabel("Improvement (95% CI)", fontsize=9)
        axis.grid(axis="x", color="#E1E5E8", linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    data = json.loads(args.table_json.read_text(encoding="utf-8"))
    if data.get("schema_version") != "muno21-balanced-tool-multiseed-table-v1":
        raise ValueError("unexpected table schema")
    if data.get("split") != "val" or data.get("test_assets_read") is not False:
        raise ValueError("plot input must be audited validation evidence")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot(
        data["controller_rows"],
        CONTROLLER_METRICS,
        args.output_dir / "controller_forest",
    )
    _plot(
        data["writeback_rows"],
        WRITEBACK_METRICS,
        args.output_dir / "writeback_forest",
    )


if __name__ == "__main__":
    main()
