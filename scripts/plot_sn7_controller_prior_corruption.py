#!/usr/bin/env python3
"""Plot controller quality-cost-safety under prior-input corruption."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

VARIANTS = (
    ("notool", "No tool", "#4C78A8"),
    ("forced", "Forced tool", "#E45756"),
    ("benefit", "Selective benefit", "#2A9D8F"),
)
PANELS = (
    ("writeback", "raster_iou_auc", "Executable raster IoU AUC", "higher is better"),
    ("writeback", "false_edit_auc", "False-edit AUC", "lower is better"),
    ("writeback", "missed_edit_auc", "Missed-edit AUC", "lower is better"),
    (
        "writeback",
        "spent_cost_auc",
        "Evidence cost AUC",
        "lower at matched quality",
    ),
    (
        "writeback",
        "episode_utility_v2_balanced_auc",
        "Balanced executable utility AUC",
        "higher is better",
    ),
    (
        "controller",
        "tool_call_episode_rate",
        "Tool-call episode rate",
        "usage (compare at matched quality)",
    ),
)


def plot(payload: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    severities = [int(row["severity"]) for row in payload["rows"]]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), constrained_layout=True)
    for axis, (source, metric, title, direction) in zip(
        axes.ravel(),
        PANELS,
        strict=True,
    ):
        for variant, label, color in VARIANTS:
            values = [
                float(row[source][variant][metric])
                for row in payload["rows"]
            ]
            axis.plot(
                severities,
                values,
                marker="o",
                linewidth=2.1,
                markersize=5.5,
                color=color,
                label=label,
            )
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Maximum prior-input translation (pixels)")
        axis.set_xticks(severities)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            0.02,
            0.03,
            direction,
            transform=axis.transAxes,
            fontsize=8,
            color="#555555",
        )
    axes[0, 0].legend(frameon=False, ncol=1, loc="center right", fontsize=8)
    fig.suptitle(
        "Frozen controller robustness to editable-prior misregistration",
        fontsize=14,
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output_dir / f"controller_prior_corruption.{suffix}",
            dpi=240 if suffix == "png" else None,
        )
    plt.close(fig)
    (output_dir / "plot_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "sn7-controller-prior-corruption-plot-v1",
                "source_schema": payload["schema_version"],
                "severities": severities,
                "variants": [item[0] for item in VARIANTS],
                "panels": [item[0] for item in PANELS],
                "test_assets_read": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    plot(json.loads(args.summary.read_text(encoding="utf-8")), args.output_dir)


if __name__ == "__main__":
    main()
