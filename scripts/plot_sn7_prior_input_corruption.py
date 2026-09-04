#!/usr/bin/env python3
"""Plot the frozen SN7 prior-input corruption robustness curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    severities = payload["severities"]
    x = [item["max_pixels"] for item in severities]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(9.2, 2.7), constrained_layout=True)

    quality = [item["aggregate"]["committed_map_iou"]["mean"] for item in severities]
    quality_std = [
        item["aggregate"]["committed_map_iou"]["std"] for item in severities
    ]
    axes[0].errorbar(
        x,
        quality,
        yerr=quality_std,
        color="#16697A",
        marker="o",
        capsize=3,
        linewidth=1.8,
    )
    axes[0].set_title("(a) Executable map quality")
    axes[0].set_ylabel("Committed raster IoU")

    operation = [
        item["aggregate"]["operation_accuracy"]["mean"] for item in severities
    ]
    axes[1].plot(
        x,
        operation,
        color="#7A5195",
        marker="s",
        linewidth=1.8,
    )
    axes[1].set_title("(b) Operation decision")
    axes[1].set_ylabel("Operation accuracy")

    false_edit = [
        item["aggregate"]["false_edit_rate"]["mean"] for item in severities
    ]
    missed_edit = [
        item["aggregate"]["missed_edit_rate"]["mean"] for item in severities
    ]
    axes[2].plot(
        x,
        false_edit,
        color="#D1495B",
        marker="^",
        linewidth=1.8,
        label="False edit",
    )
    axes[2].plot(
        x,
        missed_edit,
        color="#2A9D8F",
        marker="D",
        linewidth=1.8,
        label="Missed edit",
    )
    axes[2].set_title("(c) Safety tradeoff")
    axes[2].set_ylabel("Conditional error rate")
    axes[2].legend(frameon=False, loc="best")

    for axis in axes:
        axis.set_xlabel("Maximum prior-rendering shift (pixels)")
        axis.set_xticks(x)
        axis.grid(axis="y", color="#D9DEE3", linewidth=0.7)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=300)
    figure.savefig(args.output_prefix.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
