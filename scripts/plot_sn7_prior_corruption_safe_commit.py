#!/usr/bin/env python3
"""Plot frozen Safe Commit under SN7 prior-input corruption."""

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
    rows = payload["severities"]
    x = [row["max_pixels"] for row in rows]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(9.2, 2.7), constrained_layout=True)
    colors = {"direct": "#D1495B", "safe": "#16697A"}
    labels = {"direct": "Direct commit", "safe": "Frozen Safe Commit"}

    for policy in ("direct", "safe"):
        axes[0].errorbar(
            x,
            [
                row["aggregate"][policy]["committed_map_iou"]["mean"]
                for row in rows
            ],
            yerr=[
                row["aggregate"][policy]["committed_map_iou"]["std"]
                for row in rows
            ],
            color=colors[policy],
            marker="o" if policy == "direct" else "s",
            capsize=3,
            linewidth=1.8,
            label=labels[policy],
        )
        axes[1].plot(
            x,
            [
                row["aggregate"][policy]["false_edit_rate"]["mean"]
                for row in rows
            ],
            color=colors[policy],
            marker="o" if policy == "direct" else "s",
            linewidth=1.8,
            label=labels[policy],
        )
        axes[2].plot(
            x,
            [
                row["aggregate"][policy]["missed_edit_rate"]["mean"]
                for row in rows
            ],
            color=colors[policy],
            marker="o" if policy == "direct" else "s",
            linewidth=1.8,
            label=labels[policy],
        )

    axes[0].set_title("(a) Executable map quality")
    axes[0].set_ylabel("Committed raster IoU")
    axes[1].set_title("(b) False-edit safety")
    axes[1].set_ylabel("False-edit rate")
    axes[2].set_title("(c) Conservative tradeoff")
    axes[2].set_ylabel("Missed-edit rate")
    axes[0].legend(frameon=False, loc="best")
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
