#!/usr/bin/env python3
"""Plot the calibration-selected MUNO21 quality-safety frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.aggregate.read_text(encoding="utf-8"))
    policies = sorted(
        payload["policies"].values(), key=lambda row: row["false_edit_cap"]
    )
    seeds = payload["seeds"]
    colors = ("#177E89", "#D1495B", "#4C956C")

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for seed_index, (seed, color) in enumerate(zip(seeds, colors, strict=True)):
        false_edit = [
            policy["per_seed"][seed_index]["target_validation"]["metrics"][
                "false_edit_rate"
            ]
            for policy in policies
        ]
        macro_f1 = [
            policy["per_seed"][seed_index]["target_validation"]["metrics"][
                "macro_f1"
            ]
            for policy in policies
        ]
        axis.plot(false_edit, macro_f1, marker="o", color=color, alpha=0.75, label=str(seed))

    for policy in policies:
        target = policy["target_validation"]
        x = target["false_edit_rate"]
        y = target["macro_f1"]
        axis.errorbar(
            x["mean"],
            y["mean"],
            xerr=x["sample_std"],
            yerr=y["sample_std"],
            fmt="s",
            markersize=7,
            color="#1B1B1E",
            capsize=4,
            linewidth=1.4,
        )
        axis.annotate(
            f"calibration cap {policy['false_edit_cap']:.2f}",
            (x["mean"], y["mean"]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
        )

    axis.axvline(0.10, color="#777777", linestyle="--", linewidth=1, label="safety limit")
    axis.set_xlabel("Target false-edit rate (lower is safer)")
    axis.set_ylabel("Target operation macro-F1 (higher is better)")
    axis.set_title("MUNO21 quality-safety frontier (calibration-selected)")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
