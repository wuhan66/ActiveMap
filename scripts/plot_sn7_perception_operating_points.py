#!/usr/bin/env python3
"""Plot final validation operating points for the modern SN7 backends.

This intentionally replaces an incomparable epoch curve: each backend used
independent validation-best early stopping, so end-point map performance is the
fair comparison.  Values are copied from the locked modern-baseline receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
    }
)


PRIOR = 0.64201
METHODS = {
    "ChangeMamba": {"map_iou": 0.81152, "gain": 0.16951, "change_iou": 0.80616, "false_edit": 0.05958, "color": "#0072B2"},
    "Open-CD BAN": {"map_iou": 0.58841, "gain": -0.05360, "change_iou": 0.46957, "false_edit": 0.52662, "color": "#D55E00"},
}


def save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(output.with_suffix(suffix), bbox_inches="tight", dpi=600)


def plot(output: Path) -> dict[str, object]:
    names = list(METHODS)
    colors = [METHODS[name]["color"] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.6), constrained_layout=True)

    # Panel a: final map state, where the editable prior is an explicit baseline.
    ax = axes[0]
    labels = ["Editable\nprior", *names]
    values = [PRIOR, *(METHODS[name]["map_iou"] for name in names)]
    bars = ax.bar(np.arange(3), values, color=["#A7B0B8", *colors], width=0.62)
    ax.set_ylim(0.45, 0.88)
    ax.set_ylabel("Committed map IoU")
    ax.set_xticks(np.arange(3), labels)
    ax.set_title("a  Final editable-map quality", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#D9DEE2", linewidth=0.6)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(1, 0.468, "+0.170", ha="center", va="bottom", fontsize=8, color=METHODS["ChangeMamba"]["color"], fontweight="bold")
    ax.text(2, 0.468, "-0.054", ha="center", va="bottom", fontsize=8, color=METHODS["Open-CD BAN"]["color"], fontweight="bold")

    # Panel b: quality/safety operating points, avoiding a false training-speed claim.
    ax = axes[1]
    for name in names:
        row = METHODS[name]
        ax.scatter(row["false_edit"], row["gain"], s=100, color=row["color"], edgecolor="white", linewidth=0.8, zorder=3)
        offset = (7, 7) if name == "ChangeMamba" else (-78, -4)
        ax.annotate(name, (row["false_edit"], row["gain"]), xytext=offset, textcoords="offset points", fontsize=8, color=row["color"], fontweight="bold")
    ax.axhline(0, color="#737C85", linewidth=0.8, linestyle="--")
    ax.set_xlim(-0.02, 0.59)
    ax.set_ylim(-0.085, 0.205)
    ax.set_xlabel("False-edit rate (lower is safer)")
    ax.set_ylabel("Map-IoU gain over prior")
    ax.set_title("b  Quality-safety operating point", loc="left", fontweight="bold")
    ax.grid(color="#D9DEE2", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.annotate(
        "safer + higher quality",
        xy=(0.05958, 0.16951),
        xytext=(0.235, 0.196),
        arrowprops={"arrowstyle": "->", "color": "#59636B", "lw": 0.8},
        fontsize=7,
        color="#59636B",
    )

    save(fig, output)
    plt.close(fig)
    return {
        "schema_version": "sn7-modern-perception-operating-points-v1",
        "split": "val",
        "sample_count": 6199,
        "aoi_count": 9,
        "seeds_per_backend": 3,
        "independent_early_stopping": True,
        "values": METHODS,
        "editable_prior_map_iou": PRIOR,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="Output stem, without extension.")
    parser.add_argument("--receipt", type=Path, help="Optional JSON receipt destination.")
    args = parser.parse_args()
    payload = plot(args.output)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
