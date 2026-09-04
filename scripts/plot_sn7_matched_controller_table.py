#!/usr/bin/env python3
"""Plot the matched SN7 controller utility and safety evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "direct_sft": "Direct SFT",
    "react": "ReAct",
    "plan_execute": "Plan-Execute",
    "geommagent_style": "GeoMMAgent-style",
    "sensesearch_style": "SenseSearch-style",
    "active_map": "ActiveMap",
}
COLORS = {
    "direct_sft": "#4D4D4D",
    "react": "#3B82C4",
    "plan_execute": "#D97706",
    "geommagent_style": "#8B5FA8",
    "sensesearch_style": "#25856D",
    "active_map": "#D64545",
}


def _load(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("test_assets_read") is not False:
        raise ValueError("controller table is not validation-only")
    return payload["rows"]


def plot_utility(rows: list[dict], output: Path) -> None:
    candidates = [row for row in rows if row["method"] != "direct_sft"]
    values = np.asarray(
        [row["delta_episode_utility_v2_balanced_auc"] for row in candidates]
    )
    low = np.asarray(
        [row["ci_low_episode_utility_v2_balanced_auc"] for row in candidates]
    )
    high = np.asarray(
        [row["ci_high_episode_utility_v2_balanced_auc"] for row in candidates]
    )
    positions = np.arange(len(candidates))
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.axvline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    for index, row in enumerate(candidates):
        method = row["method"]
        axis.errorbar(
            values[index],
            positions[index],
            xerr=np.asarray(
                [
                    [values[index] - low[index]],
                    [high[index] - values[index]],
                ]
            ),
            fmt="o",
            markersize=8 if method == "active_map" else 6,
            capsize=3,
            color=COLORS[method],
            linewidth=1.8,
            zorder=3,
        )
    axis.set_yticks(
        positions, [LABELS[row["method"]] for row in candidates]
    )
    axis.invert_yaxis()
    axis.set_xlabel("Balanced Utility AUC delta vs Direct SFT")
    axis.set_title("Matched SN7 Controller Utility")
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=320, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_safety(rows: list[dict], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.8, 4.8))
    for row in rows:
        method = row["method"]
        marker = "*" if method == "active_map" else "o"
        size = 180 if method == "active_map" else 65
        axis.scatter(
            row["missed_edit_auc"],
            row["false_edit_auc"],
            s=size,
            marker=marker,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            label=LABELS[method],
            zorder=3,
        )
    axis.set_xlabel("Missed-edit AUC (lower is better)")
    axis.set_ylabel("False-edit AUC (lower is better)")
    axis.set_title("Matched SN7 Safety Plane")
    axis.grid(color="#E1E1E1", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=320, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load(args.table_json)
    plot_utility(rows, args.output_dir / "controller_utility_ci.png")
    plot_safety(rows, args.output_dir / "controller_safety_plane.png")
    print(args.output_dir)


if __name__ == "__main__":
    main()
