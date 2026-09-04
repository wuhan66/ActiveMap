#!/usr/bin/env python3
"""Plot ActiveMap Safe Commit against the matched Always-STOP control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

METRICS = (
    ("terminal_accuracy", "Terminal accuracy", 1.0),
    ("false_edit_rate", "False-edit reduction", -1.0),
    ("missed_edit_rate", "Missed-edit reduction", -1.0),
    ("mean_quality_gain", "Map-quality gain", 1.0),
    ("mean_cost", "Evidence-cost reduction", -1.0),
    ("balanced_utility", "Balanced utility", 1.0),
    ("safety_utility", "Safety utility", 1.0),
    ("cost_aware_utility", "Cost-aware utility", 1.0),
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("all_checks_passed") is not True:
        raise ValueError("audit did not pass")
    if payload.get("reference_method") != "always_stop":
        raise ValueError("audit reference must be always_stop")
    if payload.get("test_assets_read") is not False:
        raise ValueError("audit is not validation-only")
    return payload


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=320, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_oriented_forest(payload: dict[str, Any], output: Path) -> None:
    intervals = payload["comparisons_vs_reference"]["active_map_safe"]["intervals"]
    values, lows, highs = [], [], []
    for metric, _, orientation in METRICS:
        interval = intervals[metric]
        endpoints = sorted(
            (
                orientation * float(interval["ci95_low"]),
                orientation * float(interval["ci95_high"]),
            )
        )
        values.append(orientation * float(interval["observed_delta"]))
        lows.append(endpoints[0])
        highs.append(endpoints[1])

    y = np.arange(len(METRICS))
    fig, axis = plt.subplots(figsize=(7.8, 5.2))
    axis.axvline(0.0, color="#777777", linestyle="--", linewidth=1.0, zorder=1)
    for index, value in enumerate(values):
        favorable = lows[index] > 0.0
        color = "#C23B3B" if favorable else "#D9851F"
        axis.errorbar(
            value,
            y[index],
            xerr=np.asarray([[value - lows[index]], [highs[index] - value]]),
            fmt="o",
            markersize=7,
            capsize=3,
            linewidth=1.8,
            color=color,
            zorder=3,
        )
    axis.set_yticks(y, [label for _, label, _ in METRICS])
    axis.invert_yaxis()
    axis.set_xlabel("Oriented delta vs Always-STOP (positive is better)")
    axis.set_title("ActiveMap: Quality-Cost-Safety Effects", fontsize=12)
    axis.grid(axis="x", color="#E1E1E1", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, output)


def plot_safety_composition(payload: dict[str, Any], output: Path) -> None:
    rows = {row["method"]: row for row in payload["method_summary"]}
    methods = ("always_stop", "direct_vlm_no_tools", "hybrid_8k", "active_map_safe")
    labels = {
        "always_stop": "Always-STOP",
        "direct_vlm_no_tools": "Direct VLM, no tools",
        "hybrid_8k": "Hybrid 8K",
        "active_map_safe": "Hybrid 8K + Safe Commit",
    }
    colors = {
        "always_stop": "#666666",
        "direct_vlm_no_tools": "#2878B5",
        "hybrid_8k": "#D9851F",
        "active_map_safe": "#C23B3B",
    }
    markers = {
        "always_stop": "o",
        "direct_vlm_no_tools": "s",
        "hybrid_8k": "D",
        "active_map_safe": "*",
    }
    offsets = {
        "always_stop": (6, 7),
        "direct_vlm_no_tools": (-112, -18),
        "hybrid_8k": (7, 7),
        "active_map_safe": (-155, 8),
    }
    fig, axis = plt.subplots(figsize=(7.0, 4.8))
    for method in methods:
        row = rows[method]
        axis.scatter(
            float(row["missed_edit_rate"]),
            float(row["false_edit_rate"]),
            s=190 if method == "active_map_safe" else 75,
            marker=markers[method],
            color=colors[method],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axis.annotate(
            labels[method],
            (float(row["missed_edit_rate"]), float(row["false_edit_rate"])),
            xytext=offsets[method],
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Missed-edit rate (lower is better)")
    axis.set_ylabel("False-edit rate (lower is better)")
    axis.set_title("Safety Composition", fontsize=12)
    axis.margins(x=0.18, y=0.18)
    axis.grid(color="#E1E1E1", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    payload = _load(args.audit_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_oriented_forest(payload, args.output_dir / "active_map_safe_vs_always_stop.png")
    plot_safety_composition(payload, args.output_dir / "active_map_safety_composition.png")
    print(json.dumps({"output": str(args.output_dir), "figures": 2}))


if __name__ == "__main__":
    main()
