#!/usr/bin/env python3
"""Plot the audited SN7 full-validation controller matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ORDER = ("old_vla", "direct_vlm", "react", "hybrid_8k")
LABELS = {
    "old_vla": "Old VLA",
    "direct_vlm": "Direct VLM",
    "react": "ReAct protocol",
    "hybrid_8k": "ActiveMap Hybrid 8K",
}
COLORS = {
    "old_vla": "#666666",
    "direct_vlm": "#2878B5",
    "react": "#D9851F",
    "hybrid_8k": "#C23B3B",
}
MARKERS = {"old_vla": "o", "direct_vlm": "s", "react": "D", "hybrid_8k": "*"}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("all_checks_passed") is not True:
        raise ValueError("controller audit did not pass")
    if payload.get("test_assets_read") is not False:
        raise ValueError("controller audit is not validation-only")
    if payload.get("support_identical") is not True:
        raise ValueError("controller support is not identical")
    return payload


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=320, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _style(axis: plt.Axes) -> None:
    axis.grid(color="#E2E2E2", linewidth=0.7, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)


def plot_qc_forest(payload: dict[str, Any], output: Path) -> None:
    comparisons = payload["comparisons_vs_reference"]
    methods = ("direct_vlm", "react", "hybrid_8k")
    y = np.arange(len(methods))
    fig, axis = plt.subplots(figsize=(7.2, 3.5))
    axis.axvline(0.0, color="#777777", linestyle="--", linewidth=1.0, zorder=1)
    for position, method in zip(y, methods, strict=True):
        interval = comparisons[method]["intervals"]["mean_quality_cost_utility"]
        value = float(interval["observed_delta"])
        low = float(interval["ci95_low"])
        high = float(interval["ci95_high"])
        axis.errorbar(
            value,
            position,
            xerr=np.asarray([[value - low], [high - value]]),
            fmt=MARKERS[method],
            markersize=11 if method == "hybrid_8k" else 7,
            color=COLORS[method],
            capsize=4,
            linewidth=1.8,
            zorder=3,
        )
    axis.set_yticks(y, [LABELS[method] for method in methods])
    axis.invert_yaxis()
    axis.set_xlabel("Quality-cost utility delta vs Old VLA")
    axis.set_title("Matched Full-Validation Utility", fontsize=12)
    _style(axis)
    fig.tight_layout()
    _save(fig, output)


def plot_budget_frontier(payload: dict[str, Any], output: Path) -> None:
    rows = payload["budget_summary"]
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for method in ORDER:
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: float(row["budget"]),
        )
        axis.plot(
            [float(row["budget"]) for row in selected],
            [float(row["mean_quality_cost_utility"]) for row in selected],
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=9 if method == "hybrid_8k" else 6,
            linewidth=2.2 if method == "hybrid_8k" else 1.6,
            label=LABELS[method],
            zorder=3,
        )
    axis.axhline(0.0, color="#888888", linestyle="--", linewidth=0.9)
    axis.set_xlabel("Evidence budget")
    axis.set_ylabel("Quality-cost utility")
    axis.set_title("Budgeted Evidence Frontier", fontsize=12)
    axis.legend(
        frameon=False,
        fontsize=8,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
    )
    _style(axis)
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    _save(fig, output)


def plot_cost_safety(payload: dict[str, Any], output: Path) -> None:
    rows = {row["method"]: row for row in payload["method_summary"]}
    efficiency_offsets = {
        "old_vla": (-58, 8),
        "direct_vlm": (6, -15),
        "react": (9, -17),
        "hybrid_8k": (-8, 13),
    }
    safety_offsets = {
        "old_vla": (6, 8),
        "direct_vlm": (-28, -18),
        "react": (8, 8),
        "hybrid_8k": (-105, -18),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for method in ORDER:
        row = rows[method]
        size = 180 if method == "hybrid_8k" else 70
        axes[0].scatter(
            float(row["mean_cost"]),
            float(row["mean_quality_cost_utility"]),
            s=size,
            marker=MARKERS[method],
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axes[0].annotate(
            LABELS[method],
            (float(row["mean_cost"]), float(row["mean_quality_cost_utility"])),
            xytext=efficiency_offsets[method],
            textcoords="offset points",
            fontsize=8,
        )
        axes[1].scatter(
            float(row["missed_edit_rate"]),
            float(row["false_edit_rate"]),
            s=size,
            marker=MARKERS[method],
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axes[1].annotate(
            LABELS[method],
            (float(row["missed_edit_rate"]), float(row["false_edit_rate"])),
            xytext=safety_offsets[method],
            textcoords="offset points",
            fontsize=8,
        )
    axes[0].set_xlabel("Mean evidence cost (lower is better)")
    axes[0].set_ylabel("Quality-cost utility (higher is better)")
    axes[0].set_title("(a) Efficiency", fontsize=11)
    axes[1].set_xlabel("Missed-edit rate (lower is better)")
    axes[1].set_ylabel("False-edit rate (lower is better)")
    axes[1].set_title("(b) Safety tradeoff", fontsize=11)
    for axis in axes:
        axis.margins(x=0.12, y=0.16)
        _style(axis)
    fig.tight_layout(w_pad=2.5)
    _save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    payload = _load(args.audit_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_qc_forest(payload, args.output_dir / "controller_qc_delta_forest.png")
    plot_budget_frontier(payload, args.output_dir / "controller_budget_frontier.png")
    plot_cost_safety(payload, args.output_dir / "controller_cost_safety_plane.png")
    print(json.dumps({"output": str(args.output_dir), "figures": 3}))


if __name__ == "__main__":
    main()
