#!/usr/bin/env python3
"""Plot an audited quality-cost-safety effect heatmap versus Always-STOP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

METHODS = (
    ("direct_vlm_no_tools", "Direct VLM\nno tools"),
    ("geommagent", "GeoMMAgent\nprotocol"),
    ("plan_execute", "Plan-and-\nExecute"),
    ("react", "ReAct\nprotocol"),
    ("sensesearch", "SenseSearch\nprotocol"),
    ("hybrid_8k", "Hybrid 8K"),
    ("active_map_safe", "ActiveMap Safe"),
)

METRICS = (
    ("terminal_accuracy", "Terminal\naccuracy", 1.0),
    ("false_edit_rate", "False-edit\nreduction", -1.0),
    ("missed_edit_rate", "Missed-edit\nreduction", -1.0),
    ("mean_quality_gain", "Map-quality\ngain", 1.0),
    ("mean_cost", "Evidence-cost\nreduction", -1.0),
    ("balanced_utility", "Balanced\nutility", 1.0),
    ("safety_utility", "Safety\nutility", 1.0),
    ("cost_aware_utility", "Cost-aware\nutility", 1.0),
)


def load_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("all_checks_passed") is not True:
        raise ValueError("controller audit did not pass")
    if payload.get("reference_method") != "always_stop":
        raise ValueError("reference method must be always_stop")
    if payload.get("support_identical") is not True:
        raise ValueError("controller support is not identical")
    if payload.get("test_assets_read") is not False:
        raise ValueError("audit is not validation-only")
    return payload


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=320, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def plot_heatmap(payload: dict[str, Any], output: Path) -> None:
    comparisons = payload["comparisons_vs_reference"]
    values = np.zeros((len(METHODS), len(METRICS)), dtype=np.float64)
    for row_index, (method, _) in enumerate(METHODS):
        intervals = comparisons[method]["intervals"]
        for column_index, (metric, _, orientation) in enumerate(METRICS):
            interval = intervals[metric]
            values[row_index, column_index] = (
                100.0 * orientation * float(interval["observed_delta"])
            )

    limit = float(np.max(np.abs(values)))
    figure, axis = plt.subplots(figsize=(11.2, 5.5))
    image = axis.imshow(
        values,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        interpolation="nearest",
    )

    axis.set_xticks(
        np.arange(len(METRICS)),
        [label for _, label, _ in METRICS],
        fontsize=8,
    )
    axis.set_yticks(
        np.arange(len(METHODS)),
        [label for _, label in METHODS],
        fontsize=9,
    )
    axis.tick_params(axis="x", length=0, pad=8)
    axis.tick_params(axis="y", length=0, pad=8)

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            text_color = "white" if abs(value) > 0.52 * limit else "#202020"
            axis.text(
                column_index,
                row_index,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
                fontweight="normal",
            )

    axis.set_title(
        "Quality-Cost-Safety Effects vs Always-STOP",
        fontsize=13,
        pad=13,
    )
    axis.set_xlabel(
        "Oriented delta in percentage points; positive is better"
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.026, pad=0.025)
    colorbar.set_label("Oriented delta (percentage points)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    save_figure(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    payload = load_audit(args.audit_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "controller_quality_cost_safety_heatmap.png"
    plot_heatmap(payload, output)
    print(json.dumps({"output": str(output)}))


if __name__ == "__main__":
    main()
