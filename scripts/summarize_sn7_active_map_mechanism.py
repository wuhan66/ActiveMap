#!/usr/bin/env python3
"""Summarize and plot ActiveMap's matched controller mechanism evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASELINES = (
    "direct_sft",
    "react",
    "plan_execute",
    "geommagent_style",
    "sensesearch_style",
)
LABELS = {
    "direct_sft": "Direct SFT",
    "react": "ReAct",
    "plan_execute": "Plan-Execute",
    "geommagent_style": "GeoMMAgent-style",
    "sensesearch_style": "SenseSearch-style",
}
OPERATIONS = ("KEEP", "ADD", "DELETE", "RESHAPE")
COLORS = {
    "KEEP": "#3B82C4",
    "ADD": "#25856D",
    "DELETE": "#D97706",
    "RESHAPE": "#8B5FA8",
}


def _interval(row: dict, metric: str) -> tuple[float, float, float]:
    value = row["candidate_minus_baseline"][metric]
    return (
        float(value["observed_delta"]),
        float(value["ci95_low"]),
        float(value["ci95_high"]),
    )


def _plot_pairwise(pairwise: dict[str, dict], output: Path) -> None:
    values = []
    low = []
    high = []
    for baseline in BASELINES:
        interval = pairwise[baseline]["candidate_minus_seed_matched_sft"][
            "episode_utility_v2_balanced_auc"
        ]
        values.append(float(interval["observed_delta"]))
        low.append(float(interval["ci95_low"]))
        high.append(float(interval["ci95_high"]))
    values_array = np.asarray(values)
    positions = np.arange(len(BASELINES))
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.axvline(0.0, color="#777777", linestyle="--", linewidth=1)
    axis.errorbar(
        values_array,
        positions,
        xerr=np.asarray(
            [values_array - np.asarray(low), np.asarray(high) - values_array]
        ),
        fmt="o",
        color="#D64545",
        markersize=7,
        capsize=3,
        linewidth=1.8,
    )
    axis.set_yticks(positions, [LABELS[name] for name in BASELINES])
    axis.invert_yaxis()
    axis.set_xlabel("ActiveMap balanced-utility AUC delta")
    axis.set_title("ActiveMap vs Matched Controllers")
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=320, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_operations(operations: dict[str, dict], output: Path) -> None:
    metrics = (
        ("raster_iou", "Raster IoU delta"),
        ("episode_utility_v2_balanced", "Balanced utility delta"),
        ("false_edit", "False-edit delta"),
        ("missed_edit", "Missed-edit delta"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2))
    positions = np.arange(len(OPERATIONS))
    for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
        values = np.asarray(
            [_interval(operations[op], metric)[0] for op in OPERATIONS]
        )
        low = np.asarray(
            [_interval(operations[op], metric)[1] for op in OPERATIONS]
        )
        high = np.asarray(
            [_interval(operations[op], metric)[2] for op in OPERATIONS]
        )
        axis.axhline(0.0, color="#777777", linestyle="--", linewidth=0.9)
        axis.bar(
            positions,
            values,
            color=[COLORS[op] for op in OPERATIONS],
            width=0.62,
            yerr=np.asarray([values - low, high - values]),
            capsize=3,
        )
        axis.set_xticks(positions, OPERATIONS)
        axis.set_title(title)
        axis.grid(axis="y", color="#E1E1E1", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("ActiveMap Operation-Stratified Effects", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=320, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _markdown(pairwise: dict[str, dict], operations: dict[str, dict]) -> str:
    lines = [
        "# SN7 ActiveMap Mechanism Evidence",
        "",
        "Validation-only, three seeds and 512 matched records per seed.",
        "",
        "## Pairwise Controller Evidence",
        "",
        "| Baseline | Balanced-U delta | 95% CI | Safety-U delta | Raster-IoU delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for baseline in BASELINES:
        payload = pairwise[baseline]["candidate_minus_seed_matched_sft"]
        balanced = payload["episode_utility_v2_balanced_auc"]
        safety = payload["episode_utility_v2_safety_auc"]
        raster = payload["raster_iou_auc"]
        lines.append(
            f"| {LABELS[baseline]} | {balanced['observed_delta']:.6f} | "
            f"[{balanced['ci95_low']:.6f}, {balanced['ci95_high']:.6f}] | "
            f"{safety['observed_delta']:.6f} | "
            f"{raster['observed_delta']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Operation-Stratified Evidence vs Direct SFT",
            "",
            "| Operation | N | Raster delta | Balanced-U delta | False-edit delta | "
            "Missed-edit delta | Action-flip rate | Accuracy delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for operation in OPERATIONS:
        row = operations[operation]
        values = {
            metric: _interval(row, metric)[0]
            for metric in (
                "raster_iou",
                "episode_utility_v2_balanced",
                "false_edit",
                "missed_edit",
                "action_flip",
                "terminal_accuracy",
            )
        }
        lines.append(
            f"| {operation} | {row['record_count']} | "
            f"{values['raster_iou']:.6f} | "
            f"{values['episode_utility_v2_balanced']:.6f} | "
            f"{values['false_edit']:.6f} | "
            f"{values['missed_edit']:.6f} | "
            f"{values['action_flip']:.6f} | "
            f"{values['terminal_accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "KEEP gains come from executable false-edit suppression even though the "
            "terminal action policy is more aggressive. ADD gains come from evidence "
            "and geometry refinement without terminal action flips. DELETE mainly "
            "shows terminal action corrections, while RESHAPE remains unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairwise_dir", type=Path)
    parser.add_argument("strata_direct_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairwise = {
        baseline: json.loads(
            (
                args.pairwise_dir
                / f"active_map_vs_{baseline}.json"
            ).read_text(encoding="utf-8")
        )
        for baseline in BASELINES
    }
    strata = json.loads(args.strata_direct_json.read_text(encoding="utf-8"))
    operations = strata["operations"]
    combined = {
        "schema_version": "sn7-active-map-mechanism-summary-v1",
        "pairwise": pairwise,
        "operation_strata_vs_direct": strata,
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(
        _markdown(pairwise, operations), encoding="utf-8"
    )
    _plot_pairwise(pairwise, args.output_dir / "pairwise_utility_ci.png")
    _plot_operations(operations, args.output_dir / "operation_effects.png")
    print(args.output_dir)


if __name__ == "__main__":
    main()
