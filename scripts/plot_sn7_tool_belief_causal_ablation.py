#!/usr/bin/env python3
"""Plot SN7 tool execution and belief revision causal ablations."""

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
    ("mean_cost", "Evidence-cost reduction", -1.0),
    ("mean_quality_gain", "Map-quality gain", 1.0),
    ("mean_quality_cost_utility", "Quality-cost utility", 1.0),
    ("mean_episode_utility_v2_proxy_balanced", "Balanced utility", 1.0),
    ("mean_episode_utility_v2_proxy_safety", "Safety utility", 1.0),
    ("mean_episode_utility_v2_proxy_cost_aware", "Cost-aware utility", 1.0),
)


def load_aggregate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("seed_count") != 3:
        raise ValueError(f"expected three seeds: {path}")
    if payload.get("record_count_per_seed") != 6369:
        raise ValueError(f"unexpected support: {path}")
    if payload.get("aoi_count") != 9 or payload.get("repetitions") != 10000:
        raise ValueError(f"unexpected bootstrap protocol: {path}")
    if payload.get("split") != "val" or payload.get("test_assets_read") is not False:
        raise ValueError(f"aggregate is not validation-only: {path}")
    return payload


def oriented_intervals(payload: dict[str, Any]) -> tuple[list[float], list[float], list[float]]:
    intervals = payload["candidate_minus_seed_matched_sft"]
    values, lows, highs = [], [], []
    for metric, _, orientation in METRICS:
        interval = intervals[metric]
        value = orientation * float(interval["observed_delta"])
        endpoints = sorted(
            (
                orientation * float(interval["ci95_low"]),
                orientation * float(interval["ci95_high"]),
            )
        )
        values.append(value)
        lows.append(endpoints[0])
        highs.append(endpoints[1])
    return values, lows, highs


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=320, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def plot(
    tools_payload: dict[str, Any],
    belief_payload: dict[str, Any],
    output: Path,
) -> None:
    panels = (
        (tools_payload, "(a) Tool execution\nwith tools minus no tools", "#2878B5"),
        (belief_payload, "(b) Belief revision\nrecurrent minus identity", "#C23B3B"),
    )
    y = np.arange(len(METRICS))
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.7), sharex=True, sharey=True)
    for axis, (payload, title, color) in zip(axes, panels, strict=True):
        values, lows, highs = oriented_intervals(payload)
        axis.axvline(0.0, color="#777777", linestyle="--", linewidth=1.0, zorder=1)
        for index, value in enumerate(values):
            significant = lows[index] > 0.0 or highs[index] < 0.0
            axis.errorbar(
                value,
                y[index],
                xerr=np.asarray([[value - lows[index]], [highs[index] - value]]),
                fmt="o",
                markersize=7,
                capsize=3,
                linewidth=1.7,
                color=color if significant else "#777777",
                zorder=3,
            )
        axis.set_title(title, fontsize=11)
        axis.grid(axis="x", color="#E1E1E1", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, [label for _, label, _ in METRICS], fontsize=9)
    axes[0].invert_yaxis()
    figure.supxlabel("Oriented delta; positive is better", fontsize=10)
    figure.suptitle("SN7 Tool and Belief Causal Ablations", fontsize=13)
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.96), w_pad=2.0)
    save_figure(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tools_aggregate", type=Path)
    parser.add_argument("belief_aggregate", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    tools_payload = load_aggregate(args.tools_aggregate)
    belief_payload = load_aggregate(args.belief_aggregate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "sn7_tool_belief_causal_forest.png"
    plot(tools_payload, belief_payload, output)
    print(json.dumps({"output": str(output)}))


if __name__ == "__main__":
    main()
