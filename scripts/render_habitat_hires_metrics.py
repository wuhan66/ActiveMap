#!/usr/bin/env python3
"""Render class-aware held-out Habitat acquisition metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

POLICIES = ("unknown_gate@15", "novelty_gate@8")
LABELS = {"unknown_gate@15": "Unknown @15", "novelty_gate@8": "Novelty @8"}
COLORS = {"unknown_gate@15": "#6B8EAD", "novelty_gate@8": "#16877A"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output_stem", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
    )


def interval(aggregate: dict, policy: str, metric: str) -> tuple[float, float, float]:
    report = next(row for row in aggregate["paired_bootstrap"] if row["policy"] == policy)
    payload = report[metric]
    return float(payload["mean"]), float(payload["ci95"][0]), float(payload["ci95"][1])


def collect(run_root: Path) -> tuple[list[int], dict[str, dict[str, np.ndarray]], dict]:
    aggregate = load_json(run_root / "aggregate_rich_v2" / "summary.json")
    seeds = sorted(
        int(path.name.split("seed")[-1]) for path in run_root.glob("all_seed*") if path.is_dir()
    )
    rows: dict[str, dict[str, list[float]]] = {policy: {} for policy in ("acquire_all", *POLICIES)}
    directories = {
        "acquire_all": "all_seed{seed}",
        "unknown_gate@15": "unknown15_seed{seed}",
        "novelty_gate@8": "novelty_seed{seed}",
    }
    for policy, pattern in directories.items():
        for seed in seeds:
            summary = load_json(run_root / pattern.format(seed=seed) / "summary.json")
            for key, value in summary.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    rows[policy].setdefault(key, []).append(float(value))
    arrays = {
        policy: {key: np.asarray(values) for key, values in metrics.items()}
        for policy, metrics in rows.items()
    }
    return seeds, arrays, aggregate


def mean_ci(ax: plt.Axes, x: float, policy: str, aggregate: dict, metric: str) -> None:
    mean, low, high = interval(aggregate, policy, metric)
    ax.errorbar(
        x,
        mean,
        yerr=np.asarray([[mean - low], [high - mean]]),
        fmt="D",
        markersize=5.5,
        markerfacecolor=COLORS[policy],
        markeredgecolor="#111111",
        color=COLORS[policy],
        linewidth=1.2,
        capsize=2.2,
        zorder=4,
    )


def main() -> None:
    args = parse_args()
    configure_style()
    _, data, aggregate = collect(args.run_root.resolve())
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.45), layout="constrained")
    ax_calls, ax_quality, ax_safety, ax_efficiency = axes.flat
    offsets = {POLICIES[0]: -0.12, POLICIES[1]: 0.12}
    deterministic_jitter = np.linspace(-0.055, 0.055, 8)

    for y, policy in enumerate(POLICIES):
        reference_calls = data["acquire_all"]["post_initial_acquisitions"]
        values = reference_calls - data[policy]["post_initial_acquisitions"]
        ax_calls.scatter(
            values,
            y + deterministic_jitter,
            s=20,
            facecolor="white",
            edgecolor=COLORS[policy],
            linewidth=0.9,
            zorder=3,
        )
        mean, low, high = interval(aggregate, policy, "calls_saved_vs_acquire_all")
        ax_calls.errorbar(
            mean,
            y,
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt="D",
            markersize=5.8,
            markerfacecolor=COLORS[policy],
            markeredgecolor="#111111",
            color=COLORS[policy],
            linewidth=1.3,
            capsize=2.3,
            zorder=4,
        )
    ax_calls.axvline(0, color="#6F7780", linewidth=0.8, linestyle="--")
    ax_calls.set_yticks((0, 1), [LABELS[policy] for policy in POLICIES])
    ax_calls.set_xlabel("Post-initial calls saved vs acquire all")
    ax_calls.set_title("Logical RGB-D acquisition cost", loc="left")
    ax_calls.grid(axis="x", color="#E5E8EB", linewidth=0.6)
    panel_label(ax_calls, "a")

    quality_metrics = (
        ("final_reference_map_quality", "Agreement"),
        ("occupied_iou", "Occupied IoU"),
        ("balanced_iou", "Balanced IoU"),
    )
    for index, (metric, _) in enumerate(quality_metrics):
        for policy in POLICIES:
            x = index + offsets[policy]
            values = data[policy][metric]
            ax_quality.scatter(
                np.full(len(values), x) + deterministic_jitter * 0.6,
                values,
                s=17,
                facecolor="white",
                edgecolor=COLORS[policy],
                linewidth=0.8,
                alpha=0.85,
            )
            mean_ci(ax_quality, x, policy, aggregate, metric)
    ax_quality.set_xticks(range(len(quality_metrics)), [label for _, label in quality_metrics])
    ax_quality.set_ylim(0.54, 1.02)
    ax_quality.set_ylabel("Score")
    ax_quality.set_title("Class-aware map fidelity", loc="left")
    ax_quality.grid(axis="y", color="#E5E8EB", linewidth=0.6)
    panel_label(ax_quality, "b")

    for index, policy in enumerate(POLICIES):
        values = data[policy]["false_free_rate"]
        ax_safety.scatter(
            np.full(len(values), index) + deterministic_jitter,
            values,
            s=19,
            facecolor="white",
            edgecolor=COLORS[policy],
            linewidth=0.9,
            zorder=3,
        )
        mean_ci(ax_safety, index, policy, aggregate, "false_free_rate")
    for seed_index in range(8):
        ax_safety.plot(
            (0, 1),
            (
                data[POLICIES[0]]["false_free_rate"][seed_index],
                data[POLICIES[1]]["false_free_rate"][seed_index],
            ),
            color="#D3D7DB",
            linewidth=0.65,
            zorder=1,
        )
    ax_safety.set_xticks((0, 1), [LABELS[policy] for policy in POLICIES])
    ax_safety.set_ylabel("False-free rate (lower is safer)")
    ax_safety.set_ylim(-0.015, 0.42)
    ax_safety.set_title("Safety-critical occupancy errors", loc="left")
    ax_safety.grid(axis="y", color="#E5E8EB", linewidth=0.6)
    ax_safety.text(
        0.98,
        0.96,
        "False-obstacle means\n0.00114 vs 0.00096",
        transform=ax_safety.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color="#59616A",
    )
    panel_label(ax_safety, "c")

    metric = "reference_covered_cells_per_sensor_call"
    for index, policy in enumerate(POLICIES):
        values = data[policy][metric]
        ax_efficiency.scatter(
            np.full(len(values), index) + deterministic_jitter,
            values,
            s=19,
            facecolor="white",
            edgecolor=COLORS[policy],
            linewidth=0.9,
            zorder=3,
        )
        mean_ci(ax_efficiency, index, policy, aggregate, metric)
    for seed_index in range(8):
        ax_efficiency.plot(
            (0, 1),
            (data[POLICIES[0]][metric][seed_index], data[POLICIES[1]][metric][seed_index]),
            color="#D3D7DB",
            linewidth=0.65,
            zorder=1,
        )
    ax_efficiency.set_xticks((0, 1), [LABELS[policy] for policy in POLICIES])
    ax_efficiency.set_ylabel("Reference cells per call")
    ax_efficiency.set_title("Evidence efficiency", loc="left")
    ax_efficiency.grid(axis="y", color="#E5E8EB", linewidth=0.6)
    panel_label(ax_efficiency, "d")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=COLORS[policy],
            label=LABELS[policy],
        )
        for policy in POLICIES
    ]
    fig.legend(handles=handles, loc="outside upper right", ncol=2, fontsize=6.5)
    fig.suptitle(
        "High-resolution held-out Habitat evaluation (8 matched trajectories)",
        fontsize=9,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
