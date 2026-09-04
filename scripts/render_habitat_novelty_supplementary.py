#!/usr/bin/env python3
"""Render the held-out Habitat novelty-gate supplementary evidence figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

POLICY_SPECS = {
    "acquire_all": {
        "directory": "all_seed{seed}",
        "label": "Acquire all",
        "color": "#343A40",
        "marker": "^",
    },
    "unknown_gate@15": {
        "directory": "unknown15_seed{seed}",
        "label": "Unknown @15",
        "color": "#6B8EAD",
        "marker": "o",
    },
    "novelty_gate@8": {
        "directory": "novelty08_seed{seed}",
        "label": "Novelty @8",
        "color": "#16877A",
        "marker": "s",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/robotics/habitat_novelty_heldout_v2_20260830"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/robotics/habitat_novelty_heldout_v2_20260830/paper_figure"),
    )
    parser.add_argument("--representative-seed", type=int, default=20270865)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_rows(run_root: Path, aggregate: dict) -> list[dict]:
    paired_by_policy = {item["policy"]: item for item in aggregate["paired_bootstrap"]}
    novelty_seeds = sorted(
        int(path.name.split("seed")[-1])
        for path in run_root.glob("novelty08_seed*")
        if path.is_dir()
    )
    if len(novelty_seeds) != 8:
        raise ValueError(f"Expected 8 held-out seeds, found {len(novelty_seeds)}")

    rows: list[dict] = []
    for seed in novelty_seeds:
        all_summary = load_json(run_root / f"all_seed{seed}" / "summary.json")
        all_calls = int(all_summary["post_initial_acquisitions"])
        for policy, spec in POLICY_SPECS.items():
            summary = load_json(run_root / spec["directory"].format(seed=seed) / "summary.json")
            quality = summary["final_reference_map_quality"]
            rows.append(
                {
                    "seed": seed,
                    "policy": policy,
                    "post_initial_acquisitions": int(summary["post_initial_acquisitions"]),
                    "total_logical_views": int(summary["sensor_calls"]),
                    "calls_saved_vs_acquire_all": (
                        all_calls - int(summary["post_initial_acquisitions"])
                    ),
                    "reference_map_agreement": 1.0 if quality is None else float(quality),
                    "known_map_fraction": float(summary["final_known_cell_fraction"]),
                    "success": int(bool(summary["success"])),
                    "spl": float(summary["spl"]),
                    "collision_proxy_count": int(summary["collision_count"]),
                }
            )

    for policy in ("unknown_gate@15", "novelty_gate@8"):
        paired = paired_by_policy[policy]
        source_values = [
            row["calls_saved_vs_acquire_all"] for row in rows if row["policy"] == policy
        ]
        if not np.allclose(source_values, paired["calls_saved_vs_acquire_all"]["per_seed"]):
            raise ValueError(f"Per-seed call savings disagree for {policy}")
    return rows


def write_source_data(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def render(rows: list[dict], aggregate: dict, representative: Path, output_stem: Path) -> None:
    configure_matplotlib()
    fig = plt.figure(figsize=(7.20, 5.05), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.37], width_ratios=[1.45, 1.0])
    ax_tradeoff = fig.add_subplot(grid[0, 0])
    ax_calls = fig.add_subplot(grid[0, 1])
    ax_map = fig.add_subplot(grid[1, :])

    seeds = sorted({row["seed"] for row in rows})
    rows_by_key = {(row["seed"], row["policy"]): row for row in rows}

    for seed in seeds:
        seed_rows = [
            rows_by_key[(seed, policy)]
            for policy in ("novelty_gate@8", "unknown_gate@15", "acquire_all")
        ]
        ax_tradeoff.plot(
            [row["post_initial_acquisitions"] for row in seed_rows],
            [row["reference_map_agreement"] for row in seed_rows],
            color="#C9CED3",
            linewidth=0.7,
            zorder=1,
        )

    for policy, spec in POLICY_SPECS.items():
        policy_rows = [row for row in rows if row["policy"] == policy]
        x = np.asarray([row["post_initial_acquisitions"] for row in policy_rows])
        y = np.asarray([row["reference_map_agreement"] for row in policy_rows])
        ax_tradeoff.scatter(
            x,
            y,
            s=23,
            marker=spec["marker"],
            facecolor=spec["color"],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.88,
            label=spec["label"],
            zorder=3,
        )
        ax_tradeoff.scatter(
            [x.mean()],
            [y.mean()],
            s=70,
            marker=spec["marker"],
            facecolor=spec["color"],
            edgecolor="#111111",
            linewidth=0.8,
            zorder=4,
        )

    ax_tradeoff.set_xlabel("Post-initial RGB-D acquisitions")
    ax_tradeoff.set_ylabel("Agreement with all-acquired map")
    ax_tradeoff.set_xlim(left=5.2)
    ax_tradeoff.set_ylim(0.875, 1.012)
    ax_tradeoff.set_title("Held-out quality–evidence trade-off", loc="left", pad=5)
    ax_tradeoff.grid(axis="both", color="#E8EAED", linewidth=0.6, zorder=0)
    ax_tradeoff.legend(loc="lower right", ncol=1, handletextpad=0.4, borderaxespad=0.3)
    ax_tradeoff.text(
        0.01,
        0.03,
        "Large symbols: mean; thin lines: matched trajectories",
        transform=ax_tradeoff.transAxes,
        fontsize=6.2,
        color="#5B6168",
        ha="left",
        va="bottom",
    )
    panel_label(ax_tradeoff, "a")

    paired_by_policy = {item["policy"]: item for item in aggregate["paired_bootstrap"]}
    policies = ["unknown_gate@15", "novelty_gate@8"]
    y_positions = [0, 1]
    deterministic_offsets = np.linspace(-0.08, 0.08, len(seeds))
    for policy, y_pos in zip(policies, y_positions, strict=True):
        spec = POLICY_SPECS[policy]
        paired = paired_by_policy[policy]["calls_saved_vs_acquire_all"]
        values = np.asarray(paired["per_seed"], dtype=float)
        ci_low, ci_high = paired["ci95"]
        ax_calls.scatter(
            values,
            y_pos + deterministic_offsets,
            s=20,
            facecolor="white",
            edgecolor=spec["color"],
            linewidth=0.9,
            zorder=3,
        )
        ax_calls.errorbar(
            paired["mean"],
            y_pos,
            xerr=np.asarray([[paired["mean"] - ci_low], [ci_high - paired["mean"]]]),
            fmt="D",
            color=spec["color"],
            markerfacecolor=spec["color"],
            markeredgecolor="#111111",
            markersize=5.8,
            linewidth=1.5,
            capsize=2.5,
            zorder=4,
        )

    ax_calls.axvline(0, color="#737A80", linewidth=0.8, linestyle="--", zorder=1)
    ax_calls.set_yticks(y_positions, [POLICY_SPECS[p]["label"] for p in policies])
    ax_calls.set_xlabel("Calls saved vs acquire all")
    ax_calls.set_xlim(-0.5, 8.8)
    ax_calls.set_ylim(-0.45, 1.45)
    ax_calls.set_title("Paired acquisition savings", loc="left", pad=5)
    ax_calls.grid(axis="x", color="#E8EAED", linewidth=0.6, zorder=0)
    ax_calls.text(
        0.98,
        0.03,
        "Diamonds: mean with paired 95% CI\nOpen circles: trajectories (n=8)",
        transform=ax_calls.transAxes,
        fontsize=6.2,
        color="#5B6168",
        ha="right",
        va="bottom",
    )
    panel_label(ax_calls, "b")

    map_image = np.asarray(Image.open(representative).convert("RGB"))
    ax_map.imshow(map_image)
    ax_map.set_axis_off()
    ax_map.set_title("Map-native mechanism case (held-out seed 20270865)", loc="left", pad=4)
    panel_label(ax_map, "c")

    fig.suptitle(
        "Stateful novelty reduces redundant RGB-D acquisition under shared locomotion",
        fontsize=9,
        fontweight="bold",
        x=0.01,
        ha="left",
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    aggregate = load_json(run_root / "aggregate_paired_v2" / "summary.json")
    rows = collect_rows(run_root, aggregate)
    write_source_data(rows, output_dir / "source_data.csv")
    representative = run_root / "visuals" / f"heldout_seed{args.representative_seed}_map_native.png"
    if not representative.is_file():
        raise FileNotFoundError(representative)
    render(
        rows,
        aggregate,
        representative,
        output_dir / "habitat_novelty_supplementary",
    )


if __name__ == "__main__":
    main()
