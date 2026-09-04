#!/usr/bin/env python3
"""Render RGB, trajectory, and class-aware map evidence for Habitat rollouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from PIL import Image

UNKNOWN = np.float32(0.5)
COLORS = {
    "unknown": "#A7B0BA",
    "correct_free": "#FFFFFF",
    "correct_obstacle": "#252B32",
    "false_free": "#E4574F",
    "false_obstacle": "#E79A3B",
    "path": "#12A8C7",
    "unknown_policy": "#6B8EAD",
    "novelty_policy": "#16877A",
    "skip": "#E4B02E",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("topdown_context", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--selected-seeds",
        type=int,
        nargs="+",
        default=(20270865, 20270869, 20270862),
    )
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
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def occupancy_rgb(committed: np.ndarray, reference: np.ndarray) -> np.ndarray:
    valid = reference != UNKNOWN
    image = np.full((*reference.shape, 3), (246, 247, 248), dtype=np.uint8)
    image[valid & (committed == UNKNOWN)] = (167, 176, 186)
    image[valid & (committed == 0.0) & (reference == 0.0)] = (255, 255, 255)
    image[valid & (committed == 1.0) & (reference == 1.0)] = (37, 43, 50)
    image[valid & (committed == 0.0) & (reference == 1.0)] = (228, 87, 79)
    image[valid & (committed == 1.0) & (reference == 0.0)] = (231, 154, 59)
    return image


def crop_to_reference(reference: np.ndarray, padding: int = 6) -> tuple[slice, slice]:
    rows, cols = np.nonzero(reference != UNKNOWN)
    if not len(rows):
        return slice(0, reference.shape[0]), slice(0, reference.shape[1])
    return (
        slice(
            max(0, int(rows.min()) - padding),
            min(reference.shape[0], int(rows.max()) + padding + 1),
        ),
        slice(
            max(0, int(cols.min()) - padding),
            min(reference.shape[1], int(cols.max()) + padding + 1),
        ),
    )


def choose_evidence_steps(reference_dir: Path, unknown: dict, novelty: dict) -> list[int]:
    unknown_trace = {int(row["step"]): bool(row["acquired"]) for row in unknown["trace"]}
    novelty_trace = {int(row["step"]): bool(row["acquired"]) for row in novelty["trace"]}
    available = sorted(
        int(path.stem.split("_")[-1]) for path in reference_dir.glob("rgb_step_*.png")
    )
    divergence = [step for step in available if unknown_trace.get(step) != novelty_trace.get(step)]
    if len(divergence) >= 3:
        return [divergence[0], divergence[len(divergence) // 2], divergence[-1]]
    candidates = sorted(set([*divergence, available[len(available) // 3], available[-1]]))
    while len(candidates) < 3:
        candidates.append(available[len(candidates) * len(available) // 3])
        candidates = sorted(set(candidates))
    return candidates[:3]


def world_path(summary: dict) -> tuple[np.ndarray, np.ndarray]:
    positions = [summary["start_position"], *[row["position"] for row in summary["trace"]]]
    return (
        np.asarray([float(position[0]) for position in positions]),
        np.asarray([float(position[2]) for position in positions]),
    )


def map_path(summary: dict, crop: tuple[slice, slice]) -> tuple[np.ndarray, np.ndarray]:
    grid = summary["grid"]
    x, z = world_path(summary)
    cols = (x - float(grid["x_min"])) / float(grid["resolution_m"]) - crop[1].start
    rows = (float(grid["z_max"]) - z) / float(grid["resolution_m"]) - crop[0].start
    return cols, rows


def plot_rgb(
    ax: plt.Axes,
    image_path: Path,
    step: int,
    unknown: dict,
    novelty: dict,
) -> None:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    ax.imshow(image)
    unknown_action = "ACQUIRE" if unknown["trace"][step - 1]["acquired"] else "SKIP"
    novelty_action = "ACQUIRE" if novelty["trace"][step - 1]["acquired"] else "SKIP"
    ax.set_title(
        f"Step {step}  |  Unknown: {unknown_action}  ·  Novelty: {novelty_action}", loc="left"
    )
    ax.set_axis_off()


def plot_topdown(
    ax: plt.Axes,
    navigability: np.ndarray,
    metadata: dict,
    reference_summary: dict,
    unknown: dict,
    novelty: dict,
) -> None:
    lower = metadata["lower_bound_xyz"]
    upper = metadata["upper_bound_xyz"]
    extent = (lower[0], upper[0], lower[2], upper[2])
    ax.imshow(
        navigability,
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["#30363D", "#EDF1F3"]),
        interpolation="nearest",
        aspect="equal",
    )
    path_x, path_z = world_path(reference_summary)
    ax.plot(path_x, path_z, color=COLORS["path"], linewidth=1.6, zorder=3)
    for summary, marker, color in (
        (unknown, "o", COLORS["unknown_policy"]),
        (novelty, "s", COLORS["novelty_policy"]),
    ):
        acquired = [row["position"] for row in summary["trace"] if row["acquired"]]
        skipped = [row["position"] for row in summary["trace"] if not row["acquired"]]
        if acquired:
            ax.scatter(
                [position[0] for position in acquired],
                [position[2] for position in acquired],
                marker=marker,
                s=19,
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.82,
                zorder=4,
            )
        if skipped:
            ax.scatter(
                [position[0] for position in skipped],
                [position[2] for position in skipped],
                marker=marker,
                s=16,
                facecolor="none",
                edgecolor=COLORS["skip"],
                linewidth=0.7,
                alpha=0.75,
                zorder=4,
            )
    start, goal = reference_summary["start_position"], reference_summary["goal_position"]
    ax.scatter(start[0], start[2], s=38, color="#0EA5C6", edgecolor="white", zorder=5)
    ax.scatter(goal[0], goal[2], s=45, color="#B553B3", marker="*", edgecolor="white", zorder=5)
    ax.set_xlim(path_x.min() - 0.7, path_x.max() + 0.7)
    ax.set_ylim(path_z.min() - 0.7, path_z.max() + 0.7)
    ax.set_title("Executed path and policy-specific evidence decisions", loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_policy_map(
    ax: plt.Axes,
    committed: np.ndarray,
    reference: np.ndarray,
    summary: dict,
    crop: tuple[slice, slice],
    label: str,
    policy_color: str,
) -> None:
    image = occupancy_rgb(committed, reference)[crop]
    ax.imshow(image, interpolation="nearest")
    cols, rows = map_path(summary, crop)
    ax.plot(cols, rows, color=COLORS["path"], linewidth=1.0, zorder=3)
    for row in summary["trace"]:
        grid = summary["grid"]
        position = row["position"]
        col = (float(position[0]) - float(grid["x_min"])) / float(grid["resolution_m"]) - crop[
            1
        ].start
        map_row = (float(grid["z_max"]) - float(position[2])) / float(grid["resolution_m"]) - crop[
            0
        ].start
        ax.scatter(
            col,
            map_row,
            s=9,
            facecolor=policy_color if row["acquired"] else "white",
            edgecolor=policy_color if row["acquired"] else COLORS["skip"],
            linewidth=0.45,
            zorder=4,
        )
    ax.set_title(
        f"{label}\n{summary['sensor_calls']} views  ·  Occ. IoU "
        f"{summary['occupied_iou']:.3f}  ·  False-free {summary['false_free_rate']:.3f}",
        loc="left",
        fontsize=6.8,
        linespacing=1.25,
    )
    ax.set_axis_off()


def render_case(
    run_root: Path,
    output_dir: Path,
    seed: int,
    navigability: np.ndarray,
    metadata: dict,
) -> Path:
    reference_dir = run_root / f"all_seed{seed}"
    unknown_dir = run_root / f"unknown15_seed{seed}"
    novelty_dir = run_root / f"novelty_seed{seed}"
    reference_summary = load_json(reference_dir / "summary.json")
    unknown = load_json(unknown_dir / "summary.json")
    novelty = load_json(novelty_dir / "summary.json")
    reference = np.load(reference_dir / "committed_occupancy.npy").astype(np.float32)
    unknown_map = np.load(unknown_dir / "committed_occupancy.npy").astype(np.float32)
    novelty_map = np.load(novelty_dir / "committed_occupancy.npy").astype(np.float32)
    crop = crop_to_reference(reference)
    steps = choose_evidence_steps(reference_dir, unknown, novelty)

    fig = plt.figure(figsize=(7.2, 4.75), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(0.88, 1.12))
    for index, step in enumerate(steps):
        plot_rgb(
            fig.add_subplot(grid[0, index]),
            reference_dir / f"rgb_step_{step:03d}.png",
            step,
            unknown,
            novelty,
        )
    plot_topdown(
        fig.add_subplot(grid[1, 0]),
        navigability,
        metadata,
        reference_summary,
        unknown,
        novelty,
    )
    plot_policy_map(
        fig.add_subplot(grid[1, 1]),
        unknown_map,
        reference,
        unknown,
        crop,
        "Unknown @15",
        COLORS["unknown_policy"],
    )
    plot_policy_map(
        fig.add_subplot(grid[1, 2]),
        novelty_map,
        reference,
        novelty,
        crop,
        "Novelty @8",
        COLORS["novelty_policy"],
    )

    handles = [
        Line2D([0], [0], color=COLORS["path"], linewidth=1.6, label="Executed path"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["unknown_policy"],
            markeredgecolor="white",
            label="Unknown acquired",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=COLORS["novelty_policy"],
            markeredgecolor="white",
            label="Novelty acquired",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=COLORS["skip"],
            label="Skipped",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=COLORS["false_free"],
            markeredgecolor="none",
            label="False free",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=COLORS["false_obstacle"],
            markeredgecolor="none",
            label="False obstacle",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False, fontsize=6.2)
    fig.suptitle(
        f"Held-out Habitat trajectory {seed}: RGB evidence, causal decisions, "
        "and editable occupancy",
        fontsize=9,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    case_dir = output_dir / f"seed{seed}"
    case_dir.mkdir(parents=True, exist_ok=True)
    stem = case_dir / "rgb_trajectory_map_story"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    return stem.with_suffix(".png")


def write_source_data(run_root: Path, output_dir: Path, seeds: list[int]) -> None:
    rows = []
    for seed in seeds:
        for directory, policy in (
            (f"all_seed{seed}", "acquire_all"),
            (f"unknown15_seed{seed}", "unknown_gate@15"),
            (f"novelty_seed{seed}", "novelty_gate@8"),
        ):
            summary = load_json(run_root / directory / "summary.json")
            rows.append(
                {
                    "seed": seed,
                    "policy": policy,
                    "sensor_calls": summary["sensor_calls"],
                    "post_initial_acquisitions": summary["post_initial_acquisitions"],
                    "reference_map_agreement": summary.get("final_reference_map_quality"),
                    "reference_known_coverage": summary.get("reference_known_coverage"),
                    "occupied_iou": summary.get("occupied_iou"),
                    "free_iou": summary.get("free_iou"),
                    "balanced_iou": summary.get("balanced_iou"),
                    "false_free_rate": summary.get("false_free_rate"),
                    "false_obstacle_rate": summary.get("false_obstacle_rate"),
                    "reference_covered_cells_per_sensor_call": summary.get(
                        "reference_covered_cells_per_sensor_call"
                    ),
                }
            )
    with (output_dir / "source_data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    context_root = args.topdown_context.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    navigability = np.load(context_root / "navigability.npy")
    metadata = load_json(context_root / "metadata.json")
    seeds = sorted(
        int(path.name.split("seed")[-1]) for path in run_root.glob("all_seed*") if path.is_dir()
    )
    if len(seeds) != 8:
        raise ValueError(f"Expected 8 held-out trajectories, found {len(seeds)}")
    write_source_data(run_root, output_dir, seeds)
    rendered = {
        seed: render_case(run_root, output_dir, seed, navigability, metadata) for seed in seeds
    }
    missing = set(args.selected_seeds) - set(rendered)
    if missing:
        raise ValueError(f"Selected seeds are unavailable: {sorted(missing)}")
    (output_dir / "selected_cases.json").write_text(
        json.dumps(
            {
                "selected_seeds": args.selected_seeds,
                "case_files": [str(rendered[seed]) for seed in args.selected_seeds],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
