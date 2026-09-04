#!/usr/bin/env python3
"""Plot the SN7 policy-relative value of active evidence acquisition."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEAN = ROOT / ".codex-results/paper_p0_20260802/sn7_unified_executable_controller_matrix_20260802/controller_table.json"
DEFAULT_SHIFT = ROOT / "artifacts/paper_evidence/sn7_controller_prior_corruption_realization_v2/paper_assets/table.csv"
DEFAULT_OUTPUT = ROOT / "docs/figures/sn7_reliability_conditioned_frontier_20260802"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-table", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--shift-table", type=Path, default=DEFAULT_SHIFT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_data(clean_path: Path, shift_path: Path) -> dict[str, dict[str, np.ndarray]]:
    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    methods = clean["methods"]
    reference = methods["always_stop"]
    safe_names = sorted(name for name in methods if name.startswith("safe_s"))
    clean_iou = np.array(
        [methods[name]["raster_iou_auc"] - reference["raster_iou_auc"] for name in safe_names]
    )
    clean_false = np.array(
        [methods[name]["false_edit_auc"] - reference["false_edit_auc"] for name in safe_names]
    )

    with shift_path.open("r", encoding="utf-8", newline="") as handle:
        shifted = list(csv.DictReader(handle))

    result: dict[str, dict[str, np.ndarray]] = {
        "iou": {
            "center": np.array([clean_iou.mean()] + [float(row["iou_delta"]) for row in shifted]),
            "low": np.array([clean_iou.min()] + [float(row["iou_ci95_low"]) for row in shifted]),
            "high": np.array([clean_iou.max()] + [float(row["iou_ci95_high"]) for row in shifted]),
            "points": clean_iou,
        },
        "false": {
            "center": np.array([clean_false.mean()] + [float(row["false_edit_delta"]) for row in shifted]),
            "low": np.array([clean_false.min()] + [float(row["false_edit_ci95_low"]) for row in shifted]),
            "high": np.array([clean_false.max()] + [float(row["false_edit_ci95_high"]) for row in shifted]),
            "points": clean_false,
        },
    }
    return result


def draw_interval_panel(ax: plt.Axes, values: dict[str, np.ndarray], title: str, xlabel: str) -> None:
    y = np.arange(3)
    colors = ["#7A7A7A", "#0072B2", "#009E73"]
    centers = values["center"]
    xerr = np.vstack((centers - values["low"], values["high"] - centers))
    ax.axvline(0, color="#222222", linewidth=0.9, linestyle="--", zorder=0)
    for index in range(3):
        ax.errorbar(
            centers[index], y[index], xerr=xerr[:, index : index + 1], fmt="o",
            color=colors[index], ecolor=colors[index], capsize=3.5, markersize=5.5,
            linewidth=1.7, zorder=3,
        )
    jitter = np.linspace(-0.09, 0.09, len(values["points"]))
    ax.scatter(values["points"], np.full_like(jitter, y[0]) + jitter, marker="|", s=70,
               color="#222222", zorder=4)
    ax.set_yticks(y, ["Clean prior", "Prior shift: 4 px", "Prior shift: 8 px"])
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    args = parse_args()
    values = load_data(args.clean_table, args.shift_table)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), constrained_layout=True)
    draw_interval_panel(
        axes[0], values["iou"], "(a) Executable map quality", "Delta raster IoU (higher is better)"
    )
    draw_interval_panel(
        axes[1], values["false"], "(b) Edit safety", "Delta false-edit rate (lower is better)"
    )
    fig.suptitle(
        "Evidence acquisition has policy-relative value", fontsize=10.5, fontweight="bold"
    )
    fig.text(
        0.5, -0.035,
        "Clean: mean and range across 3 Safe Commit seeds (ticks show seeds). Prior shifts: paired-bootstrap 95% CI.",
        ha="center", fontsize=7, color="#444444",
    )

    stem = args.output_dir / "sn7_reliability_conditioned_frontier"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "schema_version": "sn7-reliability-conditioned-frontier-v1",
        "split": "val",
        "test_assets_read": False,
        "clean_seed_count": int(len(values["iou"]["points"])),
        "clean_iou_delta_mean": float(values["iou"]["center"][0]),
        "clean_false_edit_delta_mean": float(values["false"]["center"][0]),
        "source_clean": str(args.clean_table),
        "source_prior_shift": str(args.shift_table),
        "statistical_note": "Clean uses a three-seed range; prior shifts use paired-bootstrap 95% confidence intervals.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
