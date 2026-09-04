#!/usr/bin/env python3
"""Render the claim-aligned cross-domain evidence matrix from frozen results.

The figure is deliberately sourced from the paper evidence snapshot rather
than training logs. It shows qualified transfer and the TinyCD perception
boundary without converting conditional or negative outcomes into claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT.parent
    / "论文"
    / "iclr-2027-style-files"
    / "iclr2027"
    / "evidence"
    / "evidence_snapshot.json"
)
DEFAULT_OUTPUT = ROOT / "docs" / "figures" / "cross_domain_evidence_matrix_20260809"

COLORS = {
    "ink": "#1F2937",
    "muted": "#64748B",
    "grid": "#D9E1E8",
    "blue": "#0072B2",
    "green": "#009E73",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "gray": "#8A8F98",
    "tiny": "#B8BDC4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def require(mapping: dict, *keys: str) -> float:
    current: object = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing evidence field: {'.'.join(keys)}")
        current = current[key]
    return float(current)


def interval_plot(
    ax: plt.Axes,
    labels: list[str],
    centers: list[float],
    lows: list[float],
    highs: list[float],
    colors: list[str],
    *,
    xlabel: str,
    title: str,
    xlim: tuple[float, float],
) -> None:
    y = np.arange(len(labels))[::-1]
    ax.axvline(0.0, color=COLORS["ink"], linewidth=0.8, linestyle="--", zorder=0)
    for index, (center, low, high, color) in enumerate(zip(centers, lows, highs, colors)):
        left = center - low
        right = high - center
        ax.errorbar(
            center,
            y[index],
            xerr=np.array([[left], [right]]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=6.2,
            linewidth=1.65,
            zorder=3,
        )
    ax.set_yticks(y, labels)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=11.1, fontweight="bold", color=COLORS["ink"])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.65, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=8.6)
    ax.tick_params(axis="x", labelsize=8.4)


def build_figure(data: dict) -> plt.Figure:
    sn7 = data["sn7"]
    muno = data["muno21"]
    space = data["spacenet8"]

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.7,
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.56), constrained_layout=False)
    fig.subplots_adjust(left=0.112, right=0.992, bottom=0.182, top=0.795, wspace=0.56)

    # Lower false edits are favorable. Plot the reduction as a positive delta.
    sn7_center = [
        require(sn7, "vs_no_evidence", "raster_iou_auc_delta"),
        require(sn7, "vs_no_evidence", "safety_utility_delta"),
        -require(sn7, "vs_no_evidence", "false_edit_auc_delta"),
        -require(sn7, "vs_forced", "spent_cost_auc_delta"),
    ]
    sn7_low = [
        require(sn7, "vs_no_evidence", "raster_iou_auc_ci_low"),
        require(sn7, "vs_no_evidence", "safety_utility_ci_low"),
        -require(sn7, "vs_no_evidence", "false_edit_auc_ci_high"),
        -require(sn7, "vs_forced", "spent_cost_auc_ci_high"),
    ]
    sn7_high = [
        require(sn7, "vs_no_evidence", "raster_iou_auc_ci_high"),
        require(sn7, "vs_no_evidence", "safety_utility_ci_high"),
        -require(sn7, "vs_no_evidence", "false_edit_auc_ci_low"),
        -require(sn7, "vs_forced", "spent_cost_auc_ci_low"),
    ]
    interval_plot(
        axes[0],
        ["Raster-IoU AUC", "Safety utility", "False-edit reduction", "Cost reduction vs forced"],
        sn7_center,
        sn7_low,
        sn7_high,
        [COLORS["blue"], COLORS["green"], COLORS["green"], COLORS["orange"]],
        xlabel="Favorable delta",
        title="(a) SN7: map-level mechanism",
        xlim=(-0.004, 0.036),
    )

    muno_centers = [
        require(muno, "vs_always_stop", "apls_delta"),
        require(muno, "vs_always_stop", "pixel_f1_delta"),
        require(muno, "vs_old_selector", "apls_delta"),
        require(muno, "vs_old_selector", "pixel_f1_delta"),
    ]
    muno_lows = [
        require(muno, "vs_always_stop", "apls_ci_low"),
        require(muno, "vs_always_stop", "pixel_f1_ci_low"),
        require(muno, "vs_old_selector", "apls_ci_low"),
        require(muno, "vs_old_selector", "pixel_f1_ci_low"),
    ]
    muno_highs = [
        require(muno, "vs_always_stop", "apls_ci_high"),
        require(muno, "vs_always_stop", "pixel_f1_ci_high"),
        require(muno, "vs_old_selector", "apls_ci_high"),
        require(muno, "vs_old_selector", "pixel_f1_ci_high"),
    ]
    interval_plot(
        axes[1],
        ["APLS vs STOP", "Pixel-F1 vs STOP", "APLS vs old", "Pixel-F1 vs old"],
        muno_centers,
        muno_lows,
        muno_highs,
        [COLORS["blue"], COLORS["blue"], COLORS["purple"], COLORS["purple"]],
        xlabel="Metric delta",
        title="(b) MUNO21: graph transfer",
        xlim=(-0.04, 0.40),
    )

    backend_rows = [
        ("ChangeFormer", "changeformer", COLORS["blue"]),
        ("BAN-MiT", "ban", COLORS["purple"]),
        ("Changer (rank-2)", "changer_rank2", COLORS["orange"]),
        ("Changer (rank-100)", "changer_rank100", COLORS["green"]),
        ("TinyCD (collapsed)", "tinycd_rank100", COLORS["tiny"]),
    ]
    labels = [entry[0] for entry in backend_rows]
    centers = [require(space, entry[1], "learned_first_delta") for entry in backend_rows]
    lows = [require(space, entry[1], "learned_first_ci_low") for entry in backend_rows]
    highs = [require(space, entry[1], "learned_first_ci_high") for entry in backend_rows]
    interval_plot(
        axes[2],
        labels,
        centers,
        lows,
        highs,
        [entry[2] for entry in backend_rows],
        xlabel="Selection IoU delta vs first",
        title="(c) SpaceNet8: qualified-backend boundary",
        xlim=(-0.10, 0.54),
    )

    fig.suptitle(
        "Active verification transfers across map geometries with qualified candidate perception",
        fontsize=12.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.5,
        0.044,
        "Points: estimates; bars: paired 95% CIs. SN7: frozen three-seed AOIs. "
        "MUNO21: independent selector checkpoints.",
        ha="center",
        fontsize=7.35,
        color=COLORS["muted"],
    )
    return fig


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    figure = build_figure(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "cross_domain_evidence_matrix"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "schema_version": "cross-domain-evidence-matrix-v1",
        "source": str(args.input),
        "source_kind": "frozen paper evidence snapshot",
        "test_assets_read": False,
        "included": [
            "SN7 counterfactual verification and Safe Commit metrics",
            "MUNO21 graph-transfer metrics",
            "SpaceNet8 qualified-backend and TinyCD boundary metrics",
        ],
        "statistical_note": "All intervals are reproduced from the frozen evidence snapshot; no visual example or threshold was selected from test assets.",
    }
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
