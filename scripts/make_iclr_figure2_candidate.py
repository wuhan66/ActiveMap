#!/usr/bin/env python3
"""Render a candidate-only, provenance-locked ICLR Figure 2.

This renderer exists because the historical Figure 2 manifest points to an
older SN7 snapshot. The sealed SN7 panel is instead read from the registered
frozen-test receipt, while MUNO21 and SpaceNet8 stay on the canonical evidence
snapshot. It never writes into the ICLR submission directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


EXPECTED_SHA256 = {
    "evidence_snapshot.json": "f3c04d90f7460b9c16daab3448821e756070e9ecad0119e4053cd4f4ccc75d56",
    "sn7_frozen_result_snapshot.json": "983da3488c76689f503ff5a0d313605e1844a8741ef89c5e2975c8c7a2ef2ef9",
}

COLORS = {
    "quality": "#0072B2",
    "safety": "#009E73",
    "cost": "#E69F00",
    "transfer": "#CC79A7",
    "boundary": "#7A7A7A",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path, expected_hash: str) -> dict[str, Any]:
    actual = sha256(path)
    if actual != expected_hash:
        raise ValueError(f"unexpected source hash for {path}: {actual}")
    return json.loads(path.read_text(encoding="utf-8"))


def interval(row: dict[str, float], *, favorable_sign: float = 1.0) -> tuple[float, float, float]:
    """Return point, lower, upper after an optional sign flip."""
    point = float(row["delta"]) * favorable_sign
    low = float(row["ci95_low"]) * favorable_sign
    high = float(row["ci95_high"]) * favorable_sign
    return point, min(low, high), max(low, high)


def plot_forest(
    axis: plt.Axes,
    rows: list[tuple[str, float, float, float, str]],
    title: str,
    xlabel: str,
) -> None:
    y = np.arange(len(rows))[::-1]
    labels = [row[0] for row in rows]
    points = np.asarray([row[1] for row in rows])
    lows = np.asarray([row[2] for row in rows])
    highs = np.asarray([row[3] for row in rows])
    colors = [row[4] for row in rows]

    axis.axvline(0.0, color="#5B6570", linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
    for index, (point, low, high, color) in enumerate(zip(points, lows, highs, colors)):
        axis.errorbar(
            point,
            y[index],
            xerr=[[point - low], [high - point]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.6,
            capsize=3,
            markersize=5.3,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=3,
        )

    axis.set_yticks(y, labels, fontsize=7.4)
    axis.set_title(title, loc="left", fontsize=8.3, fontweight="bold", pad=5, linespacing=1.12)
    axis.set_xlabel(xlabel, fontsize=7.4, labelpad=5)
    axis.tick_params(axis="x", labelsize=7.0, length=3)
    axis.tick_params(axis="y", length=0, pad=3)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#6C757D")
    axis.grid(axis="x", color="#D9DEE3", linewidth=0.55, zorder=0)
    axis.set_axisbelow(True)
    axis.margins(y=0.16)


def build_rows(frozen_sn7: dict[str, Any], canonical: dict[str, Any]) -> tuple[list[Any], list[Any], list[Any]]:
    writeback = frozen_sn7["paired_intervals"]["writeback"]
    sn7_notool = writeback["benefit_vs_notool"]
    sn7_forced = writeback["benefit_vs_forced"]

    sn7 = [
        (
            "Raster-IoU AUC vs no evidence",
            *interval(sn7_notool["raster_iou_auc"]),
            COLORS["quality"],
        ),
        (
            "Safety utility vs no evidence",
            *interval(sn7_notool["episode_utility_v2_safety_auc"]),
            COLORS["safety"],
        ),
        (
            "False-edit reduction vs no evidence",
            *interval(sn7_notool["false_edit_auc"], favorable_sign=-1.0),
            COLORS["safety"],
        ),
        (
            "Cost reduction vs forced evidence",
            *interval(sn7_forced["spent_cost_auc"], favorable_sign=-1.0),
            COLORS["cost"],
        ),
    ]

    muno = canonical["muno21"]
    muno_rows = [
        (
            "APLS vs Always-STOP",
            muno["vs_always_stop"]["apls_delta"],
            muno["vs_always_stop"]["apls_ci_low"],
            muno["vs_always_stop"]["apls_ci_high"],
            COLORS["quality"],
        ),
        (
            "Pixel-F1 vs Always-STOP",
            muno["vs_always_stop"]["pixel_f1_delta"],
            muno["vs_always_stop"]["pixel_f1_ci_low"],
            muno["vs_always_stop"]["pixel_f1_ci_high"],
            COLORS["quality"],
        ),
        (
            "APLS vs old selector",
            muno["vs_old_selector"]["apls_delta"],
            muno["vs_old_selector"]["apls_ci_low"],
            muno["vs_old_selector"]["apls_ci_high"],
            COLORS["transfer"],
        ),
        (
            "Pixel-F1 vs old selector",
            muno["vs_old_selector"]["pixel_f1_delta"],
            muno["vs_old_selector"]["pixel_f1_ci_low"],
            muno["vs_old_selector"]["pixel_f1_ci_high"],
            COLORS["transfer"],
        ),
    ]

    backend_order = (
        ("ChangeFormer", "changeformer", COLORS["quality"]),
        ("BAN-MiT", "ban", COLORS["transfer"]),
        ("Changer (rank-2)", "changer_rank2", COLORS["cost"]),
        ("Changer (rank-100)", "changer_rank100", COLORS["safety"]),
        ("TinyCD (collapsed)", "tinycd_rank100", COLORS["boundary"]),
    )
    spacenet = canonical["spacenet8"]
    spacenet_rows = [
        (
            label,
            spacenet[key]["learned_first_delta"],
            spacenet[key]["learned_first_ci_low"],
            spacenet[key]["learned_first_ci_high"],
            color,
        )
        for label, key, color in backend_order
    ]
    return sn7, muno_rows, spacenet_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    snapshot_path = args.paper_root / "evidence" / "evidence_snapshot.json"
    frozen_path = args.paper_root / "evidence" / "generated" / "sn7_frozen_result_snapshot.json"
    canonical = load_json(snapshot_path, EXPECTED_SHA256[snapshot_path.name])
    frozen_sn7 = load_json(frozen_path, EXPECTED_SHA256[frozen_path.name])
    if (
        frozen_sn7.get("schema_version") != "sn7-frozen-result-paper-snapshot-v1"
        or frozen_sn7.get("split") != "test"
        or frozen_sn7.get("test_assets_read") is not True
        or frozen_sn7.get("promotion_passed") is not True
    ):
        raise ValueError("expected the registered sealed SN7 frozen-test receipt")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    sn7_rows, muno_rows, spacenet_rows = build_rows(frozen_sn7, canonical)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 3.05), constrained_layout=False)
    figure.subplots_adjust(left=0.135, right=0.99, top=0.72, bottom=0.26, wspace=1.02)
    figure.text(
        0.012,
        0.95,
        "Active verification is supported at the written-map endpoint, conditional on candidate perception",
        fontsize=9.5,
        fontweight="bold",
        ha="left",
        va="top",
    )
    plot_forest(axes[0], sn7_rows, "(a) SN7\nfrozen written-map evidence", "Favorable paired delta")
    plot_forest(axes[1], muno_rows, "(b) MUNO21\nroad-graph transfer", "Metric delta")
    plot_forest(axes[2], spacenet_rows, "(c) SpaceNet8\nqualified-backend boundary", "Selection IoU delta vs first")
    figure.text(
        0.012,
        0.045,
        "Points are registered estimates; bars are paired 95% confidence intervals. "
        "Supports: SN7 nine AOIs x three controller seeds; MUNO21 29 tasks x three selector seeds; "
        "SpaceNet8 17 held-out validation episodes.",
        fontsize=6.6,
        color="#4E5963",
        ha="left",
        va="bottom",
    )

    prefix = args.output_dir / "iclr_figure2_candidate_provenance_locked"
    figure.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    figure.savefig(prefix.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    figure.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)

    manifest = {
        "schema_version": "iclr-figure2-candidate-provenance-lock-v1",
        "candidate_only": True,
        "paper_root": str(args.paper_root.resolve()),
        "inputs": {
            str(snapshot_path.resolve()): sha256(snapshot_path),
            str(frozen_path.resolve()): sha256(frozen_path),
        },
        "sn7_source": "frozen receipt paired_intervals.writeback",
        "muno21_source": "canonical evidence_snapshot.json",
        "spacenet8_source": "canonical evidence_snapshot.json",
        "renderer_reads_test_assets": False,
        "sealed_sn7_receipt_test_assets_read": True,
        "outputs": [path.name for path in sorted(args.output_dir.glob("iclr_figure2_candidate_provenance_locked.*"))],
    }
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
