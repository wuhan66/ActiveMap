#!/usr/bin/env python3
"""Render the SN7 Step-0 quality-cost-safety validation figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


METHOD_ORDER = (
    "No tool",
    "Forced tools",
    "ActiveMap (benefit-aware)",
)

COLORS = {
    "No tool": "#555B61",
    "Forced tools": "#E07A2D",
    "ActiveMap (benefit-aware)": "#008F83",
}

SHORT_NAMES = {
    "No tool": "No tool",
    "Forced tools": "Forced",
    "ActiveMap (benefit-aware)": "ActiveMap",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed: dict[str, dict[str, float]] = {}
    for row in rows:
        method = str(row.pop("method"))
        row.pop("variant", None)
        indexed[method] = {key: float(value) for key, value in row.items()}
    if set(indexed) != set(METHOD_ORDER):
        raise ValueError("table must contain the frozen three-policy support")
    return indexed


def load_inputs(table_dir: Path) -> dict[str, Any]:
    manifest_path = table_dir / "manifest.json"
    controller_path = table_dir / "controller_table.csv"
    writeback_path = table_dir / "writeback_table.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("split") != "val"
        or manifest.get("test_assets_read") is not False
        or manifest.get("promotion_passed") is not True
    ):
        raise ValueError("figure inputs must be promoted validation-only evidence")
    return {
        "manifest": manifest,
        "controller": _read_csv(controller_path),
        "writeback": _read_csv(writeback_path),
        "paths": {
            "manifest": manifest_path,
            "controller": controller_path,
            "writeback": writeback_path,
        },
    }


def render(table_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    inputs = load_inputs(table_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "figure.dpi": 180,
            "savefig.dpi": 400,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    controller = inputs["controller"]
    writeback = inputs["writeback"]

    ax = axes[0]
    for method in METHOD_ORDER:
        row = controller[method]
        ax.errorbar(
            row["mean_tool_calls_mean"],
            row["terminal_accuracy_mean"],
            yerr=row["terminal_accuracy_std"],
            fmt="o",
            markersize=7.5 if method.endswith("(benefit-aware)") else 6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            color=COLORS[method],
            capsize=2.5,
            zorder=4,
        )
    ax.annotate(
        "",
        xy=(
            controller["ActiveMap (benefit-aware)"]["mean_tool_calls_mean"],
            0.51655,
        ),
        xytext=(
            controller["Forced tools"]["mean_tool_calls_mean"],
            0.51655,
        ),
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.1},
    )
    ax.text(
        0.095,
        0.51672,
        "95.4% fewer calls",
        color="#333333",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    offsets = {
        "No tool": (5, -12),
        "Forced tools": (-5, -22),
        "ActiveMap (benefit-aware)": (5, -22),
    }
    aligns = {
        "No tool": "left",
        "Forced tools": "right",
        "ActiveMap (benefit-aware)": "left",
    }
    for method in METHOD_ORDER:
        row = controller[method]
        ax.annotate(
            SHORT_NAMES[method],
            (row["mean_tool_calls_mean"], row["terminal_accuracy_mean"]),
            xytext=offsets[method],
            textcoords="offset points",
            ha=aligns[method],
            color=COLORS[method],
            fontsize=7.5,
            fontweight="bold" if method.endswith("(benefit-aware)") else "normal",
        )
    ax.set_xscale("symlog", linthresh=0.005, linscale=0.7)
    ax.set_xticks([0.0, 0.01, 0.1, 0.3])
    ax.set_xticklabels(["0", "0.01", "0.1", "0.3"])
    ax.set_ylim(0.5078, 0.5178)
    ax.set_xlabel("Mean tool calls per episode (log scale)")
    ax.set_ylabel("Terminal action accuracy")
    ax.set_title("(a) Sparse evidence acquisition", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#D9DDE0", linewidth=0.6, alpha=0.8)

    ax = axes[1]
    for method in METHOD_ORDER:
        row = writeback[method]
        ax.scatter(
            row["spent_cost_auc"],
            row["raster_iou_auc"],
            s=68 if method.endswith("(benefit-aware)") else 50,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
    active = writeback["ActiveMap (benefit-aware)"]
    no_tool = writeback["No tool"]
    forced = writeback["Forced tools"]
    ax.annotate(
        "",
        xy=(active["spent_cost_auc"], active["raster_iou_auc"]),
        xytext=(no_tool["spent_cost_auc"], no_tool["raster_iou_auc"]),
        arrowprops={"arrowstyle": "->", "color": "#62676C", "lw": 1.0},
    )
    ax.annotate(
        "",
        xy=(active["spent_cost_auc"], active["raster_iou_auc"]),
        xytext=(forced["spent_cost_auc"], forced["raster_iou_auc"]),
        arrowprops={"arrowstyle": "->", "color": "#62676C", "lw": 1.0},
    )
    offsets = {
        "No tool": (-3, -14),
        "Forced tools": (0, 8),
        "ActiveMap (benefit-aware)": (5, -14),
    }
    aligns = {
        "No tool": "right",
        "Forced tools": "center",
        "ActiveMap (benefit-aware)": "left",
    }
    for method in METHOD_ORDER:
        row = writeback[method]
        ax.annotate(
            SHORT_NAMES[method],
            (row["spent_cost_auc"], row["raster_iou_auc"]),
            xytext=offsets[method],
            textcoords="offset points",
            ha=aligns[method],
            color=COLORS[method],
            fontsize=7.5,
            fontweight="bold" if method.endswith("(benefit-aware)") else "normal",
        )
    ax.text(
        0.3165,
        0.49625,
        r"$\Delta$IoU = +0.0061",
        color="#44484C",
        fontsize=7.3,
    )
    ax.text(
        0.3248,
        0.50045,
        r"$\Delta$cost = -0.0266",
        color="#44484C",
        fontsize=7.3,
    )
    ax.set_xlim(0.3085, 0.343)
    ax.set_ylim(0.4921, 0.5013)
    ax.set_xlabel("Spent-cost AUC (lower is better)")
    ax.set_ylabel("Executable raster IoU AUC")
    ax.set_title("(b) Map quality-cost frontier", loc="left", fontweight="bold")
    ax.grid(color="#D9DDE0", linewidth=0.6, alpha=0.8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3, width=0.7)

    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS[method],
            markeredgecolor="white",
            markersize=6.5,
            label=SHORT_NAMES[method],
        )
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.19, top=0.82, wspace=0.31)

    output_dir.mkdir(parents=True)
    outputs = []
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"sn7_step0_quality_cost_safety.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(fig)

    provenance = {
        "schema_version": "sn7-step0-quality-cost-safety-figure-v1",
        "split": "val",
        "test_assets_read": False,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in inputs["paths"].items()
        },
        "outputs": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in outputs
        ],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.table_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
