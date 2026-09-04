#!/usr/bin/env python3
"""Build paper assets for the uncertainty-gate corruption frontier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

COLORS = {
    "ActiveMap": "#2A9D8F",
    "Uncertainty q05": "#4C78A8",
    "Uncertainty q15": "#F2A541",
    "Uncertainty q30": "#E45756",
}


def _interval(metric: dict[str, float]) -> str:
    return (
        f"{metric['delta']:+.6f} "
        f"[{metric['ci95_low']:+.6f},{metric['ci95_high']:+.6f}]"
    )


def build_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for severity_row in payload["rows"]:
        severity = int(severity_row["severity"])
        active_reference = None
        for label, item in severity_row["quantiles"].items():
            uncertainty = item["writeback"]["uncertainty"]
            active = item["writeback"]["active"]
            if active_reference is None:
                active_reference = active
            elif active != active_reference:
                raise ValueError(f"severity {severity}: ActiveMap means differ")
            paired = item["paired_writeback"]["active_vs_uncertainty"]
            quality = paired["raster_iou_auc"]
            false_edit = paired["false_edit_auc"]
            cost = paired["spent_cost_auc"]
            rows.append(
                {
                    "severity": severity,
                    "quantile": label,
                    "uncertainty_call_rate": item["controller"][
                        "tool_call_episode_rate"
                    ],
                    "uncertainty_iou_auc": uncertainty["raster_iou_auc"],
                    "active_iou_auc": active["raster_iou_auc"],
                    "active_vs_uncertainty_iou": quality,
                    "uncertainty_false_edit_auc": uncertainty["false_edit_auc"],
                    "active_false_edit_auc": active["false_edit_auc"],
                    "active_vs_uncertainty_false_edit": false_edit,
                    "uncertainty_cost_auc": uncertainty["spent_cost_auc"],
                    "active_cost_auc": active["spent_cost_auc"],
                    "active_vs_uncertainty_cost": cost,
                    "quality_ci_positive": quality["ci95_low"] > 0.0,
                    "false_edit_ci_negative": false_edit["ci95_high"] < 0.0,
                    "cost_ci_negative": cost["ci95_high"] < 0.0,
                }
            )
    return rows


def write_tables(rows: list[dict[str, Any]], output_dir: Path) -> None:
    csv_path = output_dir / "uncertainty_frontier_table.csv"
    fieldnames = (
        "severity",
        "quantile",
        "uncertainty_call_rate",
        "uncertainty_iou_auc",
        "active_iou_auc",
        "uncertainty_false_edit_auc",
        "active_false_edit_auc",
        "uncertainty_cost_auc",
        "active_cost_auc",
        "quality_ci_positive",
        "false_edit_ci_negative",
        "cost_ci_negative",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)

    lines = [
        "| Shift | Gate | Gate call rate | Gate IoU | ActiveMap IoU | "
        "Delta IoU (95% CI) | Gate FE | ActiveMap FE | Delta FE (95% CI) | "
        "Delta cost (95% CI) |",
        "| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['severity']}px | {row['quantile']} | "
            f"{row['uncertainty_call_rate']:.4f} | "
            f"{row['uncertainty_iou_auc']:.6f} | "
            f"{row['active_iou_auc']:.6f} | "
            f"{_interval(row['active_vs_uncertainty_iou'])} | "
            f"{row['uncertainty_false_edit_auc']:.6f} | "
            f"{row['active_false_edit_auc']:.6f} | "
            f"{_interval(row['active_vs_uncertainty_false_edit'])} | "
            f"{_interval(row['active_vs_uncertainty_cost'])} |"
        )
    (output_dir / "uncertainty_frontier_table.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    latex = [
        "\\begin{tabular}{rrcccc}",
        "\\toprule",
        "Shift & Gate & Call rate & Gate IoU & ActiveMap IoU & $\\Delta$ IoU \\\\",
        "\\midrule",
    ]
    for row in rows:
        latex.append(
            f"{row['severity']} & {row['quantile'][1:]} & "
            f"{100.0 * row['uncertainty_call_rate']:.1f}\\% & "
            f"{row['uncertainty_iou_auc']:.3f} & "
            f"{row['active_iou_auc']:.3f} & "
            f"{row['active_vs_uncertainty_iou']['delta']:+.3f} \\\\"
        )
    latex.extend(["\\bottomrule", "\\end{tabular}"])
    (output_dir / "uncertainty_frontier_table.tex").write_text(
        "\n".join(latex) + "\n",
        encoding="utf-8",
    )


def write_claim_gate(rows: list[dict[str, Any]], output_dir: Path) -> None:
    checks = []
    for row in rows:
        checks.append(
            {
                "severity": row["severity"],
                "quantile": row["quantile"],
                "quality_ci_positive": row["quality_ci_positive"],
                "false_edit_ci_negative": row["false_edit_ci_negative"],
                "cost_ci_negative": row["cost_ci_negative"],
                "active_strict_pareto_dominance": (
                    row["quality_ci_positive"]
                    and row["false_edit_ci_negative"]
                    and row["cost_ci_negative"]
                ),
            }
        )
    payload = {
        "schema_version": "sn7-controller-uncertainty-claim-gate-v1",
        "checks": checks,
        "all_severities_have_dominated_uncertainty_point": all(
            any(
                item["active_strict_pareto_dominance"]
                for item in checks
                if item["severity"] == severity
            )
            for severity in sorted({item["severity"] for item in checks})
        ),
        "interpretation": (
            "Strict dominance requires positive quality CI, negative false-edit "
            "CI, and negative cost CI for the same uncertainty workpoint."
        ),
        "test_assets_read": False,
    }
    (output_dir / "claim_gate.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = (
        ("active_iou_auc", "uncertainty_iou_auc", "Executable raster IoU AUC"),
        (
            "active_false_edit_auc",
            "uncertainty_false_edit_auc",
            "False-edit AUC",
        ),
        ("active_cost_auc", "uncertainty_cost_auc", "Evidence cost AUC"),
        (None, "uncertainty_call_rate", "Tool-call episode rate"),
    )
    severities = sorted({row["severity"] for row in rows})
    by_quantile = {
        quantile: sorted(
            [row for row in rows if row["quantile"] == quantile],
            key=lambda row: row["severity"],
        )
        for quantile in ("q05", "q15", "q30")
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    for axis, (active_key, uncertainty_key, title) in zip(
        axes.ravel(),
        metrics,
        strict=True,
    ):
        if active_key is not None:
            active = [by_quantile["q05"][i][active_key] for i in range(4)]
            axis.plot(
                severities,
                active,
                marker="o",
                linewidth=2.4,
                color=COLORS["ActiveMap"],
                label="ActiveMap",
            )
        for quantile in ("q05", "q15", "q30"):
            label = f"Uncertainty {quantile}"
            axis.plot(
                severities,
                [row[uncertainty_key] for row in by_quantile[quantile]],
                marker="o",
                linewidth=1.8,
                color=COLORS[label],
                label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("Maximum prior-input translation (pixels)")
        axis.set_xticks(severities)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("ActiveMap versus fixed train-only uncertainty gates", fontsize=14)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output_dir / f"uncertainty_frontier.{suffix}",
            dpi=240 if suffix == "png" else None,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = build_rows(payload)
    args.output_dir.mkdir(parents=True)
    write_tables(rows, args.output_dir)
    write_claim_gate(rows, args.output_dir)
    plot(rows, args.output_dir)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "sn7-controller-uncertainty-paper-assets-v1",
                "source": str(args.summary.resolve()),
                "row_count": len(rows),
                "test_assets_read": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
