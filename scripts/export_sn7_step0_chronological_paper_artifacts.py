#!/usr/bin/env python3
"""Export paper tables and a sensitivity figure for continual map maintenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


METRICS = (
    "carry_minus_independent",
    "risk_gated_minus_carry",
    "risk_gated_minus_independent",
    "requested_intervention_rate",
    "risk_gated_intervention_rate",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version")
        != "activemap-chronological-maintenance-three-seed-v1"
        or payload.get("test_assets_read") is not False
        or int(payload.get("seed_count", 0)) != 3
    ):
        raise ValueError(f"not a three-seed validation chronological summary: {path}")
    missing = sorted(set(METRICS) - set(payload.get("aggregate", {})))
    if missing:
        raise ValueError(f"missing chronological metrics in {path}: {missing}")
    return payload


def load(root: Path) -> tuple[list[dict[str, float]], list[dict[str, float]], list[Path]]:
    sensitivity = []
    sources = []
    for threshold, label in ((0.5, "0p5"), (0.7, "0p7"), (0.9, "0p9")):
        path = root / f"threshold_{label}" / "benefit" / "three_seed_v2.json"
        payload = _load_summary(path)
        row: dict[str, float] = {"threshold": threshold}
        for metric in METRICS:
            item = payload["aggregate"][metric]
            row[metric] = float(item["mean"])
            row[f"{metric}_ci95_low"] = float(item["hierarchical_ci95_low"])
            row[f"{metric}_ci95_high"] = float(item["hierarchical_ci95_high"])
        sensitivity.append(row)
        sources.append(path)

    policies = []
    for variant in ("notool", "forced", "benefit"):
        path = root / "threshold_0p7" / variant / "three_seed_v2.json"
        payload = _load_summary(path)
        row = {"variant": variant}
        for metric in METRICS:
            item = payload["aggregate"][metric]
            row[metric] = float(item["mean"])
            row[f"{metric}_ci95_low"] = float(item["hierarchical_ci95_low"])
            row[f"{metric}_ci95_high"] = float(item["hierarchical_ci95_high"])
        policies.append(row)
        sources.append(path)
    return sensitivity, policies, sources


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(rows: list[dict[str, float]]) -> str:
    lines = [
        "# SN7 Continual Risk-Gate Sensitivity",
        "",
        "| Threshold | Carry - reset | Gated - carry | Gated - reset | Requested | Accepted |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        values = []
        for metric in (
            "carry_minus_independent",
            "risk_gated_minus_carry",
            "risk_gated_minus_independent",
            "requested_intervention_rate",
            "risk_gated_intervention_rate",
        ):
            values.append(
                f"{row[metric]:+.6f} "
                f"[{row[f'{metric}_ci95_low']:+.6f},"
                f"{row[f'{metric}_ci95_high']:+.6f}]"
            )
        lines.append(f"| {row['threshold']:.1f} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Threshold 0.7 is primary; 0.5 and 0.9 are sensitivity analyses.",
            "Validation only; test assets were not read.",
            "",
        ]
    )
    return "\n".join(lines)


def _plot(rows: list[dict[str, float]], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    thresholds = [row["threshold"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    specs = (
        ("risk_gated_minus_carry", "#008F83", "Gated - unconditional carry"),
        ("risk_gated_minus_independent", "#555B61", "Gated - independent reset"),
    )
    for metric, color, label in specs:
        values = [row[metric] for row in rows]
        low = [row[metric] - row[f"{metric}_ci95_low"] for row in rows]
        high = [row[f"{metric}_ci95_high"] - row[metric] for row in rows]
        axes[0].errorbar(
            thresholds,
            values,
            yerr=[low, high],
            marker="o",
            color=color,
            capsize=3,
            linewidth=1.4,
            label=label,
        )
    axes[0].axhline(0.0, color="#222222", linewidth=0.8, linestyle="--")
    axes[0].axvline(0.7, color="#D9DDE0", linewidth=5, alpha=0.45, zorder=0)
    axes[0].set_xlabel("Confidence threshold")
    axes[0].set_ylabel("Mean IoU difference")
    axes[0].set_title("(a) Error-propagation control", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=7.2, loc="lower right")

    for metric, color, label in (
        ("requested_intervention_rate", "#E07A2D", "Requested edits"),
        ("risk_gated_intervention_rate", "#008F83", "Accepted edits"),
    ):
        values = [row[metric] for row in rows]
        low = [row[metric] - row[f"{metric}_ci95_low"] for row in rows]
        high = [row[f"{metric}_ci95_high"] - row[metric] for row in rows]
        axes[1].errorbar(
            thresholds,
            values,
            yerr=[low, high],
            marker="o",
            color=color,
            capsize=3,
            linewidth=1.4,
            label=label,
        )
    axes[1].axvline(0.7, color="#D9DDE0", linewidth=5, alpha=0.45, zorder=0)
    axes[1].set_xlabel("Confidence threshold")
    axes[1].set_ylabel("Intervention rate")
    axes[1].set_title("(b) Persistent edit rate", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=7.2, loc="upper right")

    for ax in axes:
        ax.set_xticks(thresholds)
        ax.grid(axis="y", color="#D9DDE0", linewidth=0.6, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.19, top=0.9, wspace=0.29)

    outputs = []
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"sn7_continual_risk_gate_sensitivity.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return outputs


def export(root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    sensitivity, policies, sources = load(root)
    output_dir.mkdir(parents=True)
    _write_csv(output_dir / "sensitivity.csv", sensitivity)
    _write_csv(output_dir / "primary_policy_comparison.csv", policies)
    (output_dir / "sensitivity.md").write_text(
        _markdown(sensitivity), encoding="utf-8"
    )
    figures = _plot(sensitivity, output_dir)
    manifest = {
        "schema_version": "sn7-continual-paper-artifacts-v1",
        "split": "val",
        "test_assets_read": False,
        "primary_threshold": 0.7,
        "support": {"chains_per_seed": 173, "transitions_per_seed": 411, "aois": 9},
        "sources": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in sources
        ],
        "figures": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in figures
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("chronological_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.chronological_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
