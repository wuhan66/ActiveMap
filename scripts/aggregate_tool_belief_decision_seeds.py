#!/usr/bin/env python3
"""Aggregate independent hierarchical decision-head validation evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

METRICS = (
    "accuracy",
    "macro_f1",
    "false_edit_rate",
    "missed_edit_rate",
    "expected_calibration_error",
)
METHODS = ("identity", "residual_argmax", "hierarchical")
ABLATIONS = ("no_quality", "no_anchor", "no_current")


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be LABEL=/path/to/summary.json")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("run must have a nonempty label and path")
    return label, Path(raw_path)


def _close(left: float, right: float, *, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _validate_reports(runs: list[tuple[str, Path]]) -> list[tuple[str, dict[str, Any]]]:
    if len(runs) < 2:
        raise ValueError("at least two independent seed reports are required")
    labels = [label for label, _ in runs]
    if len(labels) != len(set(labels)):
        raise ValueError("run labels must be unique")

    loaded: list[tuple[str, dict[str, Any]]] = []
    reference: dict[str, Any] | None = None
    for label, path in runs:
        report = json.loads(path.read_text(encoding="utf-8"))
        protocol = report.get("protocol", {})
        if protocol.get("schema_version") != "tool-belief-decision-head-eval-v1":
            raise ValueError(f"{label}: unsupported evaluation schema")
        if protocol.get("split") != "val":
            raise ValueError(f"{label}: only validation reports may be aggregated")
        if protocol.get("test_assets_read") is not False:
            raise ValueError(f"{label}: test_assets_read must be false")
        if not report.get("gates", {}).get("passed", False):
            raise ValueError(f"{label}: report failed its promotion gates")
        for method in (*METHODS, *ABLATIONS):
            if method not in report.get("summaries", {}):
                raise ValueError(f"{label}: missing summary for {method}")
        if reference is None:
            reference = report
        else:
            if protocol.get("sequence_count") != reference["protocol"]["sequence_count"]:
                raise ValueError(f"{label}: sequence count differs across seeds")
            if report["gates"]["thresholds"] != reference["gates"]["thresholds"]:
                raise ValueError(f"{label}: gate thresholds differ across seeds")
            for metric in METRICS:
                current = float(report["summaries"]["identity"][metric])
                expected = float(reference["summaries"]["identity"][metric])
                if not _close(current, expected):
                    raise ValueError(f"{label}: identity {metric} differs across seeds")
        loaded.append((label, report))
    return loaded


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate(runs: list[tuple[str, Path]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reports = _validate_reports(runs)
    seed_rows: list[dict[str, Any]] = []
    for label, report in reports:
        summaries = report["summaries"]
        learned = summaries["hierarchical"]
        identity = summaries["identity"]
        residual = summaries["residual_argmax"]
        row: dict[str, Any] = {
            "seed": label,
            "checkpoint_epoch": report["protocol"]["checkpoint_epoch"],
            "threshold": report["protocol"]["frozen_decision_threshold"],
            "gate_passed": report["gates"]["passed"],
        }
        for method, metrics in (
            ("identity", identity),
            ("residual", residual),
            ("hierarchical", learned),
        ):
            for metric in METRICS:
                row[f"{method}_{metric}"] = float(metrics[metric])
        for metric in METRICS:
            row[f"hierarchical_delta_{metric}"] = float(learned[metric] - identity[metric])
        for ablation in ABLATIONS:
            # Positive contribution means removing this input reduced macro F1.
            row[f"{ablation}_contribution"] = float(
                learned["macro_f1"] - summaries[ablation]["macro_f1"]
            )
        seed_rows.append(row)

    method_stats = {
        method: {
            metric: _stats([row[f"{method}_{metric}"] for row in seed_rows]) for metric in METRICS
        }
        for method in ("identity", "residual", "hierarchical")
    }
    delta_stats = {
        metric: _stats([row[f"hierarchical_delta_{metric}"] for row in seed_rows])
        for metric in METRICS
    }
    ablation_stats = {
        name: _stats([row[f"{name}_contribution"] for row in seed_rows]) for name in ABLATIONS
    }
    aggregate_report = {
        "protocol": {
            "schema_version": "tool-belief-decision-head-aggregate-v1",
            "split": "val",
            "seed_count": len(seed_rows),
            "sequence_count_per_seed": reports[0][1]["protocol"]["sequence_count"],
            "standard_deviation": "sample (ddof=1)",
            "test_assets_read": False,
        },
        "seed_labels": [row["seed"] for row in seed_rows],
        "method_statistics": method_stats,
        "hierarchical_minus_identity": delta_stats,
        "ablation_contributions": ablation_stats,
        "gate_summary": {
            "passed": sum(bool(row["gate_passed"]) for row in seed_rows),
            "total": len(seed_rows),
            "all_passed": all(bool(row["gate_passed"]) for row in seed_rows),
            "thresholds": reports[0][1]["gates"]["thresholds"],
        },
        "claim_boundary": (
            "Validation-only seed replication; thresholds were frozen per seed on validation. "
            "No test assets were read and this is not a held-out test claim."
        ),
    }
    return aggregate_report, seed_rows


def _render(report: dict[str, Any], rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"identity": "#64748b", "residual": "#d97706", "hierarchical": "#2563eb"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    positions = list(range(len(rows)))
    labels = [str(row["seed"])[-2:] for row in rows]

    for offset, method in zip((-0.24, 0.0, 0.24), colors, strict=True):
        axes[0, 0].bar(
            [position + offset for position in positions],
            [row[f"{method}_macro_f1"] for row in rows],
            width=0.22,
            label=method,
            color=colors[method],
        )
    axes[0, 0].set_xticks(positions, labels)
    axes[0, 0].set(title="Validation macro F1 by seed", xlabel="Seed suffix", ylabel="Macro F1")
    axes[0, 0].legend(frameon=False)

    gains = [row["hierarchical_delta_macro_f1"] for row in rows]
    axes[0, 1].bar(positions, gains, color="#2563eb")
    axes[0, 1].axhline(0.01, color="#111827", linestyle=":", label="promotion margin")
    axes[0, 1].set_xticks(positions, labels)
    axes[0, 1].set(
        title="Hierarchical gain over identity", xlabel="Seed suffix", ylabel="Delta macro F1"
    )
    axes[0, 1].legend(frameon=False)

    width = 0.34
    axes[1, 0].bar(
        [position - width / 2 for position in positions],
        [row["hierarchical_false_edit_rate"] for row in rows],
        width=width,
        label="false edit",
        color="#dc2626",
    )
    axes[1, 0].bar(
        [position + width / 2 for position in positions],
        [row["hierarchical_missed_edit_rate"] for row in rows],
        width=width,
        label="missed edit",
        color="#d97706",
    )
    axes[1, 0].set_xticks(positions, labels)
    axes[1, 0].set(title="Conditional safety rates", xlabel="Seed suffix", ylabel="Rate")
    axes[1, 0].legend(frameon=False)

    ablation_labels = [name.removeprefix("no_") for name in ABLATIONS]
    ablation_positions = list(range(len(ABLATIONS)))
    means = [report["ablation_contributions"][name]["mean"] for name in ABLATIONS]
    errors = [report["ablation_contributions"][name]["sample_std"] for name in ABLATIONS]
    axes[1, 1].bar(ablation_positions, means, yerr=errors, capsize=4, color="#059669")
    axes[1, 1].axhline(0.0, color="#111827", linewidth=0.8)
    axes[1, 1].set_xticks(ablation_positions, ablation_labels)
    axes[1, 1].set(title="Input contribution (mean +/- sample std)", ylabel="Macro F1 drop")

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Hierarchical Tool-Belief Decision Head: 3-Seed Validation", fontweight="bold")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_outputs(report: dict[str, Any], rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregate.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _render(report, rows, output_dir / "decision_head_3seed_validation.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, rows = aggregate(args.run)
    write_outputs(report, rows, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
