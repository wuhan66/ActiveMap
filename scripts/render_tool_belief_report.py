#!/usr/bin/env python3
"""Render training, safety, and intervention diagnostics for Tool-to-Belief."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _history(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty training history: {path}")
    return rows


def intervention_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = report["summaries"]
    if "paired_3" in summaries:
        names = (
            "identity",
            "paired_1",
            "paired_2",
            "paired_3",
            "no_quality_3",
            "no_current_3",
            "teacher_3",
        )
    else:
        names = (
            "identity_one_step",
            "learned_one_step",
            "teacher_one_step",
            "learned_no_current_one_step",
            "learned_temporal_3",
            "learned_quality_3",
            "learned_quality_temporal_3",
        )
    rows = []
    for name in names:
        metrics = summaries[name]
        rows.append(
            {
                "method": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "false_edit_rate": metrics["false_edit_rate"],
                "missed_edit_rate": metrics["missed_edit_rate"],
                "mean_joint_utility": metrics["mean_joint_utility"],
                "mean_tool_cost": metrics["mean_tool_cost"],
                "expected_calibration_error": metrics["expected_calibration_error"],
            }
        )
    return rows


def render(history_path: Path, intervention_path: Path, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = _history(history_path)
    report = json.loads(intervention_path.read_text(encoding="utf-8"))
    methods = intervention_rows(report)
    identity_metrics = next(row for row in methods if row["method"].startswith("identity"))
    epochs = [row["epoch"] for row in history]
    best_row = max(
        history, key=lambda row: row.get("selection_score", -row["val"]["loss"])
    )
    recurrent = "step3_macro_f1" in history[0]["val"]
    macro_f1_key = "step3_macro_f1" if recurrent else "macro_f1"
    false_edit_key = "step3_false_edit_rate" if recurrent else "false_edit_rate"
    missed_edit_key = "step3_missed_edit_rate" if recurrent else "missed_edit_rate"
    noop_keys = (
        ("noop_probability", "noop_confidence", "noop_geometry")
        if recurrent
        else (
            "noop_probability_drift",
            "noop_confidence_drift",
            "noop_geometry_drift",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    colors = {"train": "#2563eb", "val": "#dc2626", "baseline": "#64748b"}

    axes[0, 0].plot(epochs, [row["train"]["loss"] for row in history], label="train")
    axes[0, 0].plot(epochs, [row["val"]["loss"] for row in history], label="validation")
    axes[0, 0].set(title="Training objective", xlabel="Epoch", ylabel="Loss")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(
        epochs,
        [row["val"][macro_f1_key] for row in history],
        label="learned",
        color=colors["train"],
    )
    axes[0, 1].plot(
        epochs,
        [identity_metrics["macro_f1"]] * len(history),
        label="identity",
        color=colors["baseline"],
    )
    axes[0, 1].axvline(best_row["epoch"], color="#111827", linestyle=":", linewidth=1)
    axes[0, 1].set(title="Semantic operation quality", xlabel="Epoch", ylabel="Macro F1")
    axes[0, 1].legend(frameon=False)

    for metric, label, color in (
        (false_edit_key, "false edit", "#dc2626"),
        (missed_edit_key, "missed edit", "#d97706"),
    ):
        axes[0, 2].plot(
            epochs,
            [row["val"][metric] for row in history],
            label=label,
            color=color,
        )
    for metric, color in (
        ("false_edit_rate", "#dc2626"),
        ("missed_edit_rate", "#d97706"),
    ):
        axes[0, 2].axhline(
            identity_metrics[metric], color=color, linestyle=":", linewidth=1
        )
    axes[0, 2].set(title="Validation safety", xlabel="Epoch", ylabel="Conditional rate")
    axes[0, 2].legend(frameon=False)

    for metric, label, color in zip(
        noop_keys,
        ("probability", "confidence", "geometry"),
        ("#2563eb", "#059669", "#7c3aed"),
        strict=True,
    ):
        axes[1, 0].plot(
            epochs,
            [row["val"][metric] for row in history],
            label=label,
            color=color,
        )
    axes[1, 0].set(title="Quality no-op drift", xlabel="Epoch", ylabel="Mean drift")
    axes[1, 0].legend(frameon=False)

    labels = [
        row["method"].replace("learned_", "").replace("_one_step", "")
        for row in methods
    ]
    positions = list(range(len(methods)))
    axes[1, 1].bar(
        [value - 0.18 for value in positions],
        [row["macro_f1"] for row in methods],
        width=0.36,
        label="macro F1",
        color="#2563eb",
    )
    axes[1, 1].bar(
        [value + 0.18 for value in positions],
        [row["accuracy"] for row in methods],
        width=0.36,
        label="accuracy",
        color="#059669",
    )
    axes[1, 1].set_xticks(positions, labels, rotation=35, ha="right")
    axes[1, 1].set(title="Intervention outcomes", ylabel="Score", ylim=(0.0, 1.0))
    axes[1, 1].legend(frameon=False)

    axes[1, 2].bar(
        positions,
        [row["mean_joint_utility"] for row in methods],
        color=["#64748b" if "identity" in row["method"] else "#d97706" for row in methods],
    )
    axes[1, 2].axhline(0.0, color="#111827", linewidth=0.8)
    axes[1, 2].set_xticks(positions, labels, rotation=35, ha="right")
    axes[1, 2].set(title="Utility after tool cost", ylabel="Mean joint utility")

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Grounded Tool-to-Belief Validation",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output_dir / "tool_belief_report.png", dpi=180)
    plt.close(figure)

    with (output_dir / "intervention_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(methods[0]))
        writer.writeheader()
        writer.writerows(methods)
    (output_dir / "report_summary.json").write_text(
        json.dumps(
            {
                "epochs": len(history),
                "best_epoch": best_row["epoch"],
                "intervention_gates": report["gates"],
                "quality_noop_delta_max": report.get(
                    "quality_public_belief_delta_max",
                    report.get("quality_noop_delta_max"),
                ),
                "test_assets_read": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("history_jsonl", type=Path)
    parser.add_argument("intervention_summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    render(args.history_jsonl, args.intervention_summary, args.output_dir)


if __name__ == "__main__":
    main()
