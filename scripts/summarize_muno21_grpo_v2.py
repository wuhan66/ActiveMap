#!/usr/bin/env python3
"""Aggregate GRPO variants and render a paper-ready safety/utility heatmap."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.compare_agent_rollouts import _metrics


METRICS = (
    "terminal_accuracy",
    "macro_f1",
    "false_edit_rate",
    "missed_edit_rate",
    "mean_cost",
    "mean_acquisitions",
    "mean_tool_calls",
    "episode_utility_v2_proxy_balanced_auc",
    "episode_utility_v2_proxy_safety_auc",
    "episode_utility_v2_proxy_cost_aware_auc",
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty rollout file: {path}")
    if any(row.get("split") == "test" for row in rows):
        raise ValueError("test rollouts are forbidden in GRPO model selection")
    return rows


def macro_f1(rows: list[dict[str, Any]]) -> float:
    labels = sorted({str(row["target"]) for row in rows} | {str(row["prediction"]) for row in rows})
    scores = []
    for label in labels:
        tp = sum(row["target"] == label and row["prediction"] == label for row in rows)
        fp = sum(row["target"] != label and row["prediction"] == label for row in rows)
        fn = sum(row["target"] == label and row["prediction"] != label for row in rows)
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


def metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    result = _metrics(rows)
    result["macro_f1"] = macro_f1(rows)
    return {key: float(result[key]) for key in METRICS}


def parse_run(value: str) -> tuple[str, str, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--run requires VARIANT=SEED=JSONL")
    return parts[0], parts[1], Path(parts[2])


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    baseline = metrics(read_rows(args.baseline))
    per_run = []
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for variant, seed, path in args.run:
        values = metrics(read_rows(path))
        grouped[variant].append(values)
        per_run.append({"variant": variant, "seed": seed, **values})

    aggregate = []
    for variant, runs in grouped.items():
        row: dict[str, Any] = {"variant": variant, "seeds": len(runs)}
        for metric in METRICS:
            values = np.asarray([run[metric] for run in runs])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_delta_vs_sft"] = float(values.mean() - baseline[metric])
        aggregate.append(row)

    write_csv(args.output / "per_run.csv", per_run, ["variant", "seed", *METRICS])
    aggregate_fields = ["variant", "seeds"] + [
        f"{metric}_{suffix}"
        for metric in METRICS
        for suffix in ("mean", "std", "delta_vs_sft")
    ]
    write_csv(args.output / "aggregate.csv", aggregate, aggregate_fields)
    payload = {
        "schema_version": "activemap-grpo-v2-validation-summary-v1",
        "protocol": {
            "split": "val",
            "test_assets_read": False,
            "baseline": str(args.baseline),
            "aggregation": "mean and sample standard deviation across policy seeds",
        },
        "baseline_sft": baseline,
        "per_run": per_run,
        "aggregate": aggregate,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    table_metrics = (
        "terminal_accuracy", "macro_f1", "false_edit_rate", "missed_edit_rate",
        "mean_tool_calls", "episode_utility_v2_proxy_balanced_auc",
    )
    lines = [
        "| Variant | Seeds | Accuracy | Macro-F1 | False edit | Missed edit | Tool calls | Balanced utility |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        cells = [str(row["variant"]), str(row["seeds"])]
        for metric in table_metrics:
            cells.append(f'{row[f"{metric}_mean"]:.4f} +/- {row[f"{metric}_std"]:.4f}')
        lines.append("| " + " | ".join(cells) + " |")
    (args.output / "table.md").write_text("\n".join(lines) + "\n")

    import matplotlib.pyplot as plt

    display_metrics = (
        "terminal_accuracy", "macro_f1", "false_edit_rate", "missed_edit_rate",
        "mean_cost", "episode_utility_v2_proxy_balanced_auc",
    )
    display_names = ("Accuracy", "Macro-F1", "False edit", "Missed edit", "Cost", "Utility")
    signs = np.asarray([1, 1, -1, -1, -1, 1], dtype=float)
    heat = np.asarray([
        [row[f"{metric}_delta_vs_sft"] for metric in display_metrics]
        for row in aggregate
    ]) * signs
    bound = max(0.01, float(np.abs(heat).max()))
    fig, ax = plt.subplots(figsize=(9.0, max(2.4, 0.7 * len(aggregate) + 1.4)))
    image = ax.imshow(heat, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(len(display_names)), display_names)
    ax.set_yticks(range(len(aggregate)), [row["variant"] for row in aggregate])
    for row_index in range(heat.shape[0]):
        for column_index in range(heat.shape[1]):
            ax.text(column_index, row_index, f"{heat[row_index, column_index]:+.3f}",
                    ha="center", va="center", fontsize=9)
    ax.set_title("GRPO v2 improvement over SFT (higher is better)")
    fig.colorbar(image, ax=ax, label="direction-corrected delta")
    fig.tight_layout()
    fig.savefig(args.output / "grpo_v2_delta_heatmap.png", dpi=240)
    fig.savefig(args.output / "grpo_v2_delta_heatmap.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
