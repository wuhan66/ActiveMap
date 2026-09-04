#!/usr/bin/env python3
"""Build a compact table from audited MUNO21 v12 threshold writebacks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


COMPARISONS = (
    ("raw_vs_edit", "qwen_raw_vs_edit_raw.json"),
    ("safe_vs_edit", "qwen_safe_vs_edit_safe.json"),
    ("safe_vs_raw", "qwen_safe_vs_qwen_raw.json"),
)
METRICS = (
    "raster_iou_gain_auc",
    "episode_utility_v2_balanced_auc",
    "episode_utility_v2_safety_auc",
    "false_edit_auc",
    "missed_edit_auc",
)


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("test_assets_read") is not False:
        raise ValueError(f"result is not validation-only: {path}")
    if len(data.get("model_seeds", [])) != 4:
        raise ValueError(f"expected four model seeds: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("threshold007", type=Path)
    parser.add_argument("threshold009", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = []
    for threshold, root in (("0.07", args.threshold007), ("0.09", args.threshold009)):
        for comparison, filename in COMPARISONS:
            intervals = _load(root / filename)["candidate_minus_seed_matched_sft"]
            for metric in METRICS:
                value = intervals[metric]
                rows.append(
                    {
                        "threshold": threshold,
                        "comparison": comparison,
                        "metric": metric,
                        "delta": value["observed_delta"],
                        "ci95_low": value["ci95_low"],
                        "ci95_high": value["ci95_high"],
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "threshold_sensitivity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    by_key = {(row["threshold"], row["comparison"], row["metric"]): row for row in rows}
    lines = [
        "# MUNO21 v12 proactive-threshold sensitivity",
        "",
        "| Threshold | Comparison | Raster IoU | Balanced-U | Safety-U | False edit | Missed edit |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for threshold in ("0.07", "0.09"):
        for comparison, _ in COMPARISONS:
            values = [
                by_key[(threshold, comparison, metric)]["delta"] for metric in METRICS
            ]
            lines.append(
                f"| {threshold} | {comparison} | "
                + " | ".join(f"{value:+.6f}" for value in values)
                + " |"
            )
    lines.extend(
        [
            "",
            "All values are paired four-seed validation deltas. "
            "Confidence intervals remain in `threshold_sensitivity.csv`.",
        ]
    )
    (args.output / "threshold_sensitivity.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
