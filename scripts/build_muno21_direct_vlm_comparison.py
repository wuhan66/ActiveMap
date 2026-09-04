#!/usr/bin/env python3
"""Build paper-ready Direct-VLM prompt-strength comparison assets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _budget_row(summary: dict[str, Any], budget: float) -> dict[str, Any]:
    rows = summary.get("budgets", summary.get("results", []))
    return next(row for row in rows if float(row["budget"]) == budget)


def _run_row(label: str, root: Path, budget: float = 3.0) -> dict[str, Any]:
    evaluation = json.loads(
        (root / "evaluation/summary.json").read_text(encoding="utf-8")
    )
    safe = json.loads(
        (root / "writeback/safe_delta/summary.json").read_text(encoding="utf-8")
    )
    operation = evaluation["operation_metrics"]
    writeback = _budget_row(safe, budget)
    counts = evaluation["prediction_counts"]
    return {
        "method": label,
        "prompt_version": evaluation["protocol"].get("prompt_version", "minimal_v1"),
        "demonstrations": evaluation["protocol"].get("demonstration_count", 0),
        "accuracy": operation["accuracy"],
        "macro_f1": operation["macro_f1"],
        "false_edit_rate": evaluation["false_edit_rate"],
        "missed_edit_rate": evaluation["missed_edit_rate"],
        "schema_valid_rate": evaluation["schema_valid_rate"],
        "pred_keep": counts.get("KEEP", 0),
        "pred_add": counts.get("ADD", 0),
        "pred_delete": counts.get("DELETE", 0),
        "pred_reshape": counts.get("RESHAPE", 0),
        "safe_delta_raster_iou_gain": writeback["mean_raster_iou_gain"],
        "safe_delta_utility_balanced": writeback[
            "mean_episode_utility_v2_balanced"
        ],
        "budget": budget,
        "split": evaluation["protocol"]["split"],
        "test_assets_read": evaluation["test_assets_read"],
    }


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "method",
        "demonstrations",
        "accuracy",
        "macro_f1",
        "false_edit_rate",
        "missed_edit_rate",
        "safe_delta_raster_iou_gain",
    )
    labels = ("Method", "Shots", "Acc.", "Macro-F1", "False edit", "Missed edit", "Raster-IoU gain")
    lines = [
        "| " + " | ".join(labels) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=ROOT")
    parser.add_argument("--budget", type=float, default=3.0)
    args = parser.parse_args()

    rows = []
    for value in args.run:
        label, root = value.split("=", 1)
        rows.append(_run_row(label, Path(root), args.budget))
    if any(row["split"] != "val" or row["test_assets_read"] for row in rows):
        raise ValueError("Direct-VLM comparison must be validation-only")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "table.json").write_text(
        json.dumps(
            {
                "schema_version": "muno21-direct-vlm-comparison-v1",
                "rows": rows,
                "split": "val",
                "test_assets_read": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(args.output_dir / "table.md", rows)
    print(json.dumps({"rows": len(rows), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
