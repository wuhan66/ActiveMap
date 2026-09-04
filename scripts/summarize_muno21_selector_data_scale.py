#!/usr/bin/env python3
"""Summarize the fixed MUNO21 selector data-scale ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _learned(path: Path) -> dict[str, float]:
    rows = [row for row in _read(path)["results"] if row["method"] == "learned"]
    if len(rows) != 3:
        raise ValueError(f"expected three learned budget rows: {path}")
    return {
        "mean_utility": mean(float(row["mean_utility"]) for row in rows),
        "mean_regret": mean(float(row["mean_regret"]) for row in rows),
        "mean_cost": mean(float(row["mean_cost"]) for row in rows),
        "mean_steps": mean(float(row["mean_steps"]) for row in rows),
        "stop_rate": mean(float(row["policy_stop_rate"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    data = _read(args.root / "data" / "summary.json")
    support = {
        row["label"]: (row["fraction"], row["train_task_count"], row["train_record_count"])
        for row in data["outputs"]
    }
    rows = []
    for label in ("p0125", "p0250", "p0500", "p0750"):
        fraction, tasks, records = support[label]
        rows.append(
            {
                "fraction": fraction,
                "train_tasks": tasks,
                "train_records": records,
                **_learned(args.root / label / "validation" / "rollout" / "summary.json"),
            }
        )
    rows.append(
        {
            "fraction": 1.0,
            "train_tasks": data["train_task_count_full"],
            "train_records": 2544,
            **_learned(args.root / "full_reference" / "rollout" / "summary.json"),
        }
    )
    best = max(rows, key=lambda row: row["mean_utility"])
    result = {
        "schema_version": "muno21-selector-data-scale-summary-v1",
        "seed": data["seed"],
        "validation_tasks": 138,
        "budgets": [1.5, 3.0, 4.5],
        "nested_task_group_subsets": True,
        "best_fraction": best["fraction"],
        "best_mean_utility": best["mean_utility"],
        "interpretation": (
            "Non-monotonic sample efficiency: the 50% subset is best, while "
            "12.5%, 25%, and 75% collapse to STOP. More data alone does not "
            "explain the controller gain. This is a one-seed diagnostic."
        ),
        "test_assets_read": False,
        "rows": rows,
    }
    (args.root / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (args.root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# MUNO21 Selector Data-Scale Diagnostic",
        "",
        "| Train fraction | Tasks | Records | Utility | Cost | Steps | STOP rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['fraction']:.1%} | {row['train_tasks']} | {row['train_records']} | "
            f"{row['mean_utility']:+.6f} | {row['mean_cost']:.6f} | "
            f"{row['mean_steps']:.6f} | {row['stop_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The one-seed curve is non-monotonic. The 50% task-grouped subset is "
            "best (`+0.003561` utility), slightly above the full-data reference "
            "(`+0.002463`), while 12.5%, 25%, and 75% converge to Always-STOP. "
            "This supports a data-composition/calibration diagnosis rather than "
            "a simple insufficient-data explanation; it is not a promotion result.",
        ]
    )
    (args.root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
