#!/usr/bin/env python3
"""Freeze a temporal-pair closure margin from train-only rollout summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    rows = []
    for summary_path in sorted(args.root.glob("m*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        results = payload.get("results", [])
        if not results:
            raise ValueError(f"missing results: {summary_path}")
        margin = float(summary_path.parent.name[1:].replace("p", "."))
        rows.append(
            {
                "margin": margin,
                "source": str(summary_path),
                "false_edit_rate": mean(results, "false_edit_rate"),
                "missed_edit_rate": mean(results, "missed_edit_rate"),
                "mean_cost": mean(results, "mean_cost"),
                "balanced_proxy": mean(results, "mean_episode_utility_v2_proxy_balanced"),
                "safety_proxy": mean(results, "mean_episode_utility_v2_proxy_safety"),
            }
        )
    baseline = next((row for row in rows if row["margin"] == 0.0), None)
    if baseline is None:
        raise ValueError("the zero-margin baseline is required")
    feasible = [
        row for row in rows if row["false_edit_rate"] <= baseline["false_edit_rate"] + 1e-12
    ]
    selected = max(
        feasible,
        key=lambda row: (
            row["safety_proxy"],
            row["balanced_proxy"],
            -row["mean_cost"],
            -row["margin"],
        ),
    )
    payload = {
        "schema_version": "muno21-p50-temporal-pair-closure-train-selection-v1",
        "split": "train",
        "test_assets_read": False,
        "rule": "false_edit_rate <= zero_margin; maximize safety_proxy, balanced_proxy, lower cost, lower margin",
        "baseline": baseline,
        "candidates": rows,
        "selected": selected,
    }
    (args.root / "train_screen.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (args.root / "selected_margin.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
