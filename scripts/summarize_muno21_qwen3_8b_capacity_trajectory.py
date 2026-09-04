#!/usr/bin/env python3
"""Summarize static action quality across Qwen3-8B SFT checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    sources = [
        (
            str(step),
            args.run
            / "evaluation/capacity_trajectory_v1"
            / f"checkpoint-{step}/actions/summary.json",
        )
        for step in (100, 200, 300, 400)
    ]
    sources.append(
        ("final", args.run / "evaluation/capacity_static_v1/actions/summary.json")
    )
    rows = []
    for label, path in sources:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("test_assets_read") is not False:
            raise ValueError(f"non-validation result: {path}")
        tool = data["tool_metrics"]
        rows.append(
            {
                "checkpoint": label,
                "sample_count": data["sample_count"],
                "schema_valid_rate": data["schema_valid_rate"],
                "executable_valid_rate": data["executable_valid_rate"],
                "exact_action_accuracy": data["exact_action_accuracy"],
                "macro_f1": data["macro_f1"],
                "predicted_call_count": tool["predicted_call_count"],
                "target_call_count": tool["target_call_count"],
                "false_call_rate": tool["false_call_rate"],
                "grounded_call_accuracy": tool["grounded_call_accuracy"],
                "tool_positive_exact_accuracy": tool[
                    "tool_positive_exact_accuracy"
                ],
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "capacity_trajectory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# MUNO21 Qwen3-8B capacity trajectory",
        "",
        "| Checkpoint | Exact action | Macro-F1 | Calls predicted/target | False-call | Grounded-call | Tool-positive exact |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['checkpoint']} | {row['exact_action_accuracy']:.6f} | "
            f"{row['macro_f1']:.6f} | {row['predicted_call_count']}/"
            f"{row['target_call_count']} | {row['false_call_rate']:.6f} | "
            f"{row['grounded_call_accuracy']:.6f} | "
            f"{row['tool_positive_exact_accuracy']:.6f} |"
        )
    (args.output / "capacity_trajectory.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
