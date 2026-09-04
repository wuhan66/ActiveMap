#!/usr/bin/env python3
"""Summarize paired closed-loop diagnostics for VLA-RL reward scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRICS = (
    "terminal_accuracy",
    "false_edit_rate",
    "missed_edit_rate",
    "wrong_edit_rate",
    "mean_acquisitions",
    "mean_steps",
    "mean_cost",
    "mean_quality_gain",
    "mean_quality_cost_utility",
    "mean_episode_utility_v2_proxy_balanced",
    "valid_action_rate",
    "fallback_episode_rate",
)


def read_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "active-catalog-closed-loop-evaluation-v1":
        raise ValueError(f"unexpected closed-loop schema: {path}")
    if value.get("split") != "val" or value.get("test_assets_read") is not False:
        raise ValueError(f"invalid validation isolation: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sft_summary", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "SUMMARY"))
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    if not args.candidate:
        raise ValueError("at least one candidate is required")

    sft = read_summary(args.sft_summary)["metrics"]
    rows: dict[str, Any] = {}
    for name, raw_path in args.candidate:
        metrics = read_summary(Path(raw_path))["metrics"]
        rows[name] = {
            "metrics": {key: float(metrics[key]) for key in METRICS},
            "delta_vs_sft": {
                key: float(metrics[key]) - float(sft[key]) for key in METRICS
            },
        }
    result = {
        "schema_version": "sn7-vla-rl-reward-scale-closed-loop-v1",
        "sample_count": 256,
        "reference": "sft",
        "sft_metrics": {key: float(sft[key]) for key in METRICS},
        "candidates": rows,
        "test_assets_read": False,
    }
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if args.output_markdown:
        lines = [
            "| Method | Acquisitions | Terminal acc. | False edit | "
            "Balanced utility | Utility vs SFT |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, row in rows.items():
            metrics = row["metrics"]
            delta = row["delta_vs_sft"]
            lines.append(
                f"| {name} | {metrics['mean_acquisitions']:.5f} | "
                f"{metrics['terminal_accuracy']:.5f} | "
                f"{metrics['false_edit_rate']:.5f} | "
                f"{metrics['mean_episode_utility_v2_proxy_balanced']:.5f} | "
                f"{delta['mean_episode_utility_v2_proxy_balanced']:+.5f} |"
            )
        args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
