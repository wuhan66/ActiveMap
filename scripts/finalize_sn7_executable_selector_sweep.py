#!/usr/bin/env python3
"""Freeze the winning executable-supervision selector architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.summarize_selector_sweep import summarize
except ModuleNotFoundError:  # Direct execution adds scripts/ to sys.path.
    from summarize_selector_sweep import summarize


def finalize(
    root: Path,
    *,
    min_utility: float = 0.0,
    max_false_call_rate: float = 0.02,
    max_harmful_call_fraction: float = 0.30,
) -> dict[str, Any]:
    process_path = root / "process_results.json"
    if not process_path.is_file():
        raise RuntimeError("sweep is incomplete: process_results.json is missing")
    processes = json.loads(process_path.read_text(encoding="utf-8"))
    if len(processes) != 4 or any(int(row["returncode"]) != 0 for row in processes):
        raise RuntimeError("all four sweep jobs must finish successfully")
    rows = summarize(root)
    for row in rows:
        row["passes_gate"] = bool(
            row["eligible"]
            and row["mean_chosen_utility"] > min_utility
            and row["false_call_rate"] <= max_false_call_rate
            and row["harmful_call_fraction"] <= max_harmful_call_fraction
        )
    feasible = [row for row in rows if row["passes_gate"]]
    winner = (
        max(
            feasible,
            key=lambda row: (
                row["mean_chosen_utility"],
                -row["false_call_rate"],
                row["exact_acquire_recall"],
            ),
        )
        if feasible
        else None
    )
    return {
        "schema_version": "sn7-executable-selector-sweep-decision-v1",
        "selection_metric": "validation mean_chosen_utility",
        "constraints": {
            "eligible": True,
            "mean_chosen_utility_above": min_utility,
            "false_call_rate_at_most": max_false_call_rate,
            "harmful_call_fraction_at_most": max_harmful_call_fraction,
        },
        "promoted": winner is not None,
        "winner": winner,
        "rows": sorted(rows, key=lambda row: row["run"]),
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-utility", type=float, default=0.0)
    parser.add_argument("--max-false-call-rate", type=float, default=0.02)
    parser.add_argument("--max-harmful-call-fraction", type=float, default=0.30)
    args = parser.parse_args()
    decision = finalize(
        args.root,
        min_utility=args.min_utility,
        max_false_call_rate=args.max_false_call_rate,
        max_harmful_call_fraction=args.max_harmful_call_fraction,
    )
    text = json.dumps(decision, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
