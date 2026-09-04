#!/usr/bin/env python3
"""Aggregate Agent validation results and apply frozen promotion gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from activemap.agent.result_selection import load_checkpoint_metrics, select_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--labels",
        default="checkpoint-200,checkpoint-400,checkpoint-418",
    )
    parser.add_argument("--min-schema-valid", type=float, default=0.99)
    parser.add_argument("--min-executable-valid", type=float, default=0.99)
    parser.add_argument("--min-acquire-recall", type=float, default=0.05)
    parser.add_argument("--max-false-edit", type=float, default=0.05)
    parser.add_argument("--max-fallback", type=float, default=0.01)
    parser.add_argument("--min-utility-delta-vs-greedy", type=float, default=0.0)
    args = parser.parse_args()

    labels = [value.strip() for value in args.labels.split(",") if value.strip()]
    available = [
        label
        for label in labels
        if (args.evaluation_root / label / "actions" / "summary.json").is_file()
        and (args.evaluation_root / label / "rollouts" / "summary.json").is_file()
    ]
    if not available:
        raise SystemExit("no complete checkpoint evaluations found")
    metrics = [load_checkpoint_metrics(args.evaluation_root, label) for label in available]
    decision = select_checkpoint(
        metrics,
        min_schema_valid=args.min_schema_valid,
        min_executable_valid=args.min_executable_valid,
        min_acquire_recall=args.min_acquire_recall,
        max_false_edit=args.max_false_edit,
        max_fallback=args.max_fallback,
        min_utility_delta_vs_greedy=args.min_utility_delta_vs_greedy,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "promotion_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    rows = decision["checkpoints"]
    with (args.output_dir / "checkpoint_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
