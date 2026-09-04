#!/usr/bin/env python3
"""Select a functional Agent checkpoint as the base for safety DPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from activemap.agent.result_selection import load_checkpoint_metrics, select_intervention_base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=3,
        required=True,
        metavar=("NAME", "EVALUATION_ROOT", "LABEL"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-acquire-recall", type=float, default=0.05)
    args = parser.parse_args()

    metrics = []
    for name, root, label in args.candidate:
        row = load_checkpoint_metrics(Path(root), label)
        row["label"] = name
        row["evaluation_root"] = root
        row["source_label"] = label
        metrics.append(row)
    decision = select_intervention_base(
        metrics, min_acquire_recall=args.min_acquire_recall
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
