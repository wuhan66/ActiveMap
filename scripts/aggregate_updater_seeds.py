#!/usr/bin/env python3
"""Aggregate validation-gated updater decisions across random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from activemap.evaluation.updater_seed_aggregate import aggregate_seed_decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decision",
        action="append",
        required=True,
        metavar="SEED=PATH",
        help="repeat for each seed",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decisions = {}
    for item in args.decision:
        seed, separator, raw_path = item.partition("=")
        if not separator or not seed or not raw_path:
            parser.error(f"invalid --decision value: {item}")
        decisions[seed] = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    summary = aggregate_seed_decisions(decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
