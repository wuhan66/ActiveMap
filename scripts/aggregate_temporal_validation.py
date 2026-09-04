#!/usr/bin/env python3
"""Aggregate standard temporal-updater validation reports across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from activemap.evaluation.updater_temporal_promotion import (
    aggregate_temporal_validation_evaluations,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", action="append", required=True, metavar="SEED=JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluations = {}
    for item in args.evaluation:
        seed, separator, raw_path = item.partition("=")
        if not separator or not seed or not raw_path:
            parser.error(f"invalid --evaluation value: {item}")
        evaluations[seed] = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    summary = aggregate_temporal_validation_evaluations(evaluations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
