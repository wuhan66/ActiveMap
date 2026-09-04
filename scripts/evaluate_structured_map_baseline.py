#!/usr/bin/env python3
"""Evaluate an external object-level editable-map baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from activemap.evaluation.structured_map import evaluate_structured_map_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite metrics: {args.output}")
    metrics = evaluate_structured_map_predictions(
        args.samples,
        args.predictions,
        allow_test=args.allow_test,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
