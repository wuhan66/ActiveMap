#!/usr/bin/env python3
"""Calibrate a completed hierarchy run and enforce its validation promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from activemap.evaluation.updater_promotion import finalize_hierarchical_updater


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-false-edit", type=float, default=0.05)
    parser.add_argument("--grid-steps", type=int, default=33)
    parser.add_argument("--baseline-macro-f1", type=float, default=0.789307)
    parser.add_argument("--baseline-delete-f1", type=float, default=0.365979)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="calibrate and record promotion without reading the test split",
    )
    args = parser.parse_args()
    decision = finalize_hierarchical_updater(
        args.run_dir,
        args.samples,
        device=args.device,
        batch_size=args.batch_size,
        max_false_edit=args.max_false_edit,
        grid_steps=args.grid_steps,
        baseline_macro_f1=args.baseline_macro_f1,
        baseline_delete_f1=args.baseline_delete_f1,
        bootstrap_iterations=args.bootstrap,
        evaluate_test=not args.validation_only,
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
