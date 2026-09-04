#!/usr/bin/env python3
"""Freeze a train/dev false-edit cap for constrained C5 selector refresh.

The calibration comparison is intentionally updater-matched: an f0 selector
and an f1 selector are both evaluated on the f1 train/dev cache.  The chosen
cap is then applied once to the untouched f1 validation cache by the standard
executable evaluator.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


EVALUATOR = Path(__file__).with_name("evaluate_updater_conditioned_selector.py")
SPEC = importlib.util.spec_from_file_location("updater_conditioned_evaluator", EVALUATOR)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("calibration evaluation returned no rows")
    return {
        "quality_cost_utility": float(np.mean([float(row["quality_cost_utility"]) for row in rows])),
        "final_raster_iou": float(np.mean([float(row["final_raster_iou"]) for row in rows])),
        "false_edit_rate": float(np.mean([float(row["false_edit"]) for row in rows])),
        "missed_edit_rate": float(np.mean([float(row["missed_edit"]) for row in rows])),
        "mean_additional_cost": float(np.mean([float(row["additional_cost"]) for row in rows])),
        "call_rate": float(np.mean([float(row["called"]) for row in rows])),
        "risk_guard_rate": float(np.mean([float(row["risk_guard_triggered"]) for row in rows])),
    }


def select_cap(
    candidates: list[dict[str, Any]],
    *,
    stale_false_edit_rate: float,
    max_false_edit_increase: float,
) -> dict[str, Any]:
    """Select a predeclared safety-feasible cap from train/dev candidates."""

    if max_false_edit_increase < 0.0:
        raise ValueError("max_false_edit_increase must be non-negative")
    ceiling = stale_false_edit_rate + max_false_edit_increase
    eligible = [
        row
        for row in candidates
        if float(row["metrics"]["false_edit_rate"]) <= ceiling + 1e-12
    ]
    if not eligible:
        raise ValueError("no candidate risk cap satisfies the stale-policy false-edit cap")
    # Missed-edit reduction is C5's intended refresh benefit. Utility breaks
    # ties, then lower false edit and lower acquisition cost retain safety.
    return max(
        eligible,
        key=lambda row: (
            -float(row["metrics"]["missed_edit_rate"]),
            float(row["metrics"]["quality_cost_utility"]),
            float(row["metrics"]["final_raster_iou"]),
            -float(row["metrics"]["false_edit_rate"]),
            -float(row["metrics"]["mean_additional_cost"]),
            float(row["risk_cap"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stale_checkpoint", type=Path)
    parser.add_argument("refreshed_checkpoint", type=Path)
    parser.add_argument("f1_train_dev_states", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--grid-steps", type=int, default=21)
    parser.add_argument("--max-false-edit-increase", type=float, default=0.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.grid_steps < 2:
        raise ValueError("grid_steps must be at least 2")
    if args.max_false_edit_increase < 0.0:
        raise ValueError("max_false_edit_increase must be non-negative")

    device = torch.device(args.device)
    stale_rows = MODULE.evaluate(
        args.stale_checkpoint,
        args.f1_train_dev_states,
        device=device,
        batch_size=args.batch_size,
        split="val",
        max_candidate_risk=None,
    )
    stale_metrics = _metrics(stale_rows)
    refreshed_unguarded = MODULE.evaluate(
        args.refreshed_checkpoint,
        args.f1_train_dev_states,
        device=device,
        batch_size=args.batch_size,
        split="val",
        max_candidate_risk=None,
    )
    candidates: list[dict[str, Any]] = []
    for cap in np.linspace(0.0, 1.0, args.grid_steps):
        rows = MODULE.evaluate(
            args.refreshed_checkpoint,
            args.f1_train_dev_states,
            device=device,
            batch_size=args.batch_size,
            split="val",
            max_candidate_risk=float(cap),
        )
        candidates.append({"risk_cap": float(cap), "metrics": _metrics(rows)})
    output = {
        "schema_version": "updater-conditioned-risk-cap-calibration-v1",
        "selection_split": "f1 train/dev only",
        "stale_checkpoint": str(args.stale_checkpoint.resolve()),
        "refreshed_checkpoint": str(args.refreshed_checkpoint.resolve()),
        "states": str(args.f1_train_dev_states.resolve()),
        "stale_metrics": stale_metrics,
        "refreshed_unguarded_metrics": _metrics(refreshed_unguarded),
        "max_false_edit_increase": args.max_false_edit_increase,
        "false_edit_ceiling": stale_metrics["false_edit_rate"] + args.max_false_edit_increase,
        "grid_steps": args.grid_steps,
        "candidates": candidates,
        "test_assets_read": False,
    }
    try:
        selected = select_cap(
            candidates,
            stale_false_edit_rate=stale_metrics["false_edit_rate"],
            max_false_edit_increase=args.max_false_edit_increase,
        )
    except ValueError as error:
        # Preserve the full train/dev sweep when the strict safety constraint
        # is infeasible, while deliberately preventing downstream validation.
        output.update({"status": "infeasible", "failure": str(error)})
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "risk_cap_calibration_infeasible.json").write_text(
            json.dumps(output, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(output, indent=2))
        raise
    output.update({"status": "complete", "selected": selected})
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "risk_cap_calibration.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
