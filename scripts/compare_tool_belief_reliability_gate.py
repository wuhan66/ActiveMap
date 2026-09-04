#!/usr/bin/env python3
"""Compare paired ungated/gated Tool-Belief runs before closed-loop promotion."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


CONTROLLED_KEYS = (
    "train_examples",
    "val_examples",
    "epochs",
    "batch_size",
    "learning_rate",
    "hidden_dim",
    "dropout",
    "max_logit_delta",
    "geometry_scale",
    "loss_weights",
    "tool_contrastive_margin",
)


def _load(paths: list[Path], *, gated: bool) -> dict[int, dict[str, Any]]:
    runs = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if bool(payload.get("reliability_gate", False)) is not gated:
            raise ValueError(f"reliability_gate mismatch in {path}")
        seed = int(payload["seed"])
        if seed in runs:
            raise ValueError(f"duplicate seed {seed}")
        runs[seed] = payload
    return runs


def compare(
    ungated_paths: list[Path],
    gated_paths: list[Path],
    *,
    macro_f1_margin: float = 0.005,
    false_edit_margin: float = 0.005,
    ece_margin: float = 0.01,
    minimum_gate_spread: float = 0.02,
) -> dict[str, Any]:
    ungated = _load(ungated_paths, gated=False)
    gated = _load(gated_paths, gated=True)
    if set(ungated) != set(gated) or len(gated) < 3:
        raise ValueError("paired comparison requires the same three or more seeds")
    rows = []
    for seed in sorted(gated):
        left, right = ungated[seed], gated[seed]
        mismatches = [key for key in CONTROLLED_KEYS if left.get(key) != right.get(key)]
        if mismatches:
            raise ValueError(f"seed {seed} differs beyond gate: {mismatches}")
        left_metrics = left.get("best_promoted_val_metrics") or left["best_val_metrics"]
        right_metrics = right.get("best_promoted_val_metrics") or right["best_val_metrics"]
        spread = float(right_metrics["full_reliability_gate_p90"]) - float(
            right_metrics["full_reliability_gate_p10"]
        )
        rows.append(
            {
                "seed": seed,
                "ungated_macro_f1": float(left_metrics["full_macro_f1"]),
                "gated_macro_f1": float(right_metrics["full_macro_f1"]),
                "macro_f1_delta": float(right_metrics["full_macro_f1"])
                - float(left_metrics["full_macro_f1"]),
                "ungated_false_edit_rate": float(left_metrics["full_false_edit_rate"]),
                "gated_false_edit_rate": float(right_metrics["full_false_edit_rate"]),
                "false_edit_rate_delta": float(right_metrics["full_false_edit_rate"])
                - float(left_metrics["full_false_edit_rate"]),
                "ungated_ece": float(left_metrics["full_expected_calibration_error"]),
                "gated_ece": float(right_metrics["full_expected_calibration_error"]),
                "ece_delta": float(right_metrics["full_expected_calibration_error"])
                - float(left_metrics["full_expected_calibration_error"]),
                "gate_mean": float(right_metrics["full_reliability_gate_mean"]),
                "gate_spread_p90_p10": spread,
                "gate_noncollapsed": spread >= minimum_gate_spread,
                "gated_component_promoted": bool(right.get("promotion_passed")),
            }
        )
    mean = lambda key: statistics.fmean(float(row[key]) for row in rows)
    checks = {
        "macro_f1_noninferior": mean("macro_f1_delta") >= -macro_f1_margin,
        "false_edit_noninferior": mean("false_edit_rate_delta") <= false_edit_margin,
        "calibration_noninferior": mean("ece_delta") <= ece_margin,
        "gate_noncollapsed_majority": sum(row["gate_noncollapsed"] for row in rows)
        >= (len(rows) // 2 + 1),
        "component_promoted_majority": sum(
            row["gated_component_promoted"] for row in rows
        )
        >= (len(rows) // 2 + 1),
    }
    checks["passed"] = all(checks.values())
    return {
        "schema_version": "tool-belief-reliability-gate-ablation-v1",
        "paired_seeds": sorted(gated),
        "controlled_difference": "reliability_gate",
        "thresholds": {
            "macro_f1_margin": macro_f1_margin,
            "false_edit_margin": false_edit_margin,
            "ece_margin": ece_margin,
            "minimum_gate_spread": minimum_gate_spread,
        },
        "aggregate_deltas": {
            "macro_f1": mean("macro_f1_delta"),
            "false_edit_rate": mean("false_edit_rate_delta"),
            "expected_calibration_error": mean("ece_delta"),
        },
        "checks": checks,
        "promotion_passed": checks["passed"],
        "next_stage": "closed_loop_quality_cost_safety" if checks["passed"] else None,
        "per_seed": rows,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--ungated", nargs="+", type=Path, required=True)
    parser.add_argument("--gated", nargs="+", type=Path, required=True)
    parser.add_argument("--macro-f1-margin", type=float, default=0.005)
    parser.add_argument("--false-edit-margin", type=float, default=0.005)
    parser.add_argument("--ece-margin", type=float, default=0.01)
    parser.add_argument("--minimum-gate-spread", type=float, default=0.02)
    args = parser.parse_args()
    report = compare(
        args.ungated,
        args.gated,
        macro_f1_margin=args.macro_f1_margin,
        false_edit_margin=args.false_edit_margin,
        ece_margin=args.ece_margin,
        minimum_gate_spread=args.minimum_gate_spread,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
