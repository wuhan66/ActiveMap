#!/usr/bin/env python3
"""Aggregate paired spatial/no-spatial Tool-Belief validation runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_pair(
    seed: int, spatial: dict[str, Any], no_spatial: dict[str, Any]
) -> dict[str, Any]:
    supported_protocols = {
        "spatial_hierarchical_decision_head_v1_independent_eval",
        "spatial_hierarchical_decision_head_v2_gated_residual_independent_eval",
    }
    for report in (spatial, no_spatial):
        if report.get("protocol") not in supported_protocols:
            raise ValueError(f"seed {seed}: unexpected protocol")
        if report.get("split") != "val" or report.get("test_assets_read") is not False:
            raise ValueError(f"seed {seed}: aggregation requires validation-only reports")
        if float(report.get("maximum_causal_future_mass", -1.0)) != 0.0:
            raise ValueError(f"seed {seed}: causal future evidence leakage detected")
    if spatial.get("protocol") != no_spatial.get("protocol"):
        raise ValueError(f"seed {seed}: paired protocols differ")
    if spatial.get("checkpoint_uses_spatial") is not True:
        raise ValueError(f"seed {seed}: spatial checkpoint disabled spatial evidence")
    if no_spatial.get("checkpoint_uses_spatial") is not False:
        raise ValueError(f"seed {seed}: no-spatial checkpoint enabled spatial evidence")
    if no_spatial.get("no_spatial_masking_invariance") is not True:
        raise ValueError(f"seed {seed}: no-spatial masking invariance failed")
    if int(no_spatial.get("prediction_changes_when_masked", -1)) != 0:
        raise ValueError(f"seed {seed}: no-spatial predictions depend on spatial input")
    if float(no_spatial.get("maximum_update_probability_change_when_masked", -1.0)) != 0.0:
        raise ValueError(f"seed {seed}: no-spatial probabilities depend on spatial input")
    if spatial.get("example_count") != no_spatial.get("example_count"):
        raise ValueError(f"seed {seed}: paired example counts differ")
    spatial_final = spatial["spatial_by_stage"]["3"]
    no_spatial_final = no_spatial["spatial_by_stage"]["3"]
    identity_spatial = spatial["identity_by_stage"]["3"]
    identity_no_spatial = no_spatial["identity_by_stage"]["3"]
    if identity_spatial != identity_no_spatial:
        raise ValueError(f"seed {seed}: paired identity references differ")
    return {
        "seed": seed,
        "protocol": str(spatial["protocol"]),
        "example_count": int(spatial["example_count"]),
        "spatial_macro_f1": float(spatial_final["macro_f1"]),
        "no_spatial_macro_f1": float(no_spatial_final["macro_f1"]),
        "macro_f1_gain": float(spatial_final["macro_f1"])
        - float(no_spatial_final["macro_f1"]),
        "spatial_false_edit_rate": float(spatial_final["false_edit_rate"]),
        "no_spatial_false_edit_rate": float(no_spatial_final["false_edit_rate"]),
        "false_edit_delta": float(spatial_final["false_edit_rate"])
        - float(no_spatial_final["false_edit_rate"]),
        "spatial_missed_edit_rate": float(spatial_final["missed_edit_rate"]),
        "no_spatial_missed_edit_rate": float(no_spatial_final["missed_edit_rate"]),
        "missed_edit_delta": float(spatial_final["missed_edit_rate"])
        - float(no_spatial_final["missed_edit_rate"]),
        "spatial_passed": bool(spatial["passed"]),
        "no_spatial_passed": bool(no_spatial["passed"]),
    }


def aggregate(
    pairs: list[tuple[int, Path, Path]],
    *,
    minimum_mean_gain: float,
    safety_margin: float,
) -> dict[str, Any]:
    rows = [
        _validate_pair(seed, _load(spatial), _load(no_spatial))
        for seed, spatial, no_spatial in pairs
    ]
    if len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("duplicate seed")
    gains = [row["macro_f1_gain"] for row in rows]
    false_deltas = [row["false_edit_delta"] for row in rows]
    missed_deltas = [row["missed_edit_delta"] for row in rows]
    mean_gain = statistics.fmean(gains)
    report = {
        "protocol": "paired_spatial_no_spatial_v1_multiseed",
        "seed_count": len(rows),
        "seeds": [row["seed"] for row in rows],
        "rows": rows,
        "mean_macro_f1_gain": mean_gain,
        "population_std_macro_f1_gain": statistics.pstdev(gains),
        "spatial_win_count": sum(gain > 0.0 for gain in gains),
        "mean_false_edit_delta": statistics.fmean(false_deltas),
        "mean_missed_edit_delta": statistics.fmean(missed_deltas),
        "all_component_gates_passed": all(
            row["spatial_passed"] and row["no_spatial_passed"] for row in rows
        ),
        "promotion_gate": {
            "minimum_mean_gain": minimum_mean_gain,
            "safety_margin": safety_margin,
            "all_seeds_positive": all(gain > 0.0 for gain in gains),
            "mean_gain_passed": mean_gain >= minimum_mean_gain,
            "false_edit_passed": statistics.fmean(false_deltas) <= safety_margin,
            "missed_edit_passed": statistics.fmean(missed_deltas) <= safety_margin,
        },
        "test_assets_read": False,
    }
    report["passed"] = report["all_component_gates_passed"] and all(
        report["promotion_gate"][name]
        for name in (
            "all_seeds_positive",
            "mean_gain_passed",
            "false_edit_passed",
            "missed_edit_passed",
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        metavar="SEED=SPATIAL_SUMMARY,NO_SPATIAL_SUMMARY",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-mean-gain", type=float, default=0.01)
    parser.add_argument("--safety-margin", type=float, default=0.02)
    args = parser.parse_args()
    pairs = []
    for item in args.pair:
        raw_seed, separator, raw_paths = item.partition("=")
        spatial, comma, no_spatial = raw_paths.partition(",")
        if not separator or not comma or not spatial or not no_spatial:
            parser.error(f"invalid --pair value: {item}")
        pairs.append((int(raw_seed), Path(spatial), Path(no_spatial)))
    report = aggregate(
        pairs,
        minimum_mean_gain=args.minimum_mean_gain,
        safety_margin=args.safety_margin,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
