#!/usr/bin/env python3
"""Aggregate frozen post-acquisition Tool-to-Belief seed summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

METRICS = (
    "baseline_deployed_accuracy",
    "baseline_deployed_macro_f1",
    "baseline_deployed_false_edit_rate",
    "baseline_deployed_missed_edit_rate",
    "baseline_deployed_expected_calibration_error",
    "zero_tool_accuracy",
    "zero_tool_macro_f1",
    "zero_tool_false_edit_rate",
    "zero_tool_missed_edit_rate",
    "zero_tool_expected_calibration_error",
    "full_accuracy",
    "full_macro_f1",
    "full_false_edit_rate",
    "full_missed_edit_rate",
    "full_expected_calibration_error",
)

FROZEN_KEYS = (
    "protocol",
    "train_examples",
    "val_examples",
    "train_tasks",
    "val_tasks",
    "batch_size",
    "learning_rate",
    "hidden_dim",
    "dropout",
    "max_logit_delta",
    "geometry_scale",
    "safety_margin",
    "min_quality_delta",
    "min_tool_delta",
    "max_calibration_degradation",
    "loss_weights",
    "tool_contrastive_margin",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two seed summaries are required")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    seeds = [int(row["seed"]) for row in payloads]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed summaries must have unique seeds")
    reference = {key: payloads[0].get(key) for key in FROZEN_KEYS}
    for row in payloads[1:]:
        current = {key: row.get(key) for key in FROZEN_KEYS}
        if current != reference:
            raise ValueError("seed summaries do not share a frozen protocol")

    per_seed = []
    selected_metrics = []
    for path, row in sorted(
        zip(paths, payloads, strict=True), key=lambda item: int(item[1]["seed"])
    ):
        promoted = bool(row["promotion_passed"])
        metrics = (
            row["best_promoted_val_metrics"] if promoted else row["best_val_metrics"]
        )
        epoch = row["best_promoted_epoch"] if promoted else row["best_epoch"]
        if not isinstance(metrics, dict) or epoch is None:
            raise ValueError(f"summary has no selectable validation metrics: {path}")
        missing = [name for name in METRICS if name not in metrics]
        if missing:
            raise ValueError(f"summary is missing metrics {missing}: {path}")
        selected = {name: float(metrics[name]) for name in METRICS}
        selected_metrics.append(selected)
        per_seed.append(
            {
                "seed": int(row["seed"]),
                "promotion_passed": promoted,
                "selection_kind": "best_promoted" if promoted else "best_non_promoted",
                "selected_epoch": int(epoch),
                "metrics": selected,
                "deltas": {
                    "full_minus_deployed_macro_f1": selected["full_macro_f1"]
                    - selected["baseline_deployed_macro_f1"],
                    "full_minus_zero_tool_macro_f1": selected["full_macro_f1"]
                    - selected["zero_tool_macro_f1"],
                    "full_minus_deployed_false_edit_rate": selected[
                        "full_false_edit_rate"
                    ]
                    - selected["baseline_deployed_false_edit_rate"],
                },
                "source": str(path.resolve()),
                "sha256": _sha256(path),
            }
        )

    aggregate_metrics = {
        name: _mean_std([row[name] for row in selected_metrics]) for name in METRICS
    }
    delta_names = tuple(per_seed[0]["deltas"])
    aggregate_deltas = {
        name: _mean_std([float(row["deltas"][name]) for row in per_seed])
        for name in delta_names
    }
    promoted_count = sum(row["promotion_passed"] for row in per_seed)
    return {
        "schema_version": "post-acquisition-tool-belief-seed-aggregate-v1",
        "seed_count": len(per_seed),
        "promoted_seed_count": promoted_count,
        "promotion_rate": promoted_count / len(per_seed),
        "all_seeds_promoted": promoted_count == len(per_seed),
        "paper_claim_ready": promoted_count == len(per_seed),
        "frozen_protocol": reference,
        "per_seed": per_seed,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_deltas": aggregate_deltas,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = aggregate(args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
