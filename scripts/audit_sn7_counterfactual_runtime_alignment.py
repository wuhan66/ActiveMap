#!/usr/bin/env python3
"""Audit SN7 counterfactual-candidate versus fused-runtime alignment.

The validation-only audit reuses the immutable join produced by the operation
support audit. It compares the selected candidate's executable counterfactual
gain with the actual gain after multi-evidence fusion and typed writeback.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.audit_sn7_operation_support_layers import Bundle, load_bundle

OPERATIONS = ("ADD", "DELETE", "RESHAPE")


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(right) != len(left):
        return None
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale <= 1e-12 or right_scale <= 1e-12:
        return None
    return float(numerator / (left_scale * right_scale))


def _sign(value: float, epsilon: float) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def summarize(rows: list[dict[str, Any]], *, epsilon: float) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    expected = [float(row["selected_expected_raster_gain"]) for row in rows]
    actual = [float(row["raw_raster_iou_gain"]) for row in rows]
    expected_sign = [_sign(value, epsilon) for value in expected]
    actual_sign = [_sign(value, epsilon) for value in actual]
    expected_positive = [index for index, value in enumerate(expected_sign) if value > 0]
    exact_oracle = [row for row in rows if bool(row["exact_oracle_candidate"])]
    return {
        "count": len(rows),
        "source_episode_count": len({str(row["source_episode"]) for row in rows}),
        "aoi_count": len({str(row["aoi_id"]) for row in rows}),
        "mean_expected_raster_gain": _mean(expected),
        "mean_actual_raster_gain": _mean(actual),
        "mean_actual_minus_expected": _mean(
            [observed - predicted for predicted, observed in zip(expected, actual, strict=True)]
        ),
        "mean_absolute_error": _mean(
            [
                abs(observed - predicted)
                for predicted, observed in zip(expected, actual, strict=True)
            ]
        ),
        "pearson_correlation": _pearson(expected, actual),
        "sign_agreement_rate": _mean(
            [
                float(predicted == observed)
                for predicted, observed in zip(expected_sign, actual_sign, strict=True)
            ]
        ),
        "expected_positive_count": len(expected_positive),
        "expected_positive_runtime_positive_rate": _mean(
            [float(actual_sign[index] > 0) for index in expected_positive]
        ),
        "expected_positive_runtime_harmful_rate": _mean(
            [float(actual_sign[index] < 0) for index in expected_positive]
        ),
        "exact_oracle_count": len(exact_oracle),
        "exact_oracle_runtime_positive_rate": _mean(
            [float(_sign(float(row["raw_raster_iou_gain"]), epsilon) > 0) for row in exact_oracle]
        ),
        "exact_oracle_mean_actual_gain": _mean(
            [float(row["raw_raster_iou_gain"]) for row in exact_oracle]
        ),
    }


def audit(bundles: list[Bundle], *, epsilon: float) -> dict[str, Any]:
    if len({bundle.seed for bundle in bundles}) != len(bundles):
        raise ValueError("bundle seeds must be unique")
    rows: list[dict[str, Any]] = []
    receipts = []
    for bundle in bundles:
        bundle_rows, receipt = load_bundle(bundle)
        rows.extend(
            row
            for row in bundle_rows
            if row["target_operation"] in OPERATIONS and row["policy_acquire"]
        )
        receipts.append(receipt)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[("all", str(row["target_operation"]))].append(row)
        groups[(str(row["seed"]), str(row["target_operation"]))].append(row)

    summaries = []
    for (seed, operation), group_rows in sorted(groups.items()):
        summaries.append(
            {
                "seed": seed,
                "operation": operation,
                **summarize(group_rows, epsilon=epsilon),
            }
        )
    return {
        "schema_version": "sn7-counterfactual-runtime-alignment-v1",
        "split": "val",
        "test_assets_read": False,
        "epsilon": epsilon,
        "seed_count": len(bundles),
        "acquired_nonkeep_state_count": len(rows),
        "groups": summaries,
        "input_receipts": receipts,
        "interpretation": (
            "The expected value is the selected single candidate's immutable "
            "executable raster gain over the direct anchor. The actual value is "
            "the gain after policy-selected multi-evidence fusion and runtime writeback."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("alignment audit has no acquired non-KEEP operation groups")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--bundle", action="append", type=lambda value: _parse_bundle(value), required=True
    )
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")
    result = audit(args.bundle, epsilon=args.epsilon)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "alignment_by_operation.csv", result["groups"])
    print(json.dumps(result, indent=2))


def _parse_bundle(value: str) -> Bundle:
    parts = value.split(":", 4)
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "bundle must be seed:states:rollout:raw_writeback:safe_writeback"
        )
    seed, *paths = parts
    bundle = Bundle(int(seed), *(Path(path) for path in paths))
    for path in (bundle.states, bundle.rollout, bundle.raw_writeback, bundle.safe_writeback):
        if not path.is_file():
            raise argparse.ArgumentTypeError(f"bundle input does not exist: {path}")
    return bundle


if __name__ == "__main__":
    main()
