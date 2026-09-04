#!/usr/bin/env python3
"""Offline replay of delta-component pruning for MUNO21 oracle writebacks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy import ndimage


OPERATIONS = ("KEEP", "ADD", "DELETE", "RESHAPE")
SCOPES = ("all", "add_only")
PRUNING_POLICIES = ("plain", "preserve_largest")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _target_operation(row: dict[str, Any]) -> str:
    target = str(row["target"])
    if target == "REJECT":
        return "KEEP"
    if ":" in target:
        return target.rsplit(":", 1)[-1]
    return str(row.get("operation", target))


def _filter_components(
    mask: np.ndarray, minimum_pixels: int, preserve_largest: bool
) -> np.ndarray:
    binary = mask.astype(bool, copy=False)
    if minimum_pixels <= 1 or not binary.any():
        return binary.copy()
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(binary)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_pixels
    keep[0] = False
    filtered = keep[labels]
    if preserve_largest and binary.any() and not filtered.any():
        largest_label = int(np.argmax(sizes[1:]) + 1)
        filtered = labels == largest_label
    return filtered


def _component_count(mask: np.ndarray) -> int:
    _, count = ndimage.label(
        mask.astype(bool, copy=False), structure=np.ones((3, 3), dtype=np.uint8)
    )
    return int(count)


def _iou(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> float:
    left = left & valid
    right = right & valid
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def _effective_operation(prior: np.ndarray, committed: np.ndarray) -> str:
    added = np.logical_and(committed, ~prior).any()
    removed = np.logical_and(prior, ~committed).any()
    if added and removed:
        return "RESHAPE"
    if added:
        return "ADD"
    if removed:
        return "DELETE"
    return "KEEP"


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _paired_bootstrap(
    values: list[float], repetitions: int, rng: random.Random
) -> dict[str, float]:
    estimates = [
        mean(rng.choice(values) for _ in values) for _ in range(repetitions)
    ]
    return {
        "observed": mean(values),
        "ci95_low": _percentile(estimates, 0.025),
        "ci95_high": _percentile(estimates, 0.975),
    }


def _replay(
    row: dict[str, Any],
    minimum_pixels: int,
    scope: str,
    pruning_policy: str,
) -> dict[str, Any]:
    artifact = Path(row["mask_artifact"])
    with np.load(artifact) as arrays:
        committed = arrays["committed_mask"] >= 0.5
        prior = arrays["prior_mask"] >= 0.5
        target = arrays["target_mask"] >= 0.5
        valid = arrays["valid_mask"] >= 0.5

    operation = _target_operation(row)
    should_filter = scope == "all" or operation == "ADD"
    if should_filter and minimum_pixels > 1:
        preserve_largest = pruning_policy == "preserve_largest"
        added = _filter_components(
            committed & ~prior, minimum_pixels, preserve_largest
        )
        removed = _filter_components(
            prior & ~committed, minimum_pixels, preserve_largest
        )
        replayed = (prior | added) & ~removed
    else:
        replayed = committed

    effective = _effective_operation(prior, replayed)
    changed = effective != "KEEP"
    raster_iou = _iou(replayed, target, valid)
    prior_iou = _iou(prior, target, valid)
    target_changed = operation != "KEEP"
    return {
        "task_id": row["task_id"],
        "budget": float(row["budget"]),
        "operation": operation,
        "scope": scope,
        "pruning_policy": pruning_policy,
        "minimum_pixels": minimum_pixels,
        "raster_iou": raster_iou,
        "prior_raster_iou": prior_iou,
        "raster_iou_gain": raster_iou - prior_iou,
        "effective_operation": effective,
        "false_edit": operation == "KEEP" and changed,
        "missed_edit": target_changed and not changed,
        "wrong_edit": target_changed and changed and effective != operation,
        "component_count": _component_count(replayed & valid),
        "target_component_count": _component_count(target & valid),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scope": rows[0]["scope"],
        "pruning_policy": rows[0]["pruning_policy"],
        "minimum_pixels": rows[0]["minimum_pixels"],
        "operation": rows[0]["operation"],
        "budget": rows[0]["budget"],
        "sample_count": len(rows),
        "mean_raster_iou": mean(row["raster_iou"] for row in rows),
        "mean_prior_raster_iou": mean(row["prior_raster_iou"] for row in rows),
        "mean_raster_iou_gain": mean(row["raster_iou_gain"] for row in rows),
        "positive_gain_rate": mean(row["raster_iou_gain"] > 0 for row in rows),
        "negative_gain_rate": mean(row["raster_iou_gain"] < 0 for row in rows),
        "false_edit_rate": mean(row["false_edit"] for row in rows),
        "missed_edit_rate": mean(row["missed_edit"] for row in rows),
        "wrong_edit_rate": mean(row["wrong_edit"] for row in rows),
        "mean_component_count_error": mean(
            abs(row["component_count"] - row["target_component_count"]) for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("writeback_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--thresholds", default="0,4,8,16,32,64,128,256,512,1024"
    )
    parser.add_argument("--scopes", default=",".join(SCOPES))
    parser.add_argument("--policies", default=",".join(PRUNING_POLICIES))
    parser.add_argument("--selection-budget", type=float, default=3.0)
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    thresholds = sorted({int(value) for value in args.thresholds.split(",")})
    if not thresholds or thresholds[0] != 0 or any(value < 0 for value in thresholds):
        raise ValueError("threshold grid must be nonnegative and include 0")
    scopes = tuple(value for value in args.scopes.split(",") if value)
    policies = tuple(value for value in args.policies.split(",") if value)
    if not scopes or any(value not in SCOPES for value in scopes):
        raise ValueError(f"scopes must be selected from {SCOPES}")
    if not policies or any(value not in PRUNING_POLICIES for value in policies):
        raise ValueError(f"policies must be selected from {PRUNING_POLICIES}")
    source_rows = [
        json.loads(line)
        for line in args.writeback_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not source_rows:
        raise ValueError("empty writeback input")
    for row in source_rows:
        if row.get("split") != "val" or row.get("test_assets_read") is not False:
            raise ValueError("component sweep must remain validation-only")
        if not Path(row["mask_artifact"]).is_file():
            raise FileNotFoundError(row["mask_artifact"])

    replayed = [
        _replay(row, threshold, scope, pruning_policy)
        for pruning_policy in policies
        for scope in scopes
        for threshold in thresholds
        for row in source_rows
    ]
    grouped: dict[
        tuple[str, str, int, str, float], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in replayed:
        grouped[
            (
                row["pruning_policy"],
                row["scope"],
                row["minimum_pixels"],
                row["operation"],
                row["budget"],
            )
        ].append(row)
    summaries = [
        _summarize(grouped[key])
        for key in sorted(
            grouped, key=lambda item: (item[0], item[1], item[2], item[3], item[4])
        )
    ]

    selection_rows = [
        row for row in replayed if row["budget"] == args.selection_budget
    ]
    by_policy_scope_threshold: dict[
        tuple[str, str, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in selection_rows:
        by_policy_scope_threshold[
            (row["pruning_policy"], row["scope"], row["minimum_pixels"])
        ].append(row)
    rng = random.Random(args.seed)
    candidates = []
    for pruning_policy in policies:
        for scope in scopes:
            baseline = {
                row["task_id"]: row
                for row in by_policy_scope_threshold[(pruning_policy, scope, 0)]
            }
            for threshold in thresholds:
                rows = by_policy_scope_threshold[(pruning_policy, scope, threshold)]
                deltas = [
                    row["raster_iou_gain"]
                    - baseline[row["task_id"]]["raster_iou_gain"]
                    for row in rows
                ]
                missed_delta = mean(row["missed_edit"] for row in rows) - mean(
                    row["missed_edit"] for row in baseline.values()
                )
                false_delta = mean(row["false_edit"] for row in rows) - mean(
                    row["false_edit"] for row in baseline.values()
                )
                interval = _paired_bootstrap(deltas, args.repetitions, rng)
                candidates.append(
                    {
                        "pruning_policy": pruning_policy,
                        "scope": scope,
                        "minimum_pixels": threshold,
                        "sample_count": len(rows),
                        "mean_raster_iou_gain": mean(
                            row["raster_iou_gain"] for row in rows
                        ),
                        "paired_gain_vs_zero": interval,
                        "missed_edit_rate_delta": missed_delta,
                        "false_edit_rate_delta": false_delta,
                        "eligible": (
                            threshold > 0
                            and interval["ci95_low"] > 0.0
                            and missed_delta <= 0.0
                            and false_delta <= 0.0
                        ),
                    }
                )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda candidate: (
                candidate["paired_gain_vs_zero"]["ci95_low"],
                candidate["mean_raster_iou_gain"],
                -candidate["minimum_pixels"],
            ),
        )
        if eligible
        else None
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = (
            "pruning_policy",
            "scope",
            "minimum_pixels",
            "operation",
            "budget",
            "sample_count",
            "mean_raster_iou_gain",
            "positive_gain_rate",
            "negative_gain_rate",
            "false_edit_rate",
            "missed_edit_rate",
            "wrong_edit_rate",
            "mean_component_count_error",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in summaries)

    lines = [
        "# MUNO21 Oracle Delta-Component Sweep",
        "",
        "| Policy | Scope | Min pixels | Mean gain | Delta vs 0 (95% CI) | Missed delta | Eligible |",
        "|---|---|---:|---:|---:|---:|:---:|",
    ]
    for candidate in candidates:
        interval = candidate["paired_gain_vs_zero"]
        lines.append(
            "| {pruning_policy} | {scope} | {minimum_pixels} | {gain:.5f} | {delta:.5f} "
            "[{low:.5f}, {high:.5f}] | {missed:.5f} | {eligible} |".format(
                pruning_policy=candidate["pruning_policy"],
                scope=candidate["scope"],
                minimum_pixels=candidate["minimum_pixels"],
                gain=candidate["mean_raster_iou_gain"],
                delta=interval["observed"],
                low=interval["ci95_low"],
                high=interval["ci95_high"],
                missed=candidate["missed_edit_rate_delta"],
                eligible="PASS" if candidate["eligible"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            f"Selection: `{selected}`" if selected else "Selection: no promoted threshold.",
            "",
        ]
    )
    (args.output_dir / "sweep.md").write_text("\n".join(lines), encoding="utf-8")
    result = {
        "schema_version": "muno21-oracle-delta-component-sweep-v2",
        "split": "val",
        "record_count": len(source_rows),
        "thresholds": thresholds,
        "scopes": list(scopes),
        "pruning_policies": list(policies),
        "selection_budget": args.selection_budget,
        "bootstrap_repetitions": args.repetitions,
        "candidates": candidates,
        "selected": selected,
        "summaries": summaries,
        "source": {
            "path": str(args.writeback_jsonl),
            "sha256": _sha256(args.writeback_jsonl),
        },
        "test_assets_read": False,
    }
    (args.output_dir / "sweep.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
