#!/usr/bin/env python3
"""Fail-closed audit for grounded Tool-to-Belief supervision."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from activemap.agent.tool_belief_data import ToolBeliefExample
from activemap.agent.tool_features import encode_tool_result
from activemap.geo_tools.records import GeoToolName
from activemap.models import EditOperation


def _read(path: Path, expected_split: str) -> list[ToolBeliefExample]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = ToolBeliefExample.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc
            if row.split != expected_split:
                raise ValueError(
                    f"{path}:{line_number} has split={row.split}, expected {expected_split}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"empty Tool-to-Belief file: {path}")
    return rows


def _belief_deltas(row: ToolBeliefExample) -> tuple[float, float, float]:
    probability = math.fsum(
        abs(after - before)
        for before, after in zip(
            row.prior_belief.edit_probabilities,
            row.target_belief.edit_probabilities,
            strict=True,
        )
    )
    confidence = abs(row.target_belief.confidence - row.prior_belief.confidence)
    geometry = math.fsum(
        abs(after - before)
        for before, after in zip(
            row.prior_belief.geometry_delta,
            row.target_belief.geometry_delta,
            strict=True,
        )
    ) / len(row.prior_belief.geometry_delta)
    return probability, confidence, geometry


def audit(
    train_path: Path,
    val_path: Path,
    *,
    expected_train: int | None,
    expected_val: int | None,
    minimum_success_rate: float,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    by_split = {"train": _read(train_path, "train"), "val": _read(val_path, "val")}
    rows = by_split["train"] + by_split["val"]
    failures: list[str] = []
    warnings: list[str] = []

    expected_counts = {"train": expected_train, "val": expected_val}
    for split, expected in expected_counts.items():
        if expected is not None and len(by_split[split]) != expected:
            failures.append(
                f"{split} record count is {len(by_split[split])}, expected {expected}"
            )

    record_counts = Counter(row.record_id for row in rows)
    duplicate_ids = sorted(key for key, count in record_counts.items() if count > 1)
    if duplicate_ids:
        failures.append(f"duplicate record IDs: {len(duplicate_ids)}")
    train_episodes = {row.episode_id for row in by_split["train"]}
    val_episodes = {row.episode_id for row in by_split["val"]}
    overlap = train_episodes & val_episodes
    if overlap:
        failures.append(f"train/val episode leakage: {len(overlap)} episodes")

    expected_operations = {operation.value for operation in EditOperation}
    operation_counts: dict[str, dict[str, int]] = {}
    for split, split_rows in by_split.items():
        counts = Counter(row.gt_edit.value for row in split_rows)
        operation_counts[split] = dict(sorted(counts.items()))
        missing = expected_operations - set(counts)
        if missing:
            failures.append(f"{split} is missing operations: {sorted(missing)}")

    success_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    feature_vectors: dict[tuple[str, str], set[tuple[float, ...]]] = defaultdict(set)
    target_kind_counts: Counter[str] = Counter()
    noop_deltas: list[tuple[float, float, float]] = []
    semantic_deltas: list[tuple[float, float, float]] = []
    nonfinite_features = 0
    contract_violations = 0
    privacy_violations = 0
    expected_kind = {
        GeoToolName.IMAGE_QUALITY: "observational_noop",
        GeoToolName.TEMPORAL_CHANGE: "teacher_belief_update",
    }

    for row in rows:
        tool = row.tool_result.tool
        key = (row.split, tool.value)
        success_counts[key]["total"] += 1
        success_counts[key]["success"] += int(row.tool_result.success)
        features = tuple(encode_tool_result(row.tool_result))
        feature_vectors[key].add(features)
        nonfinite_features += int(not np.all(np.isfinite(features)))
        kind = str(row.metadata.get("target_kind", ""))
        target_kind_counts[kind] += 1
        if expected_kind.get(tool) != kind:
            contract_violations += 1
        if row.metadata.get("test_assets_read") is not False or row.tool_result.artifacts:
            privacy_violations += 1
        deltas = _belief_deltas(row)
        if kind == "observational_noop":
            noop_deltas.append(deltas)
        elif kind == "teacher_belief_update":
            semantic_deltas.append(deltas)

    success_rates: dict[str, float] = {}
    feature_unique_counts: dict[str, int] = {}
    for key, counts in sorted(success_counts.items()):
        label = f"{key[0]}:{key[1]}"
        rate = counts["success"] / counts["total"]
        success_rates[label] = rate
        feature_unique_counts[label] = len(feature_vectors[key])
        if rate < minimum_success_rate:
            failures.append(
                f"{label} success rate {rate:.6f} is below {minimum_success_rate:.6f}"
            )
        if len(feature_vectors[key]) < 2:
            failures.append(f"{label} has fewer than two unique encoded tool results")

    if nonfinite_features:
        failures.append(f"non-finite encoded feature rows: {nonfinite_features}")
    if contract_violations:
        failures.append(f"tool/target-kind contract violations: {contract_violations}")
    if privacy_violations:
        failures.append(f"test/artifact isolation violations: {privacy_violations}")
    if not noop_deltas:
        failures.append("no observational no-op records")
    if not semantic_deltas:
        failures.append("no semantic teacher-update records")

    noop_array = np.asarray(noop_deltas or [(math.inf, math.inf, math.inf)])
    semantic_array = np.asarray(semantic_deltas or [(0.0, 0.0, 0.0)])
    noop_max = noop_array.max(axis=0)
    semantic_probability = semantic_array[:, 0]
    if np.any(noop_max > tolerance):
        failures.append(
            "observational targets change belief: "
            f"probability={noop_max[0]:.3g}, confidence={noop_max[1]:.3g}, "
            f"geometry={noop_max[2]:.3g}"
        )
    if not np.all(np.isfinite(semantic_array)):
        failures.append("semantic teacher deltas contain non-finite values")
    elif float(np.max(semantic_probability)) <= tolerance:
        failures.append("semantic teacher targets never change edit probabilities")
    elif float(np.mean(semantic_probability > tolerance)) < 0.05:
        warnings.append("fewer than 5% of semantic targets change edit probabilities")

    report: dict[str, Any] = {
        "schema_version": "tool-belief-audit-v1",
        "passed": not failures,
        "record_counts": {split: len(items) for split, items in by_split.items()},
        "episode_counts": {
            "train": len(train_episodes),
            "val": len(val_episodes),
            "overlap": len(overlap),
        },
        "unique_record_count": len(record_counts),
        "operation_counts": operation_counts,
        "target_kind_counts": dict(sorted(target_kind_counts.items())),
        "success_rates": success_rates,
        "feature_unique_counts": feature_unique_counts,
        "noop_delta_max": {
            "probability_l1": float(noop_max[0]),
            "confidence_absolute": float(noop_max[1]),
            "geometry_mae": float(noop_max[2]),
        },
        "semantic_probability_delta": {
            "minimum": float(np.min(semantic_probability)),
            "maximum": float(np.max(semantic_probability)),
            "mean": float(np.mean(semantic_probability)),
            "nonzero_fraction": float(np.mean(semantic_probability > tolerance)),
        },
        "test_assets_read": False,
        "failures": failures,
        "warnings": warnings,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("--expected-train", type=int, default=5088)
    parser.add_argument("--expected-val", type=int, default=828)
    parser.add_argument("--minimum-success-rate", type=float, default=0.99)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_success_rate <= 1.0:
        raise ValueError("minimum-success-rate must be between zero and one")
    report = audit(
        args.train_jsonl,
        args.val_jsonl,
        expected_train=args.expected_train,
        expected_val=args.expected_val,
        minimum_success_rate=args.minimum_success_rate,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
