#!/usr/bin/env python3
"""Measure train-only confidence separability for executable DELETE proposals."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from scripts.calibrate_sn7_v5_safe_commit import (
    load_rows,
    sha256,
    target_operation,
)


def _auc(scores: list[float], labels: list[bool]) -> float | None:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        return None
    ranked = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    rank_sum = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][0] == ranked[start][0]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ranked[start:end])
        start = end
    mann_whitney_u = rank_sum - positive_count * (positive_count + 1) / 2.0
    return mann_whitney_u / (positive_count * negative_count)


def _operation(value: Any) -> str:
    return target_operation(str(value)).value


def parse_record(value: str) -> tuple[str, Path]:
    """Parse a uniquely named, train-only writeback record for this audit."""

    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("record must be NAME=WRITEBACK_JSONL")
    return name, Path(raw_path)


def compact_delete_rows(
    records: dict[str, dict[tuple[str, float], dict[str, Any]]],
    *,
    gain_epsilon: float,
) -> list[dict[str, Any]]:
    rows = []
    for policy, policy_rows in records.items():
        for row in policy_rows.values():
            if not bool(row["writeback_changed"]) or str(row["effective_operation"]) != "DELETE":
                continue
            target = _operation(row["target"])
            gain = float(row["raster_iou_gain"])
            beneficial = target == "DELETE" and gain > gain_epsilon
            harmful = target != "DELETE" or gain < -gain_epsilon
            rows.append(
                {
                    "policy": policy,
                    "task_id": str(row["task_id"]),
                    "aoi_id": str(row["aoi_id"]),
                    "budget": float(row["budget"]),
                    "target_operation": target,
                    "confidence": float(row["fused_confidence"]),
                    "raster_iou_gain": gain,
                    "beneficial_delete": beneficial,
                    "harmful_delete": harmful,
                    "target_keep_false_edit": target == "KEEP",
                }
            )
    if not rows:
        raise ValueError("no changed DELETE proposals found")
    return rows


def _iter_threshold_points(
    ranked: list[dict[str, Any]],
    thresholds: list[float],
    *,
    beneficial_count: int,
    harmful_count: int,
    total_keep_false_edits: int,
) -> Iterator[dict[str, float]]:
    """Yield cumulative acceptance statistics for descending thresholds."""

    accepted_count = 0
    accepted_beneficial = 0
    accepted_harmful = 0
    accepted_keep_false_edits = 0
    next_index = 0
    for threshold in thresholds:
        while next_index < len(ranked) and float(ranked[next_index]["confidence"]) >= threshold:
            accepted = ranked[next_index]
            accepted_count += 1
            accepted_beneficial += int(bool(accepted["beneficial_delete"]))
            accepted_harmful += int(bool(accepted["harmful_delete"]))
            accepted_keep_false_edits += int(bool(accepted["target_keep_false_edit"]))
            next_index += 1
        yield {
            "confidence_threshold": threshold,
            "beneficial_delete_recall": accepted_beneficial / max(beneficial_count, 1),
            "harmful_accept_rate": accepted_harmful / max(harmful_count, 1),
            "target_keep_false_edit_rate": (
                accepted_keep_false_edits / max(total_keep_false_edits, 1)
            ),
            "accept_rate": accepted_count / len(ranked),
        }


def _exact_thresholds(ranked: list[dict[str, Any]]) -> Iterator[float]:
    """Yield each possible gate threshold without materializing a large curve."""

    previous: float | None = None
    if not ranked or float(ranked[0]["confidence"]) < 1.0:
        yield 1.0
    for row in ranked:
        confidence = float(row["confidence"])
        if confidence != previous:
            yield confidence
            previous = confidence
    if previous is None or previous > 0.0:
        yield 0.0


def summarize(rows: list[dict[str, Any]], *, maximum_harmful_accept_rate: float) -> dict[str, Any]:
    positives = [row for row in rows if row["beneficial_delete"]]
    harmful = [row for row in rows if row["harmful_delete"]]
    ranked = sorted(rows, key=lambda row: float(row["confidence"]), reverse=True)
    total_keep_false_edits = sum(row["target_keep_false_edit"] for row in rows)
    point_kwargs = {
        "beneficial_count": len(positives),
        "harmful_count": len(harmful),
        "total_keep_false_edits": total_keep_false_edits,
    }
    selected: dict[str, float] | None = None
    for point in _iter_threshold_points(ranked, list(_exact_thresholds(ranked)), **point_kwargs):
        if point["harmful_accept_rate"] > maximum_harmful_accept_rate + 1e-12:
            continue
        if selected is None or (
            point["beneficial_delete_recall"],
            -point["harmful_accept_rate"],
            -point["confidence_threshold"],
        ) > (
            selected["beneficial_delete_recall"],
            -selected["harmful_accept_rate"],
            -selected["confidence_threshold"],
        ):
            selected = point
    if selected is None:
        raise RuntimeError("no threshold satisfies the harmful-acceptance constraint")
    reporting_thresholds = [index / 100.0 for index in range(100, -1, -1)]
    reporting_points = list(_iter_threshold_points(ranked, reporting_thresholds, **point_kwargs))
    curve = list(reversed(reporting_points))
    return {
        "proposal_count": len(rows),
        "aoi_count": len({str(row["aoi_id"]) for row in rows}),
        "beneficial_delete_count": len(positives),
        "harmful_delete_count": len(harmful),
        "confidence_auroc_beneficial_vs_rest": _auc(
            [float(row["confidence"]) for row in rows],
            [bool(row["beneficial_delete"]) for row in rows],
        ),
        "confidence_auroc_beneficial_vs_harmful": _auc(
            [
                float(row["confidence"])
                for row in rows
                if row["beneficial_delete"] or row["harmful_delete"]
            ],
            [
                bool(row["beneficial_delete"])
                for row in rows
                if row["beneficial_delete"] or row["harmful_delete"]
            ],
        ),
        "maximum_harmful_accept_rate": maximum_harmful_accept_rate,
        "best_operating_point": selected,
        "threshold_curve": curve,
        "threshold_curve_resolution": "fixed_0.01_grid",
    }


def audit(
    records: dict[str, dict[tuple[str, float], dict[str, Any]]],
    *,
    gain_epsilon: float,
    maximum_harmful_accept_rate: float,
) -> dict[str, Any]:
    rows = compact_delete_rows(records, gain_epsilon=gain_epsilon)
    groups = {"pooled": rows}
    groups.update({policy: [row for row in rows if row["policy"] == policy] for policy in records})
    return {
        "schema_version": "sn7-delete-confidence-separability-v1",
        "split": "train",
        "test_assets_read": False,
        "gain_epsilon": gain_epsilon,
        "groups": {
            name: summarize(
                group_rows,
                maximum_harmful_accept_rate=maximum_harmful_accept_rate,
            )
            for name, group_rows in groups.items()
        },
    }


def _write_curve(path: Path, result: dict[str, Any]) -> None:
    rows = [
        {"group": name, **point}
        for name, summary in result["groups"].items()
        for point in summary["threshold_curve"]
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--record", action="append", type=parse_record, required=True)
    parser.add_argument("--gain-epsilon", type=float, default=1e-6)
    parser.add_argument("--maximum-harmful-accept-rate", type=float, default=0.01)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.gain_epsilon < 0.0 or not 0.0 <= args.maximum_harmful_accept_rate <= 1.0:
        raise ValueError("invalid audit thresholds")
    paths = dict(args.record)
    if len(paths) != len(args.record):
        raise ValueError("duplicate policy record")
    records = {policy: load_rows(path, expected_split="train") for policy, path in paths.items()}
    result = audit(
        records,
        gain_epsilon=args.gain_epsilon,
        maximum_harmful_accept_rate=args.maximum_harmful_accept_rate,
    )
    result["inputs"] = {
        policy: {"path": str(path.resolve()), "sha256": sha256(path)}
        for policy, path in paths.items()
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _write_curve(args.output_dir / "threshold_curve.csv", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
