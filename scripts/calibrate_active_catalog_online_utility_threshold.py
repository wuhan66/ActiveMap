#!/usr/bin/env python3
"""Calibrate an online-state VLA utility threshold using train labels only."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_index(path: Path) -> dict[tuple[str, float, int], dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train" or row.get("model_visible") is not False:
                raise ValueError("threshold calibration requires hidden train index")
            if row.get("test_assets_read") is not False:
                raise ValueError("threshold calibration index reports test access")
            key = (
                str(row["source_episode"]),
                float(row["budget"]),
                int(row["oracle_step"]),
            )
            if key in rows:
                raise ValueError(f"duplicate train index key: {key}")
            rows[key] = row
    return rows


def calibration_records(
    traces_path: Path, index_path: Path
) -> list[dict[str, Any]]:
    index = load_index(index_path)
    records = []
    with traces_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            trace = json.loads(line)
            if trace.get("split") != "train" or trace.get("test_assets_read") is not False:
                raise ValueError("threshold calibration requires train traces")
            match = re.search(r"__s(\d+)$", str(trace["sample_id"]))
            if match is None:
                raise ValueError(f"trace sample ID lacks oracle step: {trace['sample_id']}")
            key = (
                str(trace["source_episode"]),
                float(trace["budget"]),
                int(match.group(1)),
            )
            target = index.get(key)
            if target is None:
                raise ValueError(f"trace has no train evaluation row: {key}")
            event = trace["events"][0]
            ranker_scores = event.get("ranker_scores")
            if not ranker_scores:
                raise ValueError(f"trace lacks ranker scores: {trace['sample_id']}")
            selected_id = max(ranker_scores, key=ranker_scores.get)
            candidates = {
                str(candidate["evidence_id"]): candidate
                for candidate in target["candidates"]
            }
            if selected_id not in candidates:
                raise ValueError(f"ranker/index candidate mismatch: {selected_id}")
            candidate = candidates[selected_id]
            records.append(
                {
                    "sample_id": str(trace["sample_id"]),
                    "aoi_id": str(trace["aoi_id"]),
                    "predicted_utility": float(event["predicted_acquire_utility"]),
                    "selected_evidence_id": selected_id,
                    "realized_gain": float(candidate["utility"])
                    - float(target["stop_utility"]),
                    "cost": float(candidate["cost"]),
                }
            )
    if not records:
        raise ValueError("empty threshold calibration records")
    return records


def threshold_metrics(
    records: list[dict[str, Any]], threshold: float
) -> dict[str, float]:
    called = [row for row in records if row["predicted_utility"] >= threshold]
    positive = [row for row in called if row["realized_gain"] > 0]
    count = len(records)
    return {
        "threshold": threshold,
        "calls": float(len(called)),
        "call_rate": len(called) / count,
        "precision": len(positive) / max(len(called), 1),
        "false_call_rate": (len(called) - len(positive)) / count,
        "realized_utility_mean": sum(row["realized_gain"] for row in called)
        / count,
        "mean_evidence_cost": sum(row["cost"] for row in called) / count,
    }


def select_threshold(
    records: list[dict[str, Any]],
    *,
    maximum_false_call_rate: float,
    maximum_call_rate: float,
) -> tuple[dict[str, float] | None, list[dict[str, float]]]:
    scores = sorted(set(float(row["predicted_utility"]) for row in records))
    thresholds = [scores[0] - 1e-9, *scores, scores[-1] + 1e-9]
    frontier = [threshold_metrics(records, threshold) for threshold in thresholds]
    feasible = [
        row
        for row in frontier
        if row["calls"] > 0
        and row["realized_utility_mean"] > 0
        and row["false_call_rate"] <= maximum_false_call_rate
        and row["call_rate"] <= maximum_call_rate
    ]
    if not feasible:
        return None, frontier
    selected = max(
        feasible,
        key=lambda row: (
            row["realized_utility_mean"],
            row["precision"],
            -row["mean_evidence_cost"],
            row["threshold"],
        ),
    )
    return selected, frontier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_traces", type=Path)
    parser.add_argument("train_evaluation_index", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--maximum-false-call-rate", type=float, default=0.02)
    parser.add_argument("--maximum-call-rate", type=float, default=0.15)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if not 0 <= args.maximum_false_call_rate < 1:
        raise ValueError("maximum false-call rate must be in [0, 1)")
    if not 0 < args.maximum_call_rate <= 1:
        raise ValueError("maximum call rate must be in (0, 1]")

    records = calibration_records(args.train_traces, args.train_evaluation_index)
    selected, frontier = select_threshold(
        records,
        maximum_false_call_rate=args.maximum_false_call_rate,
        maximum_call_rate=args.maximum_call_rate,
    )
    summary = {
        "schema_version": "active-catalog-online-utility-calibration-v1",
        "selection_protocol": "train-only current-policy online-state calibration",
        "train_states": len(records),
        "aoi_count": len({row["aoi_id"] for row in records}),
        "constraints": {
            "maximum_false_call_rate": args.maximum_false_call_rate,
            "maximum_call_rate": args.maximum_call_rate,
        },
        "selected": selected,
        "promotion_authorized": selected is not None,
        "score_range": {
            "minimum": min(row["predicted_utility"] for row in records),
            "maximum": max(row["predicted_utility"] for row in records),
        },
        "test_assets_read": False,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "frontier.json").write_text(
        json.dumps(frontier, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
