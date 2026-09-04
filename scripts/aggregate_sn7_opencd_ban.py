#!/usr/bin/env python3
"""Aggregate protocol-matched Open-CD BAN runs on frozen SN7 validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

METRIC_KEYS = (
    "prior_map_iou",
    "committed_map_iou",
    "map_iou_delta",
    "edited_map_iou",
    "change_iou",
    "operation_accuracy",
    "keep_false_change_fraction",
)
PROTOCOL_KEYS = (
    "source_commit",
    "manifest",
    "manifest_sha256",
    "train_count",
    "validation_count",
    "input_contract",
    "writeback",
    "selection_metric",
    "image_size",
    "batch_size",
    "positive_class_weight",
    "max_translation_pixels",
    "lovasz_weight",
    "clip_checkpoint_sha256",
    "side_checkpoint_sha256",
    "test_assets_read",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def aggregate_runs(run_dirs: list[Path]) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("at least two runs are required")
    runs: list[dict[str, Any]] = []
    reference_protocol: dict[str, Any] | None = None
    for run_dir in run_dirs:
        summary = _read_json(run_dir / "summary.json")
        history = [
            json.loads(line)
            for line in (run_dir / "history.jsonl").read_text().splitlines()
            if line.strip()
        ]
        protocol = {key: summary.get(key) for key in PROTOCOL_KEYS}
        if protocol["test_assets_read"] is not False:
            raise ValueError(f"run is not test-free: {run_dir}")
        if reference_protocol is None:
            reference_protocol = protocol
        elif protocol != reference_protocol:
            raise ValueError(f"protocol mismatch: {run_dir}")
        best_epoch = int(summary["best_epoch"])
        matching = [row for row in history if int(row["epoch"]) == best_epoch]
        if len(matching) != 1:
            raise ValueError(f"invalid best epoch in {run_dir}")
        metrics = matching[0]["val"]
        runs.append(
            {
                "run_dir": str(run_dir),
                "seed": summary["seed"],
                "best_epoch": best_epoch,
                "metrics": {key: metrics[key] for key in METRIC_KEYS},
            }
        )
    assert reference_protocol is not None
    aggregate: dict[str, dict[str, float | None]] = {}
    for key in METRIC_KEYS:
        values = [run["metrics"][key] for run in runs]
        if all(value is None for value in values):
            aggregate[key] = {"mean": None, "std": None}
        elif any(value is None for value in values):
            raise ValueError(f"partially missing metric: {key}")
        else:
            numeric = np.asarray(values, dtype=np.float64)
            aggregate[key] = {
                "mean": float(numeric.mean()),
                "std": float(numeric.std(ddof=1)),
            }
    return {
        "schema_version": "sn7-opencd-ban-three-seed-aggregate-v1",
        "run_count": len(runs),
        "protocol": reference_protocol,
        "runs": runs,
        "aggregate": aggregate,
    }


def _markdown(result: dict[str, Any]) -> str:
    rows = ["| Metric | Mean | Std |", "| --- | ---: | ---: |"]
    for key in METRIC_KEYS:
        metric = result["aggregate"][key]
        if metric["mean"] is None:
            rows.append(f"| {key} | n/a | n/a |")
        else:
            rows.append(
                f"| {key} | {metric['mean']:.6f} | {metric['std']:.6f} |"
            )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_json", type=Path)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    result = aggregate_runs(args.run_dirs)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_markdown is not None:
        args.output_markdown.write_text(_markdown(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
