#!/usr/bin/env python3
"""Apply a frozen operation-conditioned SN7 Safe Commit gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.calibrate_sn7_operation_safe_commit import (
    EDIT_OPERATIONS,
    apply_operation_safe_commit,
)
from scripts.calibrate_sn7_v5_safe_commit import load_rows, sha256, summarize


def load_calibration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "sn7-operation-safe-commit-calibration-v1":
        raise ValueError("unexpected operation-conditioned calibration schema")
    if (
        payload.get("split") != "train"
        or payload.get("test_assets_read") is not False
        or payload.get("policy_blind") is not True
    ):
        raise ValueError("operation-conditioned calibration must be train-only")
    gate = payload.get("gate")
    if not isinstance(gate, dict):
        raise ValueError("operation-conditioned calibration has no gate")
    thresholds = gate.get("confidence_thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(EDIT_OPERATIONS):
        raise ValueError("operation-conditioned thresholds are incomplete")
    if any(not 0.0 <= float(value) <= 1.0 for value in thresholds.values()):
        raise ValueError("operation-conditioned threshold is outside [0, 1]")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("writebacks", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--policy", choices=("direct", "selected", "forced"), required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    calibration = load_calibration(args.calibration)
    raw = load_rows(args.writebacks, expected_split="val")
    gate = calibration["gate"]
    safe_rows = [
        apply_operation_safe_commit(
            row,
            confidence_thresholds={
                key: float(value)
                for key, value in gate["confidence_thresholds"].items()
            },
            replay_iou_threshold=float(gate["replay_iou_threshold"]),
            require_topology=bool(gate["require_topology"]),
        )
        for _, row in sorted(raw.items())
    ]
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "writeback.jsonl"
    with output.open("x", encoding="utf-8") as handle:
        for row in safe_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    raw_metrics = summarize(list(raw.values()))
    safe_metrics = summarize(safe_rows)
    summary = {
        "schema_version": "sn7-operation-safe-commit-writeback-v1",
        "policy": args.policy,
        "split": "val",
        "raw_writebacks": str(args.writebacks.resolve()),
        "raw_writebacks_sha256": sha256(args.writebacks),
        "calibration": str(args.calibration.resolve()),
        "calibration_sha256": sha256(args.calibration),
        "gate": gate,
        "raw_metrics": raw_metrics,
        "safe_commit_metrics": safe_metrics,
        "safe_minus_raw": {
            name: safe_metrics[name] - raw_metrics[name] for name in safe_metrics
        },
        "rejected_writeback_count": sum(
            not row["safe_commit_accepted"] for row in safe_rows
        ),
        "row_count": len(safe_rows),
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
