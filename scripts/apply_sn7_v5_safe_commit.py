#!/usr/bin/env python3
"""Apply a frozen V5 Safe Commit gate to one validation writeback file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.calibrate_sn7_v5_safe_commit import (
    apply_safe_commit,
    load_rows,
    sha256,
    summarize,
)


def load_calibration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "sn7-v5-safe-commit-calibration-v1":
        raise ValueError("unexpected V5 Safe Commit calibration schema")
    if payload.get("split") != "train" or payload.get("test_assets_read") is not False:
        raise ValueError("Safe Commit calibration must be train-only")
    gate = payload.get("gate")
    if not isinstance(gate, dict):
        raise ValueError("Safe Commit calibration has no gate")
    threshold = float(gate.get("confidence_threshold", -1.0))
    replay = float(gate.get("replay_iou_threshold", -1.0))
    if not 0.0 <= threshold <= 1.0 or not 0.0 <= replay <= 1.0:
        raise ValueError("Safe Commit calibration has invalid thresholds")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("writebacks", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--policy", choices=("direct", "selected", "forced"), required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Safe Commit output: {args.output_dir}")
    calibration = load_calibration(args.calibration)
    raw = load_rows(args.writebacks, expected_split="val")
    gate = calibration["gate"]
    safe_rows = [
        apply_safe_commit(
            row,
            confidence_threshold=float(gate["confidence_threshold"]),
            replay_iou_threshold=float(gate["replay_iou_threshold"]),
            require_topology=bool(gate["require_topology"]),
        )
        for _, row in sorted(raw.items())
    ]
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "writeback.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in safe_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "sn7-v5-safe-commit-writeback-v1",
        "policy": args.policy,
        "split": "val",
        "raw_writebacks": str(args.writebacks.resolve()),
        "raw_writebacks_sha256": sha256(args.writebacks),
        "calibration": str(args.calibration.resolve()),
        "calibration_sha256": sha256(args.calibration),
        "gate": gate,
        "raw_metrics": summarize(list(raw.values())),
        "safe_commit_metrics": summarize(safe_rows),
        "safe_minus_raw": {
            name: summarize(safe_rows)[name] - summarize(list(raw.values()))[name]
            for name in summarize(safe_rows)
        },
        "rejected_writeback_count": sum(not row["safe_commit_accepted"] for row in safe_rows),
        "row_count": len(safe_rows),
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
