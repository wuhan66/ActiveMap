#!/usr/bin/env python3
"""Match runtime Tool-Need call prior using train-rollout records only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibrate(records_path: Path, gate_summary_path: Path) -> dict[str, Any]:
    gate = json.loads(gate_summary_path.read_text(encoding="utf-8"))
    if gate.get("test_assets_read") is not False:
        raise ValueError("gate summary does not preserve the frozen-test protocol")
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eligible = [
        row
        for row in rows
        if row.get("source") == "proactive_gate_check" and row.get("eligible") is True
    ]
    if not eligible or {row.get("controller") for row in eligible} != {
        "edit_conditioned_selector"
    }:
        raise ValueError("expected eligible edit-conditioned train-rollout checks")
    probabilities = sorted(float(row["call_probability"]) for row in eligible)
    target_rate = float(gate["selected"]["metrics"]["call_rate"])
    target_calls = max(1, math.ceil(target_rate * len(probabilities)))
    threshold = probabilities[-target_calls]
    realized_calls = sum(value >= threshold for value in probabilities)
    return {
        "schema_version": "tool-need-runtime-prior-calibration-v1",
        "selection_protocol": "train-rollout-only-call-prior-matching",
        "controller": "edit_conditioned_selector",
        "eligible_train_states": len(probabilities),
        "offline_oof_call_rate": target_rate,
        "target_runtime_calls": target_calls,
        "threshold": threshold,
        "realized_runtime_calls": realized_calls,
        "realized_runtime_call_rate": realized_calls / len(probabilities),
        "probability_range": {
            "min": probabilities[0],
            "max": probabilities[-1],
        },
        "sources": {
            "train_rollout_records": {
                "path": str(records_path),
                "sha256": _sha256(records_path),
            },
            "offline_gate_summary": {
                "path": str(gate_summary_path),
                "sha256": _sha256(gate_summary_path),
            },
        },
        "validation_assets_read": False,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_rollout_records", type=Path)
    parser.add_argument("gate_summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = calibrate(args.train_rollout_records, args.gate_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
