#!/usr/bin/env python3
"""Freeze an outcome-free selective-tool threshold from train belief states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _controller_state(row: dict[str, Any]) -> dict[str, Any]:
    for message in row.get("prompt", []):
        if message.get("role") != "user":
            continue
        for item in message.get("content", []):
            if item.get("type") != "text":
                continue
            value = json.loads(item["text"])
            if value.get("controller_stage") == "SELECT":
                return value
    raise ValueError("RL state lacks a SELECT controller state")


def calibrate(path: Path, *, target_call_rate: float) -> dict[str, Any]:
    if not 0.0 < target_call_rate < 1.0:
        raise ValueError("target_call_rate must be in (0, 1)")
    uncertainties: list[float] = []
    task_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train" or row.get("test_assets_read") is not False:
                raise ValueError(f"non-train or unaudited row at line {line_number}")
            state = _controller_state(row)
            if not state.get("selected_evidence_ids"):
                continue
            uncertainty = float(state["belief"]["uncertainty"])
            if not math.isfinite(uncertainty) or not 0.0 <= uncertainty <= 1.0:
                raise ValueError(f"invalid uncertainty at line {line_number}")
            uncertainties.append(uncertainty)
            task_ids.add(str(row["task_id"]))
    if len(uncertainties) < 2:
        raise ValueError("at least two eligible train states are required")
    threshold = float(
        np.quantile(
            np.asarray(uncertainties, dtype=np.float64),
            1.0 - target_call_rate,
            method="higher",
        )
    )
    observed = float(np.mean(np.asarray(uncertainties) >= threshold))
    return {
        "schema_version": "uncertainty-tool-gate-summary-v1",
        "split": "train",
        "eligible_state_count": len(uncertainties),
        "task_count": len(task_ids),
        "target_call_rate": target_call_rate,
        "observed_train_call_rate": observed,
        "selected": {
            "feature": "belief_uncertainty",
            "threshold": threshold,
            "direction": "greater_or_equal",
        },
        "outcome_labels_used": False,
        "validation_metrics_used": False,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_states", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target-call-rate", type=float, default=0.15)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    summary = calibrate(args.train_states, target_call_rate=args.target_call_rate)
    summary["source"] = {
        "path": str(args.train_states.resolve()),
        "sha256": hashlib.sha256(args.train_states.read_bytes()).hexdigest(),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "gate.json").write_text(
        json.dumps(
            {
                "schema_version": "belief-uncertainty-gate-v1",
                "threshold": summary["selected"]["threshold"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
