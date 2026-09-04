#!/usr/bin/env python3
"""Freeze a v10 controller for the naturally calibrated v11 two-stage policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(
    static: dict[str, Any], gate: dict[str, Any], *, run_dir: Path
) -> dict[str, Any]:
    if gate.get("promotion_gate", {}).get("passed") is not True:
        raise ValueError("Tool-Need gate did not pass natural validation")
    validation = gate.get("validation", {})
    if float(validation.get("false_call_rate", 1.0)) > 0.02:
        raise ValueError("Tool-Need gate exceeds frozen false-call ceiling")
    rows = [
        row
        for row in static.get("checkpoints", [])
        if row.get("test_assets_read") is False
        and float(row.get("schema_valid_rate", 0.0)) >= 0.99
        and float(row.get("executable_valid_rate", 0.0)) >= 0.99
        and int(row.get("predicted_tool_calls", 0)) > 0
        and float(row.get("tool_positive_exact", 0.0)) >= 0.10
    ]
    if not rows:
        raise ValueError("no structurally valid v10 controller checkpoint")
    selected = max(
        rows,
        key=lambda row: (
            float(row["macro_f1"]),
            float(row["exact_action_accuracy"]),
            float(row["tool_positive_exact"]),
        ),
    )
    adapter = run_dir / "checkpoints" / str(selected["label"])
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(adapter)
    return {
        "schema_version": "muno21-calibrated-two-stage-controller-freeze-v1",
        "selected_checkpoint": selected["label"],
        "adapter_path": str(adapter),
        "selection_order": [
            "macro_f1",
            "exact_action_accuracy",
            "tool_positive_exact",
        ],
        "controller_metrics": selected,
        "gate_metrics": validation,
        "role_separation": {
            "controller": "structured action, tool identity, evidence identity, terminal edit",
            "gate": "natural-prior call admission",
        },
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("static_decision", type=Path)
    parser.add_argument("gate_summary", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = select(
        json.loads(args.static_decision.read_text(encoding="utf-8")),
        json.loads(args.gate_summary.read_text(encoding="utf-8")),
        run_dir=args.run_dir,
    )
    report["sources"] = {
        "static_decision_sha256": _sha256(args.static_decision),
        "gate_summary_sha256": _sha256(args.gate_summary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
