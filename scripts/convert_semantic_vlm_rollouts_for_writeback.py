#!/usr/bin/env python3
"""Convert visual-policy traces into the frozen vector-writeback contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from activemap.models import EditOperation


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert_trace(trace: dict[str, Any], *, budget: float) -> dict[str, Any]:
    if trace.get("split") not in {"train", "val"}:
        raise ValueError("writeback conversion permits train or validation traces only")
    operation = EditOperation(str(trace["policy_operation"]))
    prediction = "REJECT" if operation == EditOperation.KEEP else f"COMMIT:{operation.value}"
    target_operation = EditOperation(str(trace["target_operation"]))
    target = (
        "REJECT"
        if target_operation == EditOperation.KEEP
        else f"COMMIT:{target_operation.value}"
    )
    evidence_id = str(trace["evidence_id"])
    if not evidence_id:
        raise ValueError("rollout trace lacks visual evidence")
    return {
        "task_id": str(trace["task_id"]),
        "budget": budget,
        "target": target,
        "prediction": prediction,
        "selected_evidence_ids": [evidence_id],
        "semantic_tool_called": bool(trace["predicted_use_tool"]),
        "semantic_tool_cost": float(trace["tool_cost"]),
        "policy_utility": float(trace["policy_utility"]),
        "source_example_id": str(trace["example_id"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--budget", type=float, default=0.75)
    args = parser.parse_args()
    if args.budget <= 0.0:
        raise ValueError("budget must be positive")
    traces = [
        json.loads(line)
        for line in args.traces.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not traces:
        raise ValueError("no rollout traces to convert")
    rows = [convert_trace(trace, budget=args.budget) for trace in traces]
    if len({(row["task_id"], row["source_example_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate rollout trace identity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "semantic-vlm-writeback-rollout-v1",
        "source": str(args.traces.resolve()),
        "source_sha256": _sha256(args.traces),
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "record_count": len(rows),
        "budget": args.budget,
        "tool_call_rate": sum(row["semantic_tool_called"] for row in rows) / len(rows),
        "visual_evidence_count_per_record": 1,
        "test_assets_read": False,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
