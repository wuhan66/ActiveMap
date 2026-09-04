#!/usr/bin/env python3
"""Apply an auditable terminal commit-operation gate to rollout JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def gate_row(row: dict[str, Any], allowed_operations: set[str]) -> dict[str, Any]:
    result = dict(row)
    prediction = str(row["prediction"])
    operation = prediction.removeprefix("COMMIT:") if prediction.startswith("COMMIT:") else None
    passed = operation is None or operation in allowed_operations
    result["terminal_operation_gate"] = {
        "allowed_operations": sorted(allowed_operations),
        "passed": passed,
        "uses_ground_truth": False,
        "spent_cost_retained": True,
    }
    if not passed:
        result["ungated_prediction"] = prediction
        result["ungated_selected_evidence_ids"] = list(row["selected_evidence_ids"])
        result["prediction"] = "REJECT"
    return result


def apply_gate(
    input_path: Path,
    output_path: Path,
    allowed_operations: set[str],
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    counts = {"rows": 0, "passed": 0, "blocked": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as source, output_path.open(
        "x", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            result = gate_row(json.loads(line), allowed_operations)
            passed = bool(result["terminal_operation_gate"]["passed"])
            counts["rows"] += 1
            counts["passed" if passed else "blocked"] += 1
            destination.write(json.dumps(result, separators=(",", ":")) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--allow-operation",
        action="append",
        choices=("ADD", "DELETE", "RESHAPE"),
        required=True,
    )
    args = parser.parse_args()
    counts = apply_gate(args.input, args.output, set(args.allow_operation))
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
