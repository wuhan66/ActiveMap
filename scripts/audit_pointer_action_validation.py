#!/usr/bin/env python3
"""Audit greedy validation of the pointer-action tool controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _method_record(records: Any, method: str) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError("summary records must be a list")
    selected = [row for row in records if isinstance(row, dict) and row.get("method") == method]
    if not selected:
        raise ValueError(f"summary contains no records for method={method}")
    return selected


def audit_pointer_action_validation(payload: dict[str, Any], *, method: str) -> dict[str, Any]:
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("summary lacks protocol")
    validity_by_method = payload.get("llm_validity_by_method")
    if not isinstance(validity_by_method, dict) or not isinstance(
        validity_by_method.get(method), dict
    ):
        raise ValueError(f"summary lacks LLM validity for method={method}")
    validity = validity_by_method[method]
    results = _method_record(payload.get("results"), method)
    positive = _method_record(payload.get("tool_positive_results"), method)
    pointer_counts = validity.get("pointer_resolution_counts", {})
    if not isinstance(pointer_counts, dict):
        raise ValueError("pointer_resolution_counts must be an object")
    pointer_resolved = int(pointer_counts.get("selected_evidence_index_v1", 0))
    positive_tool_calls = sum(
        float(row.get("mean_tool_calls", 0.0)) * int(row.get("sample_count", 0))
        for row in positive
    )
    total_tool_calls = sum(
        float(row.get("mean_tool_calls", 0.0)) * int(row.get("sample_count", 0))
        for row in results
    )
    schema_value = validity.get("schema_valid_rate")
    executable_value = validity.get("executable_valid_rate")
    fallback_value = validity.get("fallback_rate")
    schema_rate = 0.0 if schema_value is None else float(schema_value)
    executable_rate = 0.0 if executable_value is None else float(executable_value)
    fallback_rate = 1.0 if fallback_value is None else float(fallback_value)
    gates = {
        "validation_only": protocol.get("split") == "val"
        and protocol.get("test_assets_read") is False,
        "greedy_decode": protocol.get("decoding", {}).get("do_sample") is False,
        "pointer_resolved": pointer_resolved > 0,
        "schema_validity": schema_rate >= 0.99,
        "executable_validity": executable_rate >= 0.99,
        "no_material_fallback": fallback_rate <= 0.01,
        "tool_positive_branch": positive_tool_calls >= 1.0,
    }
    return {
        "schema_version": "activemap-pointer-action-validation-audit-v1",
        "method": method,
        "pointer_resolution_counts": pointer_counts,
        "pointer_resolved_calls": pointer_resolved,
        "estimated_total_tool_calls": total_tool_calls,
        "estimated_positive_subset_tool_calls": positive_tool_calls,
        "schema_valid_rate": schema_rate,
        "executable_valid_rate": executable_rate,
        "fallback_rate": fallback_rate,
        "gates": gates,
        "ready_for_executable_grpo": all(gates.values()),
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", default="qwen3_4b_sft_tool_to_belief")
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    report = audit_pointer_action_validation(payload, method=args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
