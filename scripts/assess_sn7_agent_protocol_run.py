#!/usr/bin/env python3
"""Fail closed unless a modern agent protocol run is structurally valid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROTOCOLS = {
    "geommagent_style": {
        "controller_protocol": "geommagent_plan_execute_self_evaluate",
        "policy": "geommagent_style_qwen",
        "stages": {"PLAN", "EXECUTE", "SELF_EVALUATE", "SELECT", "REACT"},
    },
    "sensesearch_style": {
        "controller_protocol": "sensesearch_iterative_search_crop_reason",
        "policy": "sensesearch_style_qwen",
        "stages": {"SEARCH", "REASON", "SELECT", "REACT"},
    },
}
VALID_EDITS = {"KEEP", "ADD", "DELETE", "RESHAPE"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def assess(
    run_root: Path,
    protocol_name: str,
    expected_count: int,
    min_valid_action_rate: float,
) -> dict[str, Any]:
    specification = PROTOCOLS[protocol_name]
    summary = _read_json(run_root / "evaluation" / "summary.json")
    traces = _read_jsonl(run_root / "evaluation" / "traces.jsonl")
    run_state = _read_json(run_root / "run_state.json")
    protocol = summary.get("protocol", {})
    metrics = summary.get("metrics", {})

    events = [event for row in traces for event in row.get("events", [])]
    tool_observations = [
        observation
        for event in events
        for observation in event.get("observable_state", {}).get(
            "tool_observations", []
        )
    ]
    registered_tools = set(protocol.get("registered_tools", []))
    required_tools = (
        {"RASTER_CROP", "RASTER_SEGMENT", "VECTOR_INSPECT"}
        if protocol_name == "sensesearch_style"
        else {"IMAGE_QUALITY", "TEMPORAL_CHANGE"}
    )
    budget_safe = all(
        float(row.get("spent_cost", 0.0)) <= float(row.get("budget", 0.0)) + 1e-6
        and int(row.get("acquisitions", 0)) <= 2
        and int(row.get("tool_calls", 0)) <= 2
        for row in traces
    )
    observable_protocol = all(
        event.get("controller_protocol") == protocol_name
        and event.get("observable_state", {}).get("controller_stage")
        in specification["stages"]
        and "target_edit" not in event.get("observable_state", {})
        for event in events
    )
    checks = {
        "completed_process": run_state.get("status") == "completed"
        and run_state.get("returncode") == 0,
        "schema": summary.get("schema_version")
        == "active-catalog-closed-loop-evaluation-v1",
        "sample_count": summary.get("sample_count") == expected_count
        and len(traces) == expected_count,
        "validation_only": summary.get("split") == "val"
        and summary.get("test_assets_read") is False
        and all(row.get("split") == "val" for row in traces)
        and all(row.get("test_assets_read") is False for row in traces),
        "policy": protocol.get("policy_mode") == protocol_name
        and protocol.get("controller_protocol")
        == specification["controller_protocol"]
        and all(row.get("policy") == specification["policy"] for row in traces),
        "model_tool_control": protocol.get("tool_mode") == "model"
        and protocol.get("explicit_geospatial_tool_calls") is True
        and required_tools.issubset(registered_tools)
        and bool(protocol.get("asset_root_maps")),
        "valid_action_rate": float(metrics.get("valid_action_rate", 0.0))
        >= min_valid_action_rate,
        "nonempty_events": len(events) >= expected_count
        and all(bool(row.get("events")) for row in traces),
        "observable_protocol": observable_protocol,
        "budget_safe": budget_safe,
        "tool_execution_success": all(
            observation.get("success") is True for observation in tool_observations
        ),
        "terminal_edits": all(row.get("predicted_edit") in VALID_EDITS for row in traces),
    }
    return {
        "schema_version": "sn7-modern-agent-protocol-assessment-v1",
        "protocol": protocol_name,
        "minimum_valid_action_rate": min_valid_action_rate,
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "sample_count": summary.get("sample_count"),
            "valid_action_rate": metrics.get("valid_action_rate"),
            "fallback_episode_rate": metrics.get("fallback_episode_rate"),
            "mean_acquisitions": metrics.get("mean_acquisitions"),
            "mean_tool_calls": metrics.get("mean_tool_calls"),
            "false_edit_rate": metrics.get("false_edit_rate"),
            "mean_quality_cost_utility": metrics.get("mean_quality_cost_utility"),
        },
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--protocol", required=True, choices=tuple(PROTOCOLS)
    )
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--min-valid-action-rate", type=float, default=0.5)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Preserve an earlier assessment beside the run before reassessing.",
    )
    args = parser.parse_args()

    output = args.run_root / "protocol_assessment.json"
    if output.exists():
        if not args.replace_existing:
            raise FileExistsError(output)
        suffix = 1
        history = args.run_root / "protocol_assessment.previous.json"
        while history.exists():
            suffix += 1
            history = args.run_root / f"protocol_assessment.previous-{suffix}.json"
        output.replace(history)
    result = assess(
        args.run_root,
        args.protocol,
        args.expected_count,
        args.min_valid_action_rate,
    )
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
