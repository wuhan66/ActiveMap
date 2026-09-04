#!/usr/bin/env python3
"""Fail closed unless a ReAct smoke produced valid, validation-only traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def assess(summary_path: Path, traces_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    traces = [
        json.loads(line)
        for line in traces_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metrics = summary.get("metrics", {})
    protocol = summary.get("protocol", {})
    checks = {
        "schema": summary.get("schema_version")
        == "active-catalog-closed-loop-evaluation-v1",
        "sample_count": summary.get("sample_count") == 2 and len(traces) == 2,
        "validation_only": summary.get("split") == "val"
        and summary.get("test_assets_read") is False
        and all(row.get("test_assets_read") is False for row in traces),
        "react_policy": protocol.get("policy_mode") == "react"
        and protocol.get("controller_protocol") == "react_observation_reason_action",
        "model_tool_control": protocol.get("tool_mode") == "model"
        and protocol.get("explicit_geospatial_tool_calls") is True,
        "valid_action_rate": float(metrics.get("valid_action_rate", 0.0)) >= 0.5,
        "nonempty_events": all(bool(row.get("events")) for row in traces),
        "observable_protocol": all(
            all(
                event.get("observable_state", {}).get("controller_stage") == "REACT"
                and "target_edit" not in event.get("observable_state", {})
                for event in row.get("events", [])
            )
            for row in traces
        ),
    }
    return {
        "schema_version": "sn7-react-smoke-assessment-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "valid_action_rate": metrics.get("valid_action_rate"),
            "fallback_episode_rate": metrics.get("fallback_episode_rate"),
            "mean_tool_calls": metrics.get("mean_tool_calls"),
        },
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("traces", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = assess(args.summary, args.traces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
