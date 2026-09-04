#!/usr/bin/env python3
"""Combine sampled-rollout and greedy-validation gates for pointer actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def combine_gates(
    diversity: dict[str, Any], validation: dict[str, Any]
) -> dict[str, Any]:
    if diversity.get("test_assets_read") is not False:
        raise ValueError("diversity audit must explicitly forbid test access")
    if validation.get("test_assets_read") is not False:
        raise ValueError("validation audit must explicitly forbid test access")
    rollout_ready = diversity.get("ready_for_tool_belief_grpo") is True
    validation_ready = validation.get("ready_for_executable_grpo") is True
    gates = {
        "sampled_rollout_diversity": rollout_ready,
        "greedy_pointer_validation": validation_ready,
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": "activemap-pointer-action-preflight-v1",
        "sampled_rollout_ready": rollout_ready,
        "greedy_validation_ready": validation_ready,
        "gates": gates,
        "ready_for_executable_grpo": not failed,
        "failed_gates": failed,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("diversity_audit", type=Path)
    parser.add_argument("validation_audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    diversity = json.loads(args.diversity_audit.read_text(encoding="utf-8"))
    validation = json.loads(args.validation_audit.read_text(encoding="utf-8"))
    report = combine_gates(diversity, validation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
