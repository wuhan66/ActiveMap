#!/usr/bin/env python3
"""Permit one SN7 contextual-RL seed only after frozen SFT safety gates pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def assess(
    selector: dict[str, Any],
    tool_branch: dict[str, Any],
    writeback: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "selector_three_seed_passed": (
            selector.get("schema_version") == "active-catalog-paper-promotion-v1"
            and selector.get("passed") is True
            and int(selector.get("seed_count", 0)) >= 3
        ),
        "tool_branch_three_seed_passed": (
            tool_branch.get("schema_version") == "active-catalog-tool-branch-promotion-v1"
            and tool_branch.get("promote") is True
            and int(tool_branch.get("minimum_seed_count", 0)) >= 3
        ),
        "executable_writeback_three_seed_passed": (
            writeback.get("schema_version") == "active-catalog-tool-writeback-promotion-v1"
            and writeback.get("promote") is True
            and int(writeback.get("minimum_seed_count", 0)) >= 3
        ),
        "test_isolated": all(
            payload.get("test_assets_read") is False
            for payload in (selector, tool_branch, writeback)
        ),
    }
    return {
        "schema_version": "active-catalog-vlm-rl-readiness-v1",
        "ready_for_single_seed_rl": all(checks.values()),
        "checks": checks,
        "permits": "one offline multimodal contextual-RL seed",
        "does_not_establish": "online recurrent RL or RL map-quality gain",
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selector_promotion", type=Path)
    parser.add_argument("tool_branch_promotion", type=Path)
    parser.add_argument("writeback_promotion", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = assess(*[
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.selector_promotion, args.tool_branch_promotion, args.writeback_promotion)
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
