#!/usr/bin/env python3
"""Audit matched recurrent protocols before learned-policy comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    required_protocol = ("tool_mode", "max_candidates", "max_acquisitions")
    left = candidate.get("protocol", {})
    right = reference.get("protocol", {})
    checks = {
        "validation_only": candidate.get("split") == reference.get("split") == "val",
        "test_isolated": candidate.get("test_assets_read") is False and reference.get("test_assets_read") is False,
        "same_sample_count": candidate.get("sample_count") == reference.get("sample_count"),
        **{f"same_{name}": left.get(name) == right.get(name) for name in required_protocol},
        "deterministic_evaluation": (
            left.get("stochastic_policy_sampling") is False
            and right.get("stochastic_policy_sampling") is False
        ),
    }
    return {
        "schema_version": "active-catalog-policy-pair-audit-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_summary", type=Path)
    parser.add_argument("reference_summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = audit(
        json.loads(args.candidate_summary.read_text(encoding="utf-8")),
        json.loads(args.reference_summary.read_text(encoding="utf-8")),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
