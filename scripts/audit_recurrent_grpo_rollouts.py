#!/usr/bin/env python3
"""Gate recurrent GRPO on executable reward variance and exploration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from activemap.agent.recurrent_grpo import audit_trajectory_groups


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"expected object at {path}:{line_number}")
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("rollouts", nargs="+", type=Path)
    parser.add_argument("--minimum-group-size", type=int, default=4)
    parser.add_argument("--minimum-variable-group-rate", type=float, default=0.20)
    parser.add_argument("--minimum-nonstop-rate", type=float, default=0.05)
    parser.add_argument("--minimum-keep-trajectories", type=int, default=0)
    parser.add_argument("--minimum-commit-trajectories", type=int, default=0)
    parser.add_argument("--minimum-tool-trajectories", type=int, default=0)
    args = parser.parse_args()
    report = audit_trajectory_groups(
        read_rows(args.rollouts),
        minimum_group_size=args.minimum_group_size,
        minimum_variable_group_rate=args.minimum_variable_group_rate,
        minimum_nonstop_rate=args.minimum_nonstop_rate,
        minimum_keep_trajectories=args.minimum_keep_trajectories,
        minimum_commit_trajectories=args.minimum_commit_trajectories,
        minimum_tool_trajectories=args.minimum_tool_trajectories,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ready_for_recurrent_grpo"]:
        raise SystemExit(12)


if __name__ == "__main__":
    main()
