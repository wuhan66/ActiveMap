#!/usr/bin/env python3
"""Deterministically repeat rare tool-call SFT records without touching validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--acquire-repeat", type=int, default=3)
    parser.add_argument("--tool-repeat", type=int, default=1)
    parser.add_argument(
        "--assume-train",
        action="store_true",
        help="Allow rows without a split field when the caller has verified a train-only input.",
    )
    args = parser.parse_args()
    if args.acquire_repeat < 1:
        parser.error("--acquire-repeat must be positive")
    if args.tool_repeat < 1:
        parser.error("--tool-repeat must be positive")

    rows = []
    acquire_count = 0
    tool_count = 0
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split = row.get("split")
            if split != "train" and not (split is None and args.assume_train):
                raise ValueError("SFT balancing is restricted to the training split")
            content = row["messages"][2]["content"]
            if isinstance(content, list):
                content = next(part["text"] for part in content if part.get("type") == "text")
            action = json.loads(content)["action"]
            repeats = args.acquire_repeat if action == "ACQUIRE" else 1
            if action == "USE_TOOL":
                repeats = args.tool_repeat
            acquire_count += int(action == "ACQUIRE") * repeats
            tool_count += int(action == "USE_TOOL") * repeats
            rows.extend([row] * repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "acquire_repeat": args.acquire_repeat,
        "tool_repeat": args.tool_repeat,
        "records": len(rows),
        "acquire_records": acquire_count,
        "acquire_fraction": acquire_count / max(len(rows), 1),
        "tool_records": tool_count,
        "tool_fraction": tool_count / max(len(rows), 1),
        "validation_oversampling_allowed": False,
        "missing_split_assumed_train": args.assume_train,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
