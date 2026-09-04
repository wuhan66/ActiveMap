#!/usr/bin/env python3
"""Compose safety and sparse-tool preferences with train-only rare-action sampling."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _action(row: dict[str, Any], field: str) -> str:
    payload = json.loads(row[field])
    if not isinstance(payload, dict) or "action" not in payload:
        raise ValueError(f"preference {field} is not a structured action")
    return str(payload["action"])


def read_preferences(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"row {line_number} in {path} is not an object")
            chosen = _action(row, "chosen")
            rejected = _action(row, "rejected")
            if row["chosen"] == row["rejected"]:
                raise ValueError(f"row {line_number} has identical actions")
            if float(row["chosen_utility"]) <= float(row["rejected_utility"]):
                raise ValueError(f"row {line_number} has a non-positive utility margin")
            if not chosen or not rejected:
                raise ValueError(f"row {line_number} has an empty action")
            rows.append(row)
    if not rows:
        raise ValueError(f"no preference rows in {path}")
    return rows


def compose_preference_rows(
    main_rows: list[dict[str, Any]],
    tool_rows: list[dict[str, Any]],
    *,
    split: str,
    use_tool_repeat: int = 5,
    no_tool_ratio: float = 3.0,
    seed: int = 20260821,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    if use_tool_repeat < 1 or no_tool_ratio < 0.0:
        raise ValueError("invalid repetition or no-tool ratio")
    training = split == "train"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tool_rows:
        if row.get("protocol") != "post-acquisition-sparse-tool-controller-v1":
            raise ValueError("tool preference has an unexpected protocol")
        if row.get("split") != split:
            raise ValueError("tool preference split mismatch")
        grouped[str(row["trajectory_id"])].append(row)
    positive_ids = sorted(
        sequence_id
        for sequence_id, rows in grouped.items()
        if any(_action(row, "chosen") == "USE_TOOL" for row in rows)
    )
    negative_ids = sorted(set(grouped) - set(positive_ids))
    selected_negative_ids = negative_ids
    if training:
        limit = min(len(negative_ids), math.ceil(len(positive_ids) * no_tool_ratio))
        selected_negative_ids = sorted(random.Random(seed).sample(negative_ids, limit))

    selected_tool: list[dict[str, Any]] = []
    for sequence_id in [*positive_ids, *selected_negative_ids]:
        for row in grouped[sequence_id]:
            repeats = (
                use_tool_repeat
                if training and _action(row, "chosen") == "USE_TOOL"
                else 1
            )
            selected_tool.extend([row] * repeats)
    tagged_main = [{**row, "composition_source": "safety_agent"} for row in main_rows]
    tagged_tool = [
        {
            **row,
            "preference_family": str(row.get("preference_family", "tool")),
            "composition_source": "sparse_tool_controller",
        }
        for row in selected_tool
    ]
    composed = [*tagged_main, *tagged_tool]
    random.Random(seed).shuffle(composed)
    pair_counts = Counter(
        f"{_action(row, 'chosen')}>{_action(row, 'rejected')}" for row in composed
    )
    summary = {
        "schema_version": "composed-agent-preferences-v1",
        "split": split,
        "training_oversampling": training,
        "seed": seed,
        "use_tool_repeat": use_tool_repeat if training else 1,
        "no_tool_ratio": no_tool_ratio if training else None,
        "main_input_count": len(main_rows),
        "tool_input_count": len(tool_rows),
        "tool_positive_sequence_count": len(positive_ids),
        "tool_negative_sequence_count": len(negative_ids),
        "selected_tool_negative_sequence_count": len(selected_negative_ids),
        "output_count": len(composed),
        "chosen_use_tool_count": sum(
            _action(row, "chosen") == "USE_TOOL" for row in composed
        ),
        "pair_counts": dict(sorted(pair_counts.items())),
        "test_assets_read": False,
    }
    return composed, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("main_preferences", type=Path)
    parser.add_argument("tool_preferences", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--use-tool-repeat", type=int, default=5)
    parser.add_argument("--no-tool-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    rows, summary = compose_preference_rows(
        read_preferences(args.main_preferences),
        read_preferences(args.tool_preferences),
        split=args.split,
        use_tool_repeat=args.use_tool_repeat,
        no_tool_ratio=args.no_tool_ratio,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
