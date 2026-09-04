#!/usr/bin/env python3
"""Compose acquisition and sparse-tool SFT without distorting validation prevalence."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOOL_PROTOCOLS = {
    "post-acquisition-sparse-tool-controller-v1",
    "post-acquisition-reachable-tool-controller-v2",
    "post-acquisition-reachable-tool-controller-v3-pointer-actions",
}


def _action(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("SFT row must contain exactly three messages")
    payload = json.loads(messages[2]["content"])
    return str(payload["action"])


def _split(row: dict[str, Any]) -> str:
    messages = row["messages"]
    observation = json.loads(messages[1]["content"])
    split = str(observation["split"])
    if split not in {"train", "val"}:
        raise ValueError(f"composed SFT forbids split={split!r}")
    return split


def read_sft(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"row {line_number} in {path} is not an object")
            _action(row)
            _split(row)
            rows.append(row)
    if not rows:
        raise ValueError(f"no SFT rows in {path}")
    return rows


def compose_sft_rows(
    main_rows: list[dict[str, Any]],
    tool_rows: list[dict[str, Any]],
    *,
    training: bool,
    use_tool_repeat: int = 5,
    no_tool_ratio: float = 3.0,
    keep_all_tool_sequences: bool = False,
    seed: int = 20260821,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if use_tool_repeat < 1:
        raise ValueError("use_tool_repeat must be positive")
    if no_tool_ratio < 0.0:
        raise ValueError("no_tool_ratio must be non-negative")
    expected_split = "train" if training else "val"
    if any(_split(row) != expected_split for row in [*main_rows, *tool_rows]):
        raise ValueError(f"all rows must use split={expected_split}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tool_rows:
        if row.get("protocol") not in TOOL_PROTOCOLS:
            raise ValueError("tool row has an unexpected protocol")
        grouped[str(row["trajectory_id"])].append(row)
    positive_ids = sorted(
        sequence_id
        for sequence_id, rows in grouped.items()
        if int(rows[0]["oracle_tool_stage"]) > 0
    )
    negative_ids = sorted(set(grouped) - set(positive_ids))

    selected_tool_rows: list[dict[str, Any]] = []
    selected_negative_ids = negative_ids
    if training and not keep_all_tool_sequences:
        rng = random.Random(seed)
        negative_limit = min(
            len(negative_ids), math.ceil(len(positive_ids) * no_tool_ratio)
        )
        selected_negative_ids = sorted(rng.sample(negative_ids, negative_limit))
    for sequence_id in [*positive_ids, *selected_negative_ids]:
        for row in grouped[sequence_id]:
            repeats = use_tool_repeat if training and _action(row) == "USE_TOOL" else 1
            selected_tool_rows.extend([row] * repeats)

    tagged_main = [{**row, "composition_source": "acquisition_agent"} for row in main_rows]
    tagged_tool = [
        {**row, "composition_source": "sparse_tool_controller"}
        for row in selected_tool_rows
    ]
    composed = [*tagged_main, *tagged_tool]
    random.Random(seed).shuffle(composed)
    action_counts = Counter(_action(row) for row in composed)
    source_counts = Counter(str(row["composition_source"]) for row in composed)
    summary = {
        "schema_version": "composed-agent-sft-v1",
        "split": expected_split,
        "training_oversampling": training,
        "seed": seed,
        "use_tool_repeat": use_tool_repeat if training else 1,
        "no_tool_ratio": no_tool_ratio if training else None,
        "keep_all_tool_sequences": keep_all_tool_sequences or not training,
        "main_input_count": len(main_rows),
        "tool_input_count": len(tool_rows),
        "tool_positive_sequence_count": len(positive_ids),
        "tool_negative_sequence_count": len(negative_ids),
        "selected_tool_negative_sequence_count": len(selected_negative_ids),
        "output_count": len(composed),
        "action_counts": dict(sorted(action_counts.items())),
        "action_fractions": {
            action: count / len(composed)
            for action, count in sorted(action_counts.items())
        },
        "source_counts": dict(sorted(source_counts.items())),
        "tool_protocols": sorted(
            {
                str(row.get("protocol"))
                for row in selected_tool_rows
                if row.get("protocol") is not None
            }
        ),
        "test_assets_read": False,
    }
    return composed, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("main_sft", type=Path)
    parser.add_argument("tool_sft", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--use-tool-repeat", type=int, default=5)
    parser.add_argument("--no-tool-ratio", type=float, default=3.0)
    parser.add_argument("--keep-all-tool-sequences", action="store_true")
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    rows, summary = compose_sft_rows(
        read_sft(args.main_sft),
        read_sft(args.tool_sft),
        training=args.training,
        use_tool_repeat=args.use_tool_repeat,
        no_tool_ratio=args.no_tool_ratio,
        keep_all_tool_sequences=args.keep_all_tool_sequences,
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
