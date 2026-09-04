#!/usr/bin/env python3
"""Build complete PRE/POST visual states for cached closed-loop evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from activemap.agent.tool_sft import terminal_action
from activemap.geo_tools.records import GeoToolCall, GeoToolName
from scripts.build_semantic_vlm_sft import (
    _index_rows,
    _messages,
    _observation,
    _sha256,
    consensus_tool_opportunity,
)


def build_rollout_dataset(
    semantic_paths: list[Path],
    visual_root: Path,
    output_dir: Path,
    *,
    minimum_votes: int,
) -> dict[str, Any]:
    order, indices = _index_rows(semantic_paths)
    split = indices[0][order[0]].split
    if split not in {"train", "val"} or any(
        row.split != split for index in indices for row in index.values()
    ):
        raise ValueError("rollout input must contain one train or validation split")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    records = []
    action_counts: Counter[str] = Counter()
    for example_id in order:
        rows = [index[example_id] for index in indices]
        row = rows[0]
        decision = consensus_tool_opportunity(rows, minimum_votes=minimum_votes)
        visual_path = visual_root / f"{row.task_id}.jpg"
        if not visual_path.is_file():
            raise FileNotFoundError(visual_path)
        terminal = terminal_action(row.gt_edit).model_dump_json(exclude_none=True)
        if decision["use_tool"]:
            pre_action = json.dumps(
                {
                    "action": "USE_TOOL",
                    "tool_call": GeoToolCall(
                        call_id=f"vlm-{example_id[:12]}",
                        tool=GeoToolName.RASTER_SEGMENT,
                        inputs={"evidence_id": row.evidence_id},
                    ).model_dump(mode="json", exclude_none=True),
                },
                separators=(",", ":"),
            )
            action_counts["USE_TOOL"] += 1
        else:
            pre_action = terminal
            action_counts["TERMINAL"] += 1
        shared = {
            "example_id": example_id,
            "task_id": row.task_id,
            "oracle_use_tool": bool(decision["use_tool"]),
            "consensus": decision,
            "split": split,
        }
        records.append(
            {
                **shared,
                "messages": _messages(
                    visual_path, _observation(row, post_tool=False), pre_action
                ),
                "stage": "PRE_TOOL",
            }
        )
        records.append(
            {
                **shared,
                "messages": _messages(
                    visual_path, _observation(row, post_tool=True), terminal
                ),
                "stage": "POST_TOOL",
            }
        )
    output_dir.mkdir(parents=True)
    output_path = output_dir / "rollout.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "semantic-vlm-rollout-eval-v1",
        "split": split,
        "source_example_count": len(order),
        "record_count": len(records),
        "complete_pre_post_pairs": len(records) == 2 * len(order),
        "task_count": len({indices[0][example_id].task_id for example_id in order}),
        "minimum_beneficial_votes": minimum_votes,
        "pre_action_counts": dict(action_counts),
        "test_assets_read": False,
        "sources": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in semantic_paths
        ],
        "rollout_sha256": _sha256(output_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("visual_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--semantic", type=Path, action="append", required=True)
    parser.add_argument("--minimum-votes", type=int, default=2)
    args = parser.parse_args()
    result = build_rollout_dataset(
        args.semantic,
        args.visual_root,
        args.output_dir,
        minimum_votes=args.minimum_votes,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
