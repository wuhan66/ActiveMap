#!/usr/bin/env python3
"""Convert Active-Catalog action SFT into a gate-only visual SFT task."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from activemap.agent.vlm_sft import load_vlm_sft_rows

GATE_SYSTEM_PROMPT = (
    "You are the ActiveMap evidence-need gate. Given the current map-update image, "
    "belief, budget, and available evidence metadata, decide whether any candidate "
    "is worth acquiring. Output exactly one JSON object: "
    '{"selection":"ACQUIRE"} or {"selection":"STOP"}.'
)


def _text(message: dict[str, Any]) -> str:
    return next(part["text"] for part in message["content"] if part.get("type") == "text")


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    original = json.loads(_text(row["messages"][2]))
    selection = str(original["selection"])
    if selection not in {"ACQUIRE", "STOP"}:
        raise ValueError(f"unsupported selector target: {selection}")
    user_content = [dict(part) for part in row["messages"][1]["content"]]
    return {
        **{key: value for key, value in row.items() if key != "messages"},
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": GATE_SYSTEM_PROMPT}],
            },
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"selection": selection}, separators=(",", ":")
                        ),
                    }
                ],
            },
        ],
        "training_role": "active_catalog_visual_tool_need_gate",
        "test_assets_read": False,
    }


def _write(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    train = [convert_row(row) for row in load_vlm_sft_rows(args.train_jsonl)]
    validation = [convert_row(row) for row in load_vlm_sft_rows(args.val_jsonl)]
    if {row["split"] for row in train} != {"train"}:
        raise ValueError("train source contains a non-train row")
    if {row["split"] for row in validation} != {"val"}:
        raise ValueError("validation source contains a non-validation row")
    args.output_dir.mkdir(parents=True)
    _write(train, args.output_dir / "train.jsonl")
    _write(validation, args.output_dir / "val.jsonl")
    summary = {
        "schema_version": "active-catalog-gate-sft-v1",
        "train_rows": len(train),
        "val_rows": len(validation),
        "train_actions": dict(
            Counter(
                json.loads(_text(row["messages"][2]))["selection"] for row in train
            )
        ),
        "val_actions": dict(
            Counter(json.loads(_text(row["messages"][2]))["selection"] for row in validation)
        ),
        "candidate_identity_supervision_removed": True,
        "visual_input_retained": True,
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
