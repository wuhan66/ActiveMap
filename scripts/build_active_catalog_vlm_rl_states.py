#!/usr/bin/env python3
"""Build test-free multimodal contextual policy-optimization states."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.build_active_catalog_vlm_preferences import action_message, read_jsonl


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt(row: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    messages = copy.deepcopy(row["messages"][:-1])
    images = [
        part
        for message in messages
        for part in message.get("content", [])
        if isinstance(part, dict) and part.get("type") == "image"
    ]
    if len(images) != 1:
        raise ValueError("RL prompt must contain exactly one image")
    path = Path(str(images[0]["image"]))
    if not path.is_absolute():
        path = (root / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    images[0]["image"] = str(path)
    return messages


def build_states(
    sft_path: Path,
    index_path: Path,
    output: Path,
    *,
    expected_split: str,
    max_candidates: int = 16,
) -> dict[str, Any]:
    if expected_split not in {"train", "val"} or max_candidates < 1:
        raise ValueError("invalid RL state settings")
    sft_rows = read_jsonl(sft_path)
    index_rows = read_jsonl(index_path)
    index = {str(row["example_id"]): row for row in index_rows}
    if len(index) != len(index_rows) or len(sft_rows) != len(index_rows):
        raise ValueError("SFT/index identity mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    action_counts: Counter[str] = Counter()
    task_ids: set[str] = set()
    aoi_ids: set[str] = set()
    seen: set[str] = set()
    with output.open("x", encoding="utf-8") as destination:
        for row in sft_rows:
            example_id = str(row["example_id"])
            hidden = index.get(example_id)
            if example_id in seen or hidden is None:
                raise ValueError(f"duplicate or missing RL state: {example_id}")
            seen.add(example_id)
            if row.get("split") != expected_split or hidden.get("split") != expected_split:
                raise ValueError(f"split mismatch for {example_id}")
            if hidden.get("model_visible") is not False or hidden.get("test_assets_read") is not False:
                raise ValueError(f"hidden utility contract failed for {example_id}")
            actions = [
                {
                    "key": "STOP",
                    "message": action_message("STOP"),
                    "utility": float(hidden["stop_utility"]),
                    "cost": 0.0,
                }
            ]
            for candidate in hidden["candidates"][:max_candidates]:
                key = f"ACQUIRE:{candidate['evidence_id']}"
                actions.append(
                    {
                        "key": key,
                        "message": action_message(key),
                        "utility": float(candidate["utility"]),
                        "cost": float(candidate["cost"]),
                    }
                )
            if len({action["key"] for action in actions}) != len(actions):
                raise ValueError(f"duplicate executable action for {example_id}")
            target_key = (
                "STOP"
                if hidden["target_selection"] == "STOP"
                else f"ACQUIRE:{hidden['target_evidence_id']}"
            )
            target = next((action for action in actions if action["key"] == target_key), None)
            if target is None or abs(target["utility"] - float(hidden["target_utility"])) > 1e-8:
                raise ValueError(f"target utility mismatch for {example_id}")
            if target["utility"] + 1e-8 < max(action["utility"] for action in actions):
                raise ValueError(f"target is not utility-optimal for {example_id}")
            output_row = {
                "schema_version": "active-catalog-vlm-rl-state-v1",
                "example_id": example_id,
                "task_id": str(row["task_id"]),
                "aoi_id": str(hidden.get("aoi_id", hidden.get("source_aoi", row["task_id"]))),
                "split": expected_split,
                "prompt": _prompt(row, sft_path.parent),
                "actions": actions,
                "target_action_key": target_key,
                "model_visible_utility": False,
                "test_assets_read": False,
            }
            destination.write(json.dumps(output_row, separators=(",", ":")) + "\n")
            action_counts[target["key"].split(":", 1)[0]] += 1
            task_ids.add(output_row["task_id"])
            aoi_ids.add(output_row["aoi_id"])
    if seen != set(index):
        raise ValueError("SFT/index IDs do not match exactly")
    summary = {
        "schema_version": "active-catalog-vlm-rl-state-summary-v1",
        "split": expected_split,
        "states": len(seen),
        "tasks": len(task_ids),
        "aois": len(aoi_ids),
        "target_action_counts": dict(sorted(action_counts.items())),
        "max_candidates": max_candidates,
        "sft_sha256": _sha256(sft_path),
        "index_sha256": _sha256(index_path),
        "output_sha256": _sha256(output),
        "online_closed_loop": False,
        "model_visible_utility": False,
        "test_assets_read": False,
    }
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sft", type=Path)
    parser.add_argument("evaluation_index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-split", choices=("train", "val"), required=True)
    parser.add_argument("--max-candidates", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(build_states(
        args.sft, args.evaluation_index, args.output,
        expected_split=args.expected_split, max_candidates=args.max_candidates,
    ), indent=2))


if __name__ == "__main__":
    main()
