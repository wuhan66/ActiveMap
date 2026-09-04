#!/usr/bin/env python3
"""Create a deterministic STOP/ACQUIRE visual-SFT smoke subset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _action(row: dict[str, Any]) -> str:
    content = row["messages"][-1]["content"]
    text = next(part["text"] for part in content if part.get("type") == "text")
    payload = json.loads(text)
    return str(payload.get("selection", payload.get("action", "UNKNOWN")))


def _select(path: Path, actions: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {action: [] for action in actions}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            action = _action(row)
            if action in buckets:
                buckets[action].append(row)
    missing = [action for action, rows in buckets.items() if not rows]
    if missing:
        raise ValueError(f"missing requested actions in {path}: {missing}")
    selected = [
        min(buckets[action], key=lambda row: str(row.get("example_id", "")))
        for action in actions
    ]
    result = copy.deepcopy(selected)
    for row in result:
        for part in row["messages"][1]["content"]:
            image = part.get("image")
            if part.get("type") == "image" and isinstance(image, str):
                image_path = Path(image)
                if not image_path.is_absolute():
                    part["image"] = str((path.parent / image_path).resolve())
    return result


def _write(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "records": len(rows),
        "actions": dict(sorted(Counter(_action(row) for row in rows).items())),
        "example_ids": [str(row.get("example_id")) for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--action", action="append", default=[])
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    actions = tuple(args.action or ["STOP", "ACQUIRE"])
    if len(set(actions)) != len(actions):
        raise ValueError("actions must be unique")
    for output in (args.train_output, args.val_output):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "visual-sft-balanced-smoke-v2",
        "train": _write(args.train_output, _select(args.train_jsonl, actions)),
        "val": _write(args.val_output, _select(args.val_jsonl, actions)),
        "test_assets_read": False,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.summary is not None:
        args.summary.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
