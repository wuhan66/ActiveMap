#!/usr/bin/env python3
"""Build leakage-separated Direct-VLM SFT records from prepared MUNO21 visuals."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_muno21_direct_vlm import SYSTEM_PROMPTS, _prompt_text
except ModuleNotFoundError:
    from evaluate_muno21_direct_vlm import SYSTEM_PROMPTS, _prompt_text


OPERATIONS = ("KEEP", "ADD", "DELETE", "RESHAPE")


def _load(path: Path, split: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(row.get("split") != split for row in rows):
        raise ValueError(f"{path} must contain non-empty {split} records only")
    return rows


def _balance(
    records: list[tuple[dict[str, Any], str]]
) -> list[tuple[dict[str, Any], str, int]]:
    groups: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for record, operation in records:
        groups[operation].append((record, operation))
    missing = [operation for operation in OPERATIONS if not groups[operation]]
    if missing:
        raise ValueError(f"cannot balance missing operations: {missing}")
    target = max(len(groups[operation]) for operation in OPERATIONS)
    output = []
    for operation in OPERATIONS:
        group = groups[operation]
        for index in range(target):
            record, label = group[index % len(group)]
            output.append((record, label, index // len(group)))
    return output


def _assistant(operation: str) -> str:
    if operation == "KEEP":
        return '{"action":"REJECT"}'
    if operation not in OPERATIONS:
        raise ValueError(f"unknown operation: {operation}")
    return f'{{"action":"COMMIT","edit":"{operation}"}}'


def build_rows(
    inputs: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    split: str,
    balance: bool,
) -> list[dict[str, Any]]:
    labels_by_id = {str(row["example_id"]): str(row["edit"]) for row in labels}
    if set(labels_by_id) != {str(row["example_id"]) for row in inputs}:
        raise ValueError("input and label supports differ")
    source = [(row, labels_by_id[str(row["example_id"])]) for row in inputs]
    expanded = _balance(source) if balance else [
        (record, operation, 0) for record, operation in source
    ]
    rows = []
    for record, operation, repeat in expanded:
        rows.append(
            {
                "schema_version": "muno21-direct-vlm-sft-v1",
                "example_id": f"{record['example_id']}__r{repeat}",
                "source_example_id": record["example_id"],
                "split": split,
                "operation": operation,
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": SYSTEM_PROMPTS["operational_v2"]}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": record["images"]["composite"],
                            },
                            {
                                "type": "text",
                                "text": _prompt_text(
                                    "operational_v2", "composite"
                                ),
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": _assistant(operation)}],
                    },
                ],
                "test_assets_read": False,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--balance", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    inputs = _load(args.inputs, args.split)
    labels = _load(args.labels, args.split)
    rows = build_rows(inputs, labels, split=args.split, balance=args.balance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "split": args.split,
        "balanced": args.balance,
        "source_samples": len(inputs),
        "output_samples": len(rows),
        "operation_counts": dict(sorted(Counter(row["operation"] for row in rows).items())),
        "test_assets_read": False,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
