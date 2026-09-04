#!/usr/bin/env python3
"""Audit every visual SFT record for context and assistant-label coverage."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


def _write_progress(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _assistant_action(row: dict[str, Any]) -> str:
    content = row["messages"][-1]["content"]
    text = next(part["text"] for part in content if part.get("type") == "text")
    payload = json.loads(text)
    return str(payload.get("selection", payload.get("action", "UNKNOWN")))


def _distribution(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.int64)
    return {
        "minimum": int(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": int(array.max()),
        "mean": float(array.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("jsonl", type=Path, nargs="+")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-reported-offenders", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-output", type=Path)
    args = parser.parse_args()
    if args.max_length <= 0:
        raise ValueError("max-length must be positive")
    if args.max_reported_offenders <= 0:
        raise ValueError("max-reported-offenders must be positive")

    from transformers import AutoProcessor

    from activemap.agent.vlm_sft import encode_vlm_action_example, load_vlm_sft_rows

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    progress_output = args.progress_output
    if progress_output is None and args.output is not None:
        progress_output = args.output.parent / f"{args.output.stem}.progress.json"
    total_records = sum(
        1 for path in args.jsonl for line in path.open(encoding="utf-8") if line.strip()
    )
    processed_records = 0
    _write_progress(
        progress_output,
        {
            "status": "running",
            "processed_records": 0,
            "total_records": total_records,
            "current_split": None,
            "test_assets_read": False,
        },
    )
    splits: dict[str, Any] = {}
    total_over_limit = 0
    total_zero_labels = 0
    for path in args.jsonl:
        rows = load_vlm_sft_rows(path)
        lengths: list[int] = []
        supervised: list[int] = []
        over_limit: list[dict[str, Any]] = []
        zero_labels: list[str] = []
        actions: Counter[str] = Counter()
        iterator = tqdm(rows, desc=f"Token audit {path.stem}", unit="record")
        for row in iterator:
            encoded = encode_vlm_action_example(row, processor, max_length=10**9)
            length = int(encoded["input_ids"].numel())
            labels = int((encoded["labels"] != -100).sum().item())
            example_id = str(row.get("example_id", "UNKNOWN"))
            lengths.append(length)
            supervised.append(labels)
            actions[_assistant_action(row)] += 1
            if length > args.max_length:
                over_limit.append({"example_id": example_id, "tokens": length})
            if labels <= 0:
                zero_labels.append(example_id)
            processed_records += 1
            if processed_records % 100 == 0 or processed_records == total_records:
                _write_progress(
                    progress_output,
                    {
                        "status": "running",
                        "processed_records": processed_records,
                        "total_records": total_records,
                        "current_split": path.stem,
                        "over_max_length_records": total_over_limit + len(over_limit),
                        "zero_supervised_label_records": total_zero_labels
                        + len(zero_labels),
                        "test_assets_read": False,
                    },
                )
        over_limit.sort(key=lambda item: (-int(item["tokens"]), str(item["example_id"])))
        total_over_limit += len(over_limit)
        total_zero_labels += len(zero_labels)
        splits[path.stem] = {
            "path": str(path.resolve()),
            "records": len(rows),
            "actions": dict(sorted(actions.items())),
            "tokens": _distribution(lengths),
            "supervised_tokens": _distribution(supervised),
            "over_max_length_count": len(over_limit),
            "over_max_length_examples": over_limit[: args.max_reported_offenders],
            "zero_supervised_labels": zero_labels,
        }
    report = {
        "schema_version": "visual-sft-token-audit-v1",
        "model": str(Path(args.model).resolve()),
        "max_length": args.max_length,
        "splits": splits,
        "over_max_length_records": total_over_limit,
        "zero_supervised_label_records": total_zero_labels,
        "passed": total_over_limit == 0 and total_zero_labels == 0,
        "test_assets_read": False,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    _write_progress(
        progress_output,
        {
            "status": "complete",
            "processed_records": processed_records,
            "total_records": total_records,
            "over_max_length_records": total_over_limit,
            "zero_supervised_label_records": total_zero_labels,
            "passed": report["passed"],
            "test_assets_read": False,
        },
    )
    print(payload, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
