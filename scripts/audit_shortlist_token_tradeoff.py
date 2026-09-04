#!/usr/bin/env python3
"""Sample the Qwen token/target-retention tradeoff across nested shortlists."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sample(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(str(row["example_id"]).encode()).hexdigest(),
    )[:count]


def _state(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    user_text = next(
        part for part in row["messages"][1]["content"] if part.get("type") == "text"
    )
    assistant_text = next(
        part for part in row["messages"][2]["content"] if part.get("type") == "text"
    )
    return json.loads(user_text["text"]), json.loads(assistant_text["text"])


def _truncate(row: dict[str, Any], size: int) -> tuple[dict[str, Any], bool]:
    result = copy.deepcopy(row)
    state, action = _state(result)
    candidates = state["candidate_evidence"][:size]
    state["candidate_evidence"] = candidates
    user_text = next(
        part for part in result["messages"][1]["content"] if part.get("type") == "text"
    )
    user_text["text"] = json.dumps(state, separators=(",", ":"))
    target = action.get("evidence_id")
    retained = target is None or any(item.get("evidence_id") == target for item in candidates)
    return result, retained


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples-per-split", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--size", type=int, action="append", default=[])
    args = parser.parse_args()
    if args.samples_per_split < 1 or args.max_length < 1:
        raise ValueError("sample count and max length must be positive")

    from transformers import AutoProcessor

    from activemap.agent.vlm_sft import encode_vlm_action_example, load_vlm_sft_rows

    sizes = sorted(set(args.size or [18, 24, 32, 36, 48]))
    if sizes[-1] > 48:
        raise ValueError("exact audit sizes cannot exceed the h48 source corpus")
    rows = []
    for path in (args.train_jsonl, args.val_jsonl):
        rows.extend(_sample(load_vlm_sft_rows(path), args.samples_per_split))
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    lengths: dict[int, list[int]] = {size: [] for size in sizes}
    retained: dict[int, list[bool]] = {size: [] for size in sizes}
    acquire_retained: dict[int, list[bool]] = {size: [] for size in sizes}
    for row in rows:
        _, original_action = _state(row)
        is_acquire = original_action.get("selection") == "ACQUIRE"
        for size in sizes:
            candidate, target_retained = _truncate(row, size)
            encoded = encode_vlm_action_example(candidate, processor, max_length=10**9)
            lengths[size].append(int(encoded["input_ids"].numel()))
            retained[size].append(target_retained)
            if is_acquire:
                acquire_retained[size].append(target_retained)

    exact = {
        f"h{size}": {
            "tokens": _distribution(lengths[size]),
            "over_limit_fraction": float(np.mean(np.asarray(lengths[size]) > args.max_length)),
            "target_retained_fraction": float(np.mean(retained[size])),
            "acquire_target_retained_fraction": (
                float(np.mean(acquire_retained[size])) if acquire_retained[size] else 1.0
            ),
        }
        for size in sizes
    }
    if 36 in lengths and 48 in lengths:
        slope = (np.asarray(lengths[48]) - np.asarray(lengths[36])) / 12.0
        extrapolated = {}
        for size in (54, 60):
            estimates = np.asarray(lengths[48]) + slope * (size - 48)
            extrapolated[f"h{size}"] = {
                "tokens": _distribution(estimates.tolist()),
                "estimated_over_limit_fraction": float(np.mean(estimates > args.max_length)),
                "method": "per-example linear extrapolation from h36 to h48",
            }
    else:
        extrapolated = {}
    report = {
        "schema_version": "shortlist-token-tradeoff-v1",
        "model": str(Path(args.model).resolve()),
        "samples": len(rows),
        "samples_per_split": args.samples_per_split,
        "max_length": args.max_length,
        "exact": exact,
        "extrapolated": extrapolated,
        "test_assets_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
