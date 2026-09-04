#!/usr/bin/env python3
"""Print processor tensor shapes for a few real visual SFT records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from activemap.agent.vlm_sft import encode_vlm_action_example, load_vlm_sft_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("sft_jsonl", type=Path)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    rows = load_vlm_sft_rows(args.sft_jsonl, args.limit)
    payload = []
    for index, row in enumerate(rows):
        encoded = encode_vlm_action_example(row, processor, max_length=args.max_length)
        payload.append(
            {
                "index": index,
                "example_id": row.get("example_id"),
                "fields": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in sorted(encoded.items())
                },
            }
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
