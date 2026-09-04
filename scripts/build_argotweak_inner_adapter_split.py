#!/usr/bin/env python3
"""Create a deterministic inner holdout for ArgoTweak adapter selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _indexed(values: list[str]) -> dict[str, str]:
    return {f"{index:05d}": value for index, value in enumerate(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-count", type=int, default=6)
    parser.add_argument("--salt", default="activemap-argotweak-lane-head-adapter-v1")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.holdout_count < 1:
        raise ValueError("holdout count must be positive")
    source = json.loads(args.subset.read_text(encoding="utf-8"))
    if source.get("test_assets_read") is not False:
        raise ValueError("source subset must explicitly exclude test assets")
    train_ids = list((source.get("splits") or {}).get("train", {}).values())
    outer_validation_ids = list((source.get("splits") or {}).get("val", {}).values())
    if args.holdout_count >= len(train_ids):
        raise ValueError("holdout must leave at least one training log")

    ranked = sorted(
        train_ids,
        key=lambda segment_id: hashlib.sha256(
            f"{args.salt}:{segment_id}".encode("utf-8")
        ).hexdigest(),
    )
    inner_holdout = ranked[: args.holdout_count]
    inner_train = sorted(set(train_ids) - set(inner_holdout))
    if len(inner_train) + len(inner_holdout) != len(train_ids):
        raise ValueError("duplicate segment IDs in source training split")

    output = {
        "schema_version": "activemap-argotweak-inner-adapter-split-v1",
        "source_subset": str(args.subset),
        "selection_policy": "fixed SHA256 rank over original training logs; no model outputs",
        "salt": args.salt,
        "test_assets_read": False,
        "splits": {
            "train": _indexed(inner_train),
            "val": _indexed(inner_holdout),
            "outer_confirmation": _indexed(outer_validation_ids),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "train_logs": len(inner_train),
                "inner_holdout_logs": len(inner_holdout),
                "outer_confirmation_logs": len(outer_validation_ids),
                "test_assets_read": False,
            }
        )
    )


if __name__ == "__main__":
    main()
