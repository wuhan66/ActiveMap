#!/usr/bin/env python3
"""Fit one policy-blind V5 Safe Commit gate for the non-KEEP mechanism study.

This script is intentionally separate from ``calibrate_sn7_v5_safe_commit``.
The registered V5 2x2 branch calibrates a gate over its direct/selected pair;
the mechanism experiment needs one predeclared gate fitted on the matched union
of every deployable selector policy.  The two protocols must never share an
output directory or calibration receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.calibrate_sn7_v5_safe_commit import (
    choose_common_threshold,
    load_rows,
    sha256,
)


POLICIES = (
    "direct",
    "random",
    "uncertainty",
    "generic",
    "policy_relative",
    "forced",
)


def parse_record(value: str) -> tuple[str, Path]:
    policy, separator, raw_path = value.partition("=")
    if not separator or policy not in POLICIES or not raw_path:
        raise argparse.ArgumentTypeError(
            "record must use one of "
            f"{', '.join(POLICIES)}=TRAIN_WRITEBACK_JSONL"
        )
    return policy, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--record", action="append", type=parse_record, required=True)
    parser.add_argument("--replay-iou-threshold", type=float, default=0.99)
    parser.add_argument("--allow-invalid-topology", action="store_true")
    parser.add_argument("--threshold-count", type=int, default=101)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {args.output}")
    if not 0.0 <= args.replay_iou_threshold <= 1.0 or args.threshold_count < 2:
        raise ValueError("invalid Safe Commit calibration grid")
    records = dict(args.record)
    if len(records) != len(args.record):
        raise ValueError("duplicate common Safe Commit policy record")
    if tuple(records) != POLICIES:
        raise ValueError(f"records must appear once in the fixed order: {POLICIES}")
    by_policy = {
        policy: load_rows(path, expected_split="train") for policy, path in records.items()
    }
    result = choose_common_threshold(
        by_policy,
        expected_policies=POLICIES,
        replay_iou_threshold=args.replay_iou_threshold,
        require_topology=not args.allow_invalid_topology,
        threshold_count=args.threshold_count,
    )
    result.update(
        {
            "schema_version": "sn7-v5-common-safe-commit-calibration-v1",
            "split": "train",
            "policy_blind": True,
            "inputs": {
                policy: {"path": str(path.resolve()), "sha256": sha256(path)}
                for policy, path in records.items()
            },
            "gate": {
                "confidence_threshold": result["selected"]["confidence_threshold"],
                "replay_iou_threshold": args.replay_iou_threshold,
                "require_topology": not args.allow_invalid_topology,
            },
            "test_assets_read": False,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
