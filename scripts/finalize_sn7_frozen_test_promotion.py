#!/usr/bin/env python3
"""Bind an SN7 promotion decision to its completed one-shot test ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(ledger_path: Path, raw_promotion_path: Path) -> dict[str, Any]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if (
        ledger.get("schema_version") != "activemap-frozen-test-access-v1"
        or ledger.get("status") != "complete"
        or ledger.get("returncode") != 0
    ):
        raise ValueError("SN7 frozen test ledger is not complete")
    promotion = json.loads(raw_promotion_path.read_text(encoding="utf-8"))
    if promotion.get("schema_version") != "active-catalog-tool-writeback-promotion-v1":
        raise ValueError("unexpected SN7 promotion schema")
    if promotion.get("test_assets_read") is not True:
        raise ValueError("SN7 promotion is not frozen-test evidence")
    if int(promotion.get("minimum_seed_count", 0)) < 3:
        raise ValueError("SN7 frozen promotion requires at least three seeds")
    return {
        **promotion,
        "formalized_from_completed_ledger": True,
        "frozen_test_ledger": str(ledger_path.resolve()),
        "frozen_test_ledger_sha256": _sha256(ledger_path),
        "raw_promotion": str(raw_promotion_path.resolve()),
        "raw_promotion_sha256": _sha256(raw_promotion_path),
        "claim_boundary": (
            "Three-seed one-shot frozen-test evidence for selective grounded tool use "
            "and executable SN7 map writeback."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("raw_promotion", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = finalize(args.ledger, args.raw_promotion)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
