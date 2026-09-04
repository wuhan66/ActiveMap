#!/usr/bin/env python3
"""Freeze a target evidence-call rate from a training-only controller trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _support_sha256(rows: list[dict[str, Any]]) -> str:
    identities = []
    for row in rows:
        try:
            identities.append(
                (
                    str(row["source_episode"]),
                    float(row["budget"]),
                    str(row["aoi_id"]),
                )
            )
        except KeyError as error:
            raise ValueError(f"target-rate trace lacks support identity: {error.args[0]}") from error
    if len(set(identities)) != len(identities):
        raise ValueError("target-rate trace has duplicate episode-budget identities")
    payload = "\n".join(
        f"{episode}\t{budget:.8f}\t{aoi}" for episode, budget, aoi in sorted(identities)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_receipt(trace_path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("target-rate trace is empty")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("target-rate trace must contain training rows only")
    if any("acquisitions" not in row for row in rows):
        raise ValueError("target-rate trace lacks acquisition counts")
    if any(bool(row.get("test_assets_read")) for row in rows):
        raise ValueError("target-rate trace must not read test assets")
    acquisitions = [float(row["acquisitions"]) for row in rows]
    if any(value not in {0.0, 1.0} for value in acquisitions):
        raise ValueError(
            "matched-rate single-acquisition receipt requires every training episode to have 0 or 1 acquisition"
        )
    rate = sum(acquisitions) / len(rows)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(
            "matched-rate single-acquisition protocol requires a target in [0, 1]"
        )
    return {
        "schema_version": "activemap-train-call-rate-receipt-v1",
        "split": "train",
        "test_assets_read": False,
        "trace": {
            "path": str(trace_path.resolve()),
            "sha256": _sha256(trace_path),
            "record_count": len(rows),
        },
        "target_call_rate": rate,
        "target_call_count": sum(acquisitions),
        "support_sha256": _support_sha256(rows),
        "aoi_count": len({str(row["aoi_id"]) for row in rows}),
        "protocol": {
            "unit": "initial controller episode",
            "max_supported_acquisitions": 1,
            "derived_from": "fixed controller trace on training support",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    receipt = build_receipt(args.train_trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
