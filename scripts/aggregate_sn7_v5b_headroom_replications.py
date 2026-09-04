#!/usr/bin/env python3
"""Fail closed when authorizing V5-B matched-controller evaluation.

This tool never aggregates model scores. It validates that each independently
trained V5-B updater passed the same validation-only candidate-headroom gate,
then emits a hash-pinned three-seed authorization receipt for the downstream
factorial writeback experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


OPERATIONS = ("ADD", "DELETE", "RESHAPE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_record(value: str) -> tuple[int, Path, Path]:
    try:
        raw_seed, raw_paths = value.split("=", 1)
        raw_summary, raw_checkpoint = raw_paths.split(",", 1)
        return int(raw_seed), Path(raw_summary), Path(raw_checkpoint)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "record must be SEED=HEADROOM_SUMMARY,CHECKPOINT_RECEIPT"
        ) from exc


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def validate_record(seed: int, summary_path: Path, receipt_path: Path) -> dict[str, Any]:
    summary = load_json(summary_path)
    receipt = load_json(receipt_path)
    errors: list[str] = []
    if summary.get("schema_version") != "sn7-nonkeep-candidate-headroom-v1":
        errors.append("unexpected headroom schema")
    if summary.get("test_assets_read") is not False:
        errors.append("headroom summary is not validation-only")
    gate = summary.get("gate")
    if not isinstance(gate, dict) or gate.get("passes_candidate_recovery_preflight") is not True:
        errors.append("aggregate candidate-recovery gate did not pass")
    slices = summary.get("slices")
    if not isinstance(slices, dict):
        errors.append("headroom slices are missing")
        slices = {}
    operations: dict[str, Any] = {}
    for operation in OPERATIONS:
        entry = slices.get(operation)
        if not isinstance(entry, dict):
            errors.append(f"{operation} slice is missing")
            continue
        interval = entry.get("safe_map_headroom_ci95")
        lower = float(interval[0]) if isinstance(interval, list) and len(interval) == 2 else None
        if gate is None or gate.get("passes_by_operation", {}).get(operation) is not True:
            errors.append(f"{operation} gate did not pass")
        if lower is None or lower <= 0.0:
            errors.append(f"{operation} lower headroom bound is not strictly positive")
        operations[operation] = {
            "rows": int(entry.get("row_count", 0)),
            "aois": int(entry.get("aoi_count", 0)),
            "safe_map_headroom_ci95": interval,
            "passes_preflight": entry.get("passes_preflight"),
        }
    checkpoint = receipt.get("checkpoint")
    checkpoint_hash = receipt.get("checkpoint_sha256")
    if not isinstance(checkpoint, str) or not checkpoint:
        errors.append("checkpoint receipt has no checkpoint path")
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
        errors.append("checkpoint receipt has no SHA-256")
    return {
        "seed": seed,
        "passed": not errors,
        "errors": errors,
        "headroom_summary": str(summary_path.resolve()),
        "headroom_summary_sha256": sha256(summary_path),
        "checkpoint_receipt": str(receipt_path.resolve()),
        "checkpoint_receipt_sha256": sha256(receipt_path),
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_hash,
        "operations": operations,
        "state_file": summary.get("states"),
        "state_file_sha256": summary.get("states_sha256"),
        "test_assets_read": summary.get("test_assets_read"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--record",
        type=parse_record,
        action="append",
        required=True,
        metavar="SEED=HEADROOM_SUMMARY,CHECKPOINT_RECEIPT",
    )
    parser.add_argument("--expected-seeds", default="20260817,20260818,20260819")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = dict()
    for seed, summary, receipt in args.record:
        if seed in records:
            raise ValueError(f"duplicate seed: {seed}")
        records[seed] = validate_record(seed, summary, receipt)
    expected = tuple(int(value) for value in args.expected_seeds.split(",") if value.strip())
    missing = sorted(set(expected) - set(records))
    unexpected = sorted(set(records) - set(expected))
    if missing or unexpected:
        raise ValueError(f"seed set mismatch; missing={missing}, unexpected={unexpected}")
    result = {
        "schema_version": "sn7-v5b-three-seed-headroom-authorization-v1",
        "expected_seeds": list(expected),
        "records": [records[seed] for seed in sorted(records)],
        "authorization": "matched_nonkeep_factorial_writeback"
        if all(record["passed"] for record in records.values())
        else "blocked",
        "test_assets_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["authorization"] != "matched_nonkeep_factorial_writeback":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
