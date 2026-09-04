#!/usr/bin/env python3
"""Restrict one validation trace to an immutable episode-budget manifest."""

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


def _manifest_keys(path: Path) -> set[tuple[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "val" or payload.get("test_assets_read") is not False:
        raise ValueError("manifest is not validation-only")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest contains no support records")
    keys = {(str(row["source_episode"]), float(row["budget"])) for row in records}
    if len(keys) != len(records):
        raise ValueError("manifest has duplicate episode-budget records")
    return keys


def filter_trace(trace_path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    keys = _manifest_keys(manifest_path)
    selected: dict[tuple[str, float], dict[str, Any]] = {}
    for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != "val" or row.get("test_assets_read") is not False:
            raise ValueError(f"trace row {line_number} is not validation-only")
        key = (str(row["source_episode"]), float(row["budget"]))
        if key not in keys:
            continue
        if key in selected:
            raise ValueError(f"trace repeats manifest identity: {key}")
        selected[key] = row
    if selected.keys() != keys:
        missing = sorted(keys - selected.keys())
        raise ValueError(f"trace does not exactly cover manifest support; missing={missing[:3]}")
    rows = [selected[key] for key in sorted(selected)]
    return rows, {
        "schema_version": "activemap-validation-trace-manifest-filter-v1",
        "split": "val",
        "test_assets_read": False,
        "record_count": len(rows),
        "trace": {"path": str(trace_path.resolve()), "sha256": _sha256(trace_path)},
        "manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256(manifest_path)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    rows, receipt = filter_trace(args.trace, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    receipt["filtered_trace"] = {
        "path": str(args.output.resolve()),
        "sha256": _sha256(args.output),
    }
    receipt_path = args.output.with_suffix(args.output.suffix + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
