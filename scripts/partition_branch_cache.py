#!/usr/bin/env python3
"""Partition an audited branch cache into one split-specific receipt."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    source_summary_path = args.source_root / "summary.json"
    source_traces_path = args.source_root / "traces.jsonl"
    source_summary: dict[str, Any] = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source_summary.get("test_assets_read") is not False:
        raise ValueError("source violates the frozen-test contract")
    rows = [
        json.loads(line)
        for line in source_traces_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = [row for row in rows if str(row.get("split")) == args.split]
    if not selected:
        raise ValueError(f"source contains no {args.split} rows")
    if any(str(row.get("split")) != args.split for row in selected):
        raise RuntimeError("partition retained an invalid split")
    ids = [str(row["example_id"]) for row in selected]
    if len(set(ids)) != len(ids):
        raise ValueError("partition contains duplicate example IDs")
    args.output_root.mkdir(parents=True)
    traces_path = args.output_root / "traces.jsonl"
    with traces_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "split-branch-cache-v1",
        "adapter": source_summary.get("adapter"),
        "split": args.split,
        "sample_count": len(selected),
        "task_count": len({str(row["task_id"]) for row in selected}),
        "trace_sha256": _sha256(traces_path),
        "source": {
            "root": str(args.source_root.resolve()),
            "summary_sha256": _sha256(source_summary_path),
            "trace_sha256": _sha256(source_traces_path),
        },
        "test_assets_read": False,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
