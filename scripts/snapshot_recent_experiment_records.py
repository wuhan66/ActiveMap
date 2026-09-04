#!/usr/bin/env python3
"""Create a checksummed inventory of recent paper-facing experiment records."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


DEFAULT_PATTERNS = ("*.json", "*.jsonl", "*.csv", "*.md")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/RECENT_EXPERIMENT_MANIFEST.json"),
    )
    parser.add_argument(
        "--name-token",
        default="202607",
        help="Only inventory files whose names contain this token.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs_dir = args.docs_dir.resolve()
    output = args.output.resolve()
    paths = {
        path
        for pattern in DEFAULT_PATTERNS
        for path in docs_dir.glob(pattern)
        if args.name_token in path.name and path.resolve() != output
    }
    records = []
    for path in sorted(paths, key=lambda item: item.name):
        stat = path.stat()
        records.append(
            {
                "path": path.relative_to(docs_dir.parent).as_posix(),
                "bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
                "sha256": sha256(path),
            }
        )

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "name_token": args.name_token,
        "record_count": len(records),
        "records": records,
        "remote_canonical_roots": [
            "/home/wh/ActiveMap/runs/evidence_value_executable_val_20260725",
            "/home/wh/ActiveMap/runs/muno21_evidence_value_v1_20260725",
            "/home/wh/ActiveMap/runs/muno21_evidence_value_writeback_val_20260725",
            "/home/wh/ActiveMap/runs/muno21_evidence_value_official_val_v2_20260725",
            "/home/wh/ActiveMap/runs/muno21_graph_tolerance_ablation_20260725",
            "/home/wh/ActiveMap/runs/muno21_graph_tolerance3_three_seed_20260725",
        ],
        "active_experiments": [
            {
                "name": "MUNO21 graph simplification tolerance ablation",
                "status": "COMPLETED",
                "candidates": [0.0, 0.5, 1.0, 2.0, 3.0],
                "selected": 3.0,
                "note": (
                    "Tolerance 3.0 is frozen on validation for the primary "
                    "topology setting; 2.0 is retained as the safety-oriented point."
                ),
            },
            {
                "name": "MUNO21 tolerance 3.0 official three-seed replication",
                "status": "COMPLETED",
                "seeds": [1, 2, 3],
                "note": (
                    "Three-seed official validation is complete; APLS improves "
                    "significantly over Always STOP under hierarchical bootstrap."
                ),
            },
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
