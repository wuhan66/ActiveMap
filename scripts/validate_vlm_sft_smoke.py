#!/usr/bin/env python3
"""Approve a completed VLM SFT smoke only when its peak memory is safe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smoke_root", type=Path)
    parser.add_argument("--max-peak-memory-mib", type=int, default=23500)
    args = parser.parse_args()
    results_path = args.smoke_root / "process_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    rows = results if isinstance(results, list) else results.get("results", [])
    if not rows or any(int(row.get("returncode", 1)) != 0 for row in rows):
        raise RuntimeError("smoke training did not complete successfully")
    peak = max(int(row.get("peak_memory_used_mib", 0)) for row in rows)
    if peak <= 0 or peak > args.max_peak_memory_mib:
        raise RuntimeError(
            f"smoke peak {peak} MiB is outside (0, {args.max_peak_memory_mib}]"
        )
    approval = {
        "schema_version": "vlm-sft-smoke-approval-v1",
        "peak_memory_used_mib": peak,
        "max_peak_memory_mib": args.max_peak_memory_mib,
        "batch_size": 2,
        "gradient_accumulation": 8,
    }
    (args.smoke_root / "smoke_approved.json").write_text(
        json.dumps(approval, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(approval, indent=2))


if __name__ == "__main__":
    main()
