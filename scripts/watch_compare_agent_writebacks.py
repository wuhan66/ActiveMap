#!/usr/bin/env python3
"""Wait for writebacks and run audited paired comparisons."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compare_agent_writebacks import compare


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def writeback_complete(path: Path) -> bool:
    state_path = path.parent.parent / "run_state.json"
    if not path.is_file() or not state_path.is_file():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return state.get("status") == "completed" and state.get("returncode") == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("LABEL", "BASELINE", "CANDIDATE"),
        required=True,
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--group-key", choices=("task_id", "aoi_id"), default="aoi_id")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    if args.bootstrap <= 0 or args.poll_seconds <= 0 or args.timeout_hours <= 0:
        raise ValueError("bootstrap, polling, and timeout values must be positive")
    labels = [row[0] for row in args.pair]
    if len(labels) != len(set(labels)):
        raise ValueError("comparison labels must be unique")

    pairs = [
        {"label": label, "baseline": Path(baseline), "candidate": Path(candidate)}
        for label, baseline, candidate in args.pair
    ]
    args.output_root.mkdir(parents=True)
    state_path = args.output_root / "watcher_state.json"
    deadline = time.monotonic() + args.timeout_hours * 3600.0
    write_json(
        state_path,
        {
            "schema_version": "writeback-comparison-watcher-v1",
            "status": "waiting_for_writebacks",
            "pairs": [{key: str(value) for key, value in row.items()} for row in pairs],
            "test_assets_read": False,
        },
    )
    required = {path for row in pairs for path in (row["baseline"], row["candidate"])}
    while not all(writeback_complete(path) for path in required):
        if time.monotonic() >= deadline:
            missing = sorted(
                str(path) for path in required if not writeback_complete(path)
            )
            write_json(
                state_path,
                {
                    "schema_version": "writeback-comparison-watcher-v1",
                    "status": "timed_out",
                    "missing": missing,
                    "test_assets_read": False,
                },
            )
            raise TimeoutError(f"writeback inputs did not complete: {missing}")
        time.sleep(args.poll_seconds)

    write_json(
        state_path,
        {
            "schema_version": "writeback-comparison-watcher-v1",
            "status": "comparing",
            "test_assets_read": False,
        },
    )
    summaries = {}
    for index, row in enumerate(pairs):
        result = compare(
            row["baseline"],
            row["candidate"],
            bootstrap=args.bootstrap,
            seed=args.seed + index,
            group_key=args.group_key,
            split=args.split,
        )
        output = args.output_root / f"{row['label']}.json"
        write_json(output, result)
        summaries[row["label"]] = {
            "output": str(output.resolve()),
            "paired_delta": result["paired_delta"],
        }
    write_json(
        args.output_root / "results.json",
        {
            "schema_version": "writeback-comparison-watcher-results-v1",
            "bootstrap": args.bootstrap,
            "group_key": args.group_key,
            "split": args.split,
            "comparisons": summaries,
            "test_assets_read": False,
        },
    )
    write_json(
        state_path,
        {
            "schema_version": "writeback-comparison-watcher-v1",
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "test_assets_read": False,
        },
    )


if __name__ == "__main__":
    main()
