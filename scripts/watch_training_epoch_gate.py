#!/usr/bin/env python3
"""Request graceful Trainer stops when active runs reach a frozen epoch gate."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json_if_ready(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, nargs="+")
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.epoch <= 0 or args.poll_seconds <= 0:
        raise ValueError("epoch and poll interval must be positive")
    if len(set(args.run)) != len(args.run):
        raise ValueError("run directories must be unique")

    pending = {path.resolve() for path in args.run}
    events: list[dict[str, Any]] = []
    while pending:
        for run in sorted(pending):
            state_path = run / "run_state.json"
            if not state_path.is_file():
                continue
            state = _read_json_if_ready(state_path)
            if state is None:
                continue
            status = str(state.get("status", "unknown"))
            event = state.get("last_training_event") or {}
            epoch = event.get("epoch")
            if status in {"completed", "failed"}:
                events.append({"run": str(run), "status": status, "epoch": epoch})
                pending.remove(run)
            elif epoch is not None and float(epoch) >= args.epoch:
                stop = run / "control" / "STOP"
                stop.parent.mkdir(parents=True, exist_ok=True)
                stop.write_text(
                    f"epoch_gate={args.epoch}\nobserved_epoch={epoch}\n",
                    encoding="utf-8",
                )
                events.append(
                    {
                        "run": str(run),
                        "status": "stop_requested",
                        "epoch": float(epoch),
                        "stop_file": str(stop),
                    }
                )
                pending.remove(run)
            _write(
                args.output,
                {
                    "schema_version": "training-epoch-gate-v1",
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "epoch_gate": args.epoch,
                    "pending": [str(path) for path in sorted(pending)],
                    "events": events,
                    "test_assets_read": False,
                },
            )
        if pending:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
