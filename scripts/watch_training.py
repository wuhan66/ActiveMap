"""Terminal watcher for ActiveMap nightly run state and latest epoch metrics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _last_jsonl(path: Path) -> dict[str, Any] | None:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        return json.loads(lines[-1]) if lines else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def snapshot(run_root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"nightly": _read_json(run_root / "nightly_state.json")}
    for stage in ("updater", "selector"):
        stage_dir = run_root / stage
        state = _read_json(stage_dir / "state.json")
        latest = _last_jsonl(stage_dir / "history.jsonl")
        if state is not None or latest is not None:
            payload[stage] = {"state": state, "latest_epoch": latest}
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        print(json.dumps(snapshot(args.run_root), indent=2), flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
