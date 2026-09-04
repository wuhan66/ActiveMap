#!/usr/bin/env python3
"""Promote one temporal updater checkpoint after validation-only calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from activemap.evaluation.updater_temporal_promotion import select_temporal_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME=CALIBRATION_JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibrations = {}
    for item in args.candidate:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            parser.error(f"invalid --candidate value: {item}")
        calibrations[name] = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    decision = select_temporal_checkpoint(calibrations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
