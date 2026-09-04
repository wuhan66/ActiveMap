#!/usr/bin/env python3
"""Print the latest scalar values from a TensorBoard run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--last", type=int, default=5)
    args = parser.parse_args()

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = sorted(args.path.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"no TensorBoard events below {args.path}")
    accumulator = EventAccumulator(str(event_files[-1]), size_guidance={"scalars": 0})
    accumulator.Reload()
    output = {}
    for tag in accumulator.Tags().get("scalars", []):
        values = accumulator.Scalars(tag)[-args.last :]
        output[tag] = [
            {"step": value.step, "value": value.value, "wall_time": value.wall_time}
            for value in values
        ]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
