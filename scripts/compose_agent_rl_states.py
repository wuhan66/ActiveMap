#!/usr/bin/env python3
"""Compose base and grounded-tool GRPO states without resampling validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def compose_rl_states(inputs: list[Path], output: Path, *, expected_split: str) -> dict[str, Any]:
    if expected_split == "test":
        raise ValueError("RL composition must not read test")
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    source_counts: Counter[str] = Counter()
    best_counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    state_count = 0
    with output.open("w", encoding="utf-8") as destination:
        for path in inputs:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    row: dict[str, Any] = json.loads(line)
                    if row.get("split") != expected_split:
                        raise ValueError(f"split mismatch in {path}: {row.get('split')!r}")
                    identity = hashlib.sha256(
                        str(row["observation_json"]).encode("utf-8")
                    ).hexdigest()
                    if identity in seen:
                        raise ValueError(f"duplicate RL observation across inputs: {identity}")
                    seen.add(identity)
                    source_name = str(row.get("rl_source", "counterfactual_trajectory"))
                    row["rl_source"] = source_name
                    source_counts[source_name] += 1
                    best_counts[str(row["oracle_best_action_key"]).split(":", 1)[0]] += 1
                    serialized = json.dumps(row, separators=(",", ":"))
                    destination.write(serialized + "\n")
                    digest.update(serialized.encode("utf-8"))
                    digest.update(b"\n")
                    state_count += 1
    summary = {
        "schema_version": "activemap-composed-grpo-v2",
        "inputs": [str(path.resolve()) for path in inputs],
        "expected_split": expected_split,
        "state_count": state_count,
        "source_counts": dict(sorted(source_counts.items())),
        "oracle_best_action_counts": dict(sorted(best_counts.items())),
        "sha256": digest.hexdigest(),
        "resampled": False,
        "online_closed_loop": False,
        "test_assets_read": False,
    }
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--expected-split", choices=("train", "val"), required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            compose_rl_states(args.inputs, args.output, expected_split=args.expected_split),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
