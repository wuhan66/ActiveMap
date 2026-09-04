#!/usr/bin/env python3
"""Freeze V5 qualitative cases before controller writeback results are read."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from activemap.agent.identifiers import public_task_id
from activemap.selector_records import SelectorSample


OPERATIONS = ("ADD", "DELETE", "RESHAPE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_cases(
    states: Path, *, per_operation: int, seed: int
) -> dict[str, Any]:
    if per_operation < 1:
        raise ValueError("per_operation must be positive")
    candidates: dict[str, dict[str, SelectorSample]] = defaultdict(dict)
    with states.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sample = SelectorSample.model_validate_json(line)
            if sample.split != "val":
                raise ValueError("qualitative manifest accepts only validation states")
            if int(sample.metadata.get("oracle_step", -1)) != 0:
                continue
            operation = str(sample.metadata.get("gt_edit", ""))
            if operation not in OPERATIONS:
                continue
            source = str(sample.metadata.get("source_episode", ""))
            if not source:
                raise ValueError(f"state {line_number} lacks source_episode")
            candidates[operation].setdefault(source, sample)
    selected = []
    for operation in OPERATIONS:
        pool = candidates[operation]
        if len(pool) < per_operation:
            raise ValueError(f"{operation} has only {len(pool)} source episodes")
        ordered = sorted(
            pool.items(),
            key=lambda item: hashlib.sha256(
                f"{seed}:{operation}:{item[0]}".encode("utf-8")
            ).hexdigest(),
        )
        for source, sample in ordered[:per_operation]:
            selected.append(
                {
                    "operation": operation,
                    "source_episode": source,
                    "task_id": public_task_id(source),
                    "aoi_id": str(sample.metadata["aoi_id"]),
                    "budget": float(sample.metadata["budget"]),
                    "state_id": sample.sample_id,
                    "initial_evidence_ids": list(sample.metadata["selected_evidence_ids"]),
                }
            )
    return {
        "schema_version": "sn7-v5-predeclared-qualitative-manifest-v1",
        "selection_rule": "lowest SHA-256(seed, operation, source_episode); no score, outcome, or image-quality ranking",
        "seed": seed,
        "per_operation": per_operation,
        "states": str(states.resolve()),
        "states_sha256": sha256(states),
        "cases": selected,
        "split": "val",
        "controller_or_writeback_outputs_read": False,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-operation", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite qualitative manifest: {args.output}")
    result = select_cases(args.states, per_operation=args.per_operation, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
