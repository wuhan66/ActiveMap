#!/usr/bin/env python3
"""Build deduplicated natural-prior Tool-Need features from Agent SFT records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from activemap.agent.records import AgentAction, AgentActionType, AgentObservation
from activemap.agent.tool_need_gate import (
    TOOL_NEED_FEATURE_NAMES,
    structured_tool_need_features,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(input_path: Path, output_dir: Path, *, split: str) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    features: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    seen: dict[str, bool] = {}
    duplicate_count = 0
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            messages = payload.get("messages", [])
            user = next((item for item in messages if item.get("role") == "user"), None)
            assistant = next(
                (item for item in reversed(messages) if item.get("role") == "assistant"),
                None,
            )
            if user is None or assistant is None:
                raise ValueError(f"missing user/assistant message at line {line_number}")
            observation = AgentObservation.model_validate_json(user["content"])
            action = AgentAction.model_validate_json(assistant["content"])
            if observation.split != split:
                raise ValueError(f"expected split={split} at line {line_number}")
            canonical = observation.model_dump_json(exclude_none=True)
            example_id = hashlib.sha256(canonical.encode()).hexdigest()[:24]
            label = action.action == AgentActionType.USE_TOOL
            if example_id in seen:
                if seen[example_id] != label:
                    raise ValueError(f"conflicting duplicate Tool-Need label: {example_id}")
                duplicate_count += 1
                continue
            seen[example_id] = label
            features.append(structured_tool_need_features(observation))
            records.append(
                {
                    "example_id": example_id,
                    "task_id": observation.task_id,
                    "split": split,
                    "oracle_use_tool": label,
                    "available_tool_count": len(observation.available_tools),
                    "source_line": line_number,
                }
            )
    if not records or not any(row["oracle_use_tool"] for row in records):
        raise ValueError("Tool-Need feature set must contain positive and negative records")
    matrix = np.stack(features).astype(np.float32)
    output_dir.mkdir(parents=True)
    np.save(output_dir / "features.npy", matrix)
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    counts = Counter(bool(row["oracle_use_tool"]) for row in records)
    summary: dict[str, object] = {
        "schema_version": "structured-tool-need-features-v1",
        "split": split,
        "records": len(records),
        "tasks": len({str(row["task_id"]) for row in records}),
        "feature_dim": int(matrix.shape[1]),
        "feature_names": TOOL_NEED_FEATURE_NAMES,
        "positive_count": counts[True],
        "negative_count": counts[False],
        "positive_rate": counts[True] / len(records),
        "deduplicated_records": duplicate_count,
        "utility_metadata": "none",
        "source": {"path": str(input_path), "sha256": _sha256(input_path)},
        "test_assets_read": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output_dir, split=args.split), indent=2))


if __name__ == "__main__":
    main()
