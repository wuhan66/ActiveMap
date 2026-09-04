#!/usr/bin/env python3
"""Audit composed action-SFT records used by the recurrent controller."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VALID_ACTIONS = {"ACQUIRE", "USE_TOOL", "COMMIT", "REJECT"}
VALID_TOOLS = {"IMAGE_QUALITY", "TEMPORAL_CHANGE"}


def _read(path: Path, expected_split: str) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    trajectory_ids: list[str] = []
    identities: list[tuple[str, int]] = []
    invalid_rows = 0
    split_mismatches = 0
    tool_rows = 0
    grounded_tool_rows = 0
    pointer_tool_rows = 0
    tool_schema_errors = 0
    unavailable_tool_rows = 0
    ungrounded_tool_rows = 0

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                messages = row["messages"]
                if not isinstance(messages, list) or len(messages) != 3:
                    raise ValueError("messages must contain system/user/assistant")
                observation = json.loads(str(messages[1]["content"]))
                action = json.loads(str(messages[2]["content"]))
                action_name = str(action["action"])
                if action_name not in VALID_ACTIONS:
                    raise ValueError(f"unknown action: {action_name}")
                row_split = str(observation["split"])
                if row_split != expected_split:
                    split_mismatches += 1
                trajectory_id = str(row["trajectory_id"])
                step = int(row["step"])
                trajectory_ids.append(trajectory_id)
                identities.append((trajectory_id, step))
                action_counts[action_name] += 1
                source_counts[str(row.get("composition_source", "UNKNOWN"))] += 1

                if action_name == "USE_TOOL":
                    tool_rows += 1
                    call = action.get("tool_call")
                    if not isinstance(call, dict):
                        tool_schema_errors += 1
                        continue
                    tool = str(call.get("tool", ""))
                    inputs = call.get("inputs")
                    evidence_id = inputs.get("evidence_id") if isinstance(inputs, dict) else None
                    evidence_index = inputs.get("evidence_index") if isinstance(inputs, dict) else None
                    available = {str(value) for value in observation.get("available_tools", [])}
                    selected = [str(value) for value in observation.get("selected_evidence_ids", [])]
                    if tool not in VALID_TOOLS or not isinstance(inputs, dict):
                        tool_schema_errors += 1
                    elif tool not in available:
                        unavailable_tool_rows += 1
                    elif evidence_id is not None and str(evidence_id) in selected:
                        grounded_tool_rows += 1
                    elif (
                        evidence_id is None
                        and isinstance(evidence_index, int)
                        and 0 <= evidence_index < len(selected)
                    ):
                        # Pointer actions intentionally avoid opaque runtime IDs.
                        pointer_tool_rows += 1
                        grounded_tool_rows += 1
                    elif evidence_id is None and evidence_index is not None:
                        tool_schema_errors += 1
                    else:
                        ungrounded_tool_rows += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                invalid_rows += 1
                if invalid_rows <= 3:
                    print(f"warning: {path}:{line_number}: {exc}")

    total = sum(action_counts.values())
    duplicate_identity_count = len(identities) - len(set(identities))
    return {
        "path": str(path.resolve()),
        "split": expected_split,
        "records": total + invalid_rows,
        "valid_records": total,
        "invalid_rows": invalid_rows,
        "split_mismatches": split_mismatches,
        "unique_trajectory_count": len(set(trajectory_ids)),
        "duplicate_identity_count": duplicate_identity_count,
        "action_counts": dict(sorted(action_counts.items())),
        "action_fractions": {
            key: value / max(total, 1) for key, value in sorted(action_counts.items())
        },
        "composition_source_counts": dict(sorted(source_counts.items())),
        "tool_rows": tool_rows,
        "grounded_tool_rows": grounded_tool_rows,
        "pointer_tool_rows": pointer_tool_rows,
        "tool_schema_errors": tool_schema_errors,
        "unavailable_tool_rows": unavailable_tool_rows,
        "ungrounded_tool_rows": ungrounded_tool_rows,
        "tool_grounding_rate": grounded_tool_rows / max(tool_rows, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=Path)
    parser.add_argument("val", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = {
        "schema_version": "activemap-action-sft-audit-v1",
        "train": _read(args.train, "train"),
        "val": _read(args.val, "val"),
        "test_assets_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if any(
        report[split][field] > 0
        for split in ("train", "val")
        for field in ("invalid_rows", "split_mismatches", "tool_schema_errors", "unavailable_tool_rows", "ungrounded_tool_rows")
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
