#!/usr/bin/env python3
"""Audit grounded Active-Catalog tool pairs before Tool-Belief training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from activemap.agent.tool_belief_data import PostAcquisitionToolPairExample
from activemap.agent.tool_features import TOOL_RESULT_FEATURE_NAMES, encode_tool_result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path, expected_split: str) -> tuple[list[PostAcquisitionToolPairExample], dict[str, Any]]:
    rows = []
    example_ids = set()
    transition_ids = set()
    values = {name: set() for name in TOOL_RESULT_FEATURE_NAMES}
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = PostAcquisitionToolPairExample.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc
            if row.split != expected_split:
                raise ValueError(f"unexpected split at {path}:{line_number}")
            transition_id = row.metadata.get("source_transition_id")
            if not isinstance(transition_id, str) or not transition_id:
                raise ValueError("grounded pair lacks source_transition_id")
            if row.example_id in example_ids or transition_id in transition_ids:
                raise ValueError("duplicate example or source transition identity")
            if row.metadata.get("selected_by_model") is not True:
                raise ValueError("grounded pair was not selected by the model")
            if row.metadata.get("oracle_next_state_replay") is not False:
                raise ValueError("grounded pair permits oracle next-state replay")
            if row.metadata.get("oracle_action_exported") is not False:
                raise ValueError("grounded pair exports oracle action")
            for result in (row.quality_result, row.temporal_result):
                if result.outputs.get("evidence_id") != row.evidence_id:
                    raise ValueError("tool result and selected evidence disagree")
                for name, value in zip(
                    TOOL_RESULT_FEATURE_NAMES, encode_tool_result(result), strict=True
                ):
                    values[name].add(value)
            example_ids.add(row.example_id)
            transition_ids.add(transition_id)
            rows.append(row)
            counts[f"target:{row.gt_edit.value}"] += 1
    if not rows:
        raise ValueError(f"empty grounded tool data: {path}")
    variable_features = sorted(name for name, items in values.items() if len(items) > 1)
    nonzero_features = sorted(
        name for name, items in values.items() if any(value != 0.0 for value in items)
    )
    return rows, {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "split": expected_split,
        "examples": len(rows),
        "tasks": len({row.task_id for row in rows}),
        "unique_examples": len(example_ids),
        "unique_source_transitions": len(transition_ids),
        "counts": dict(sorted(counts.items())),
        "nonzero_tool_features": nonzero_features,
        "variable_tool_features": variable_features,
        "test_assets_read": False,
    }


def audit(train_path: Path, val_path: Path) -> dict[str, Any]:
    train, train_summary = _read(train_path, "train")
    val, val_summary = _read(val_path, "val")
    task_overlap = {row.task_id for row in train} & {row.task_id for row in val}
    example_overlap = {row.example_id for row in train} & {row.example_id for row in val}
    transition_overlap = {
        str(row.metadata["source_transition_id"]) for row in train
    } & {str(row.metadata["source_transition_id"]) for row in val}
    if task_overlap or example_overlap or transition_overlap:
        raise ValueError("grounded train/validation tool data leaks identities")
    if not train_summary["nonzero_tool_features"]:
        raise ValueError("train tool features are all zero")
    if not val_summary["nonzero_tool_features"]:
        raise ValueError("validation tool features are all zero")
    return {
        "schema_version": "active-catalog-grounded-tool-audit-v1",
        "passed": True,
        "train": train_summary,
        "val": val_summary,
        "task_overlap": 0,
        "example_overlap": 0,
        "source_transition_overlap": 0,
        "model_selected_state_transitions": True,
        "explicit_geospatial_tool_calls": True,
        "oracle_next_state_replay": False,
        "oracle_action_exported": False,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=Path)
    parser.add_argument("val", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    summary = audit(args.train, args.val)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
