#!/usr/bin/env python3
"""Restore primary SELECT labels and attach cross-fit reliability weights.

The primary on-policy realized advantage is the evaluation target.  A
task-held-out cross-fit advantage is useful evidence about label reliability,
but changing the primary target to a lower bound creates a train/evaluation
target shift.  This script keeps the primary target intact and emits a
train-only scalar weight for samples whose independent signs disagree.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _assistant_target(row: dict[str, Any]) -> tuple[int, int, dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("missing messages")
    for message_index in range(len(messages) - 1, -1, -1):
        message = messages[message_index]
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            raise ValueError("assistant content is invalid")
        for content_index, part in enumerate(content):
            if part.get("type") == "text":
                target = json.loads(str(part["text"]))
                if target.get("stage") != "SELECT":
                    raise ValueError("manifest contains a non-SELECT target")
                return message_index, content_index, target
    raise ValueError("missing assistant SELECT target")


def _set_selection(row: dict[str, Any], selection: str) -> dict[str, Any]:
    result = copy.deepcopy(row)
    message_index, content_index, target = _assistant_target(result)
    target["selection"] = selection
    result["messages"][message_index]["content"][content_index]["text"] = json.dumps(
        target, separators=(",", ":")
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selector_manifest", type=Path)
    parser.add_argument("branch_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--disagreement-weight", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not 0.0 < args.disagreement_weight <= 1.0:
        raise ValueError("disagreement weight must be in (0, 1]")

    source_summary_path = args.branch_root / "summary.json"
    source_trace_path = args.branch_root / "traces.jsonl"
    summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if summary.get("test_assets_read") is not False:
        raise ValueError("branch cache violates the frozen-test contract")
    branches = {
        str(row["example_id"]): row
        for row in _rows(source_trace_path)
        if str(row.get("split")) == args.split
    }
    manifest = _rows(args.selector_manifest)
    if {str(row.get("split")) for row in manifest} != {args.split}:
        raise ValueError("selector manifest split disagrees with --split")

    output_rows: list[dict[str, Any]] = []
    agreement_counts: Counter[str] = Counter()
    changed_labels = 0
    weights: list[float] = []
    for row in manifest:
        if str(row.get("stage")) != "SELECT":
            raise ValueError("selector manifest contains a non-SELECT row")
        trajectory_id = str(row["trajectory_id"])
        if not trajectory_id.startswith("seq-"):
            raise ValueError(f"unexpected selector trajectory ID: {trajectory_id}")
        branch = branches.get(trajectory_id.removeprefix("seq-"))
        if branch is None:
            raise ValueError(f"missing branch for {trajectory_id}")
        primary_source = (
            branch["onpolicy_advantage"]
            if "onpolicy_advantage" in branch
            else branch["policy_relative_advantage"]
        )
        primary_advantage = float(primary_source)
        primary_selection = "ACQUIRE" if primary_advantage > 0.0 else "STOP"
        _, _, previous_target = _assistant_target(row)
        if str(previous_target.get("selection")) != primary_selection:
            changed_labels += 1

        crossfit_advantage = branch.get("crossfit_advantage")
        if args.split == "train" and crossfit_advantage is None:
            raise ValueError("train branches require task-held-out cross-fit advantages")
        if crossfit_advantage is None:
            agreement = "not_applicable"
            record_weight = 1.0
        else:
            crossfit_selection = "ACQUIRE" if float(crossfit_advantage) > 0.0 else "STOP"
            agreement = "agree" if crossfit_selection == primary_selection else "disagree"
            record_weight = 1.0 if agreement == "agree" else args.disagreement_weight
        agreement_counts[agreement] += 1
        weights.append(record_weight)

        output = _set_selection(row, primary_selection)
        output.update(
            {
                "selected_tool": primary_selection == "ACQUIRE",
                "policy_relative_advantage": primary_advantage,
                "crossfit_reliability_weight": record_weight,
                "crossfit_label_agreement": agreement,
                "label_protocol": "primary-onpolicy-target-crossfit-train-reliability",
            }
        )
        output_rows.append(output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    result = {
        "schema_version": "crossfit-supported-selector-sft-v1",
        "split": args.split,
        "record_count": len(output_rows),
        "primary_acquire_count": sum(row["selected_tool"] for row in output_rows),
        "primary_acquire_rate": sum(row["selected_tool"] for row in output_rows)
        / len(output_rows),
        "changed_label_count": changed_labels,
        "crossfit_agreement_counts": dict(sorted(agreement_counts.items())),
        "disagreement_weight": args.disagreement_weight,
        "record_weight_mean": sum(weights) / len(weights),
        "selector_manifest": {
            "path": str(args.selector_manifest.resolve()),
            "sha256": _sha256(args.selector_manifest),
        },
        "branch_cache": {
            "path": str(args.branch_root.resolve()),
            "summary_sha256": _sha256(source_summary_path),
            "trace_sha256": _sha256(source_trace_path),
        },
        "output_sha256": _sha256(args.output),
        "test_assets_read": False,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
