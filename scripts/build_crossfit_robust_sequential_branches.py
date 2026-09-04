#!/usr/bin/env python3
"""Build conservative sequential SELECT labels from independent policy snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _summary(root: Path) -> dict[str, Any]:
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("test_assets_read") is not False:
        raise ValueError(f"source violates the frozen-test contract: {root}")
    return summary


def _index(rows: list[dict[str, Any]], source: Path) -> dict[str, dict[str, Any]]:
    index = {str(row["example_id"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"duplicate example IDs in {source}")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onpolicy_train_root", type=Path)
    parser.add_argument("crossfit_branches_root", type=Path)
    parser.add_argument("validation_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")

    onpolicy_summary = _summary(args.onpolicy_train_root)
    validation_summary = _summary(args.validation_root)
    onpolicy_rows = _read_jsonl(args.onpolicy_train_root / "traces.jsonl")
    if any(row.get("split") != "train" for row in onpolicy_rows):
        raise ValueError("on-policy source must contain train rows only")
    onpolicy = _index(onpolicy_rows, args.onpolicy_train_root)

    crossfit: dict[str, dict[str, Any]] = {}
    crossfit_sources = []
    for root in sorted(args.crossfit_branches_root.glob("fold*")):
        summary = _summary(root)
        rows = _read_jsonl(root / "traces.jsonl")
        if any(row.get("split") != "train" for row in rows):
            raise ValueError(f"cross-fit source contains non-train rows: {root}")
        for example_id, row in _index(rows, root).items():
            if example_id in crossfit:
                raise ValueError(f"cross-fit example appears in more than one fold: {example_id}")
            crossfit[example_id] = row
        crossfit_sources.append(
            {
                "root": str(root.resolve()),
                "summary_sha256": _sha256(root / "summary.json"),
                "trace_sha256": _sha256(root / "traces.jsonl"),
                "adapter": summary.get("adapter"),
            }
        )
    if set(onpolicy) != set(crossfit):
        raise ValueError("on-policy and cross-fit train IDs differ")

    robust_train = []
    changed = 0
    for example_id, current in onpolicy.items():
        heldout = crossfit[example_id]
        if str(current["task_id"]) != str(heldout["task_id"]):
            raise ValueError(f"task mismatch for {example_id}")
        current_advantage = float(current["policy_relative_advantage"])
        crossfit_advantage = float(heldout["policy_relative_advantage"])
        robust_advantage = min(current_advantage, crossfit_advantage)
        selected = robust_advantage > 0.0
        if selected != bool(current["policy_relative_use_tool"]):
            changed += 1
        robust_train.append(
            {
                **current,
                "policy_relative_use_tool": selected,
                "policy_relative_advantage": robust_advantage,
                "robust_policy_advantage": robust_advantage,
                "onpolicy_advantage": current_advantage,
                "crossfit_advantage": crossfit_advantage,
                "label_protocol": "positive-only-if-onpolicy-and-task-heldout-crossfit-utility-positive",
            }
        )

    validation_rows = _read_jsonl(args.validation_root / "traces.jsonl")
    if any(row.get("split") != "val" for row in validation_rows):
        raise ValueError("validation source must contain validation rows only")
    output_rows = sorted(robust_train + validation_rows, key=lambda row: (str(row["split"]), str(row["example_id"])))
    args.output_root.mkdir(parents=True)
    traces_path = args.output_root / "traces.jsonl"
    with traces_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    trace_hash = _sha256(traces_path)
    summary = {
        "schema_version": "crossfit-robust-sequential-branch-cache-v1",
        "adapter": "crossfit-robust-policy-labels",
        "label_protocol": "minimum-realized-advantage-onpolicy-and-task-heldout-crossfit",
        "train_examples": len(robust_train),
        "validation_examples": len(validation_rows),
        "train_positive_count": sum(bool(row["policy_relative_use_tool"]) for row in robust_train),
        "labels_changed_from_onpolicy": changed,
        "trace_sha256": trace_hash,
        "sources": {
            "onpolicy_train": {
                "root": str(args.onpolicy_train_root.resolve()),
                "summary_sha256": _sha256(args.onpolicy_train_root / "summary.json"),
                "trace_sha256": _sha256(args.onpolicy_train_root / "traces.jsonl"),
                "adapter": onpolicy_summary.get("adapter"),
            },
            "crossfit_train": crossfit_sources,
            "validation": {
                "root": str(args.validation_root.resolve()),
                "summary_sha256": _sha256(args.validation_root / "summary.json"),
                "trace_sha256": _sha256(args.validation_root / "traces.jsonl"),
                "adapter": validation_summary.get("adapter"),
            },
        },
        "test_assets_read": False,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
