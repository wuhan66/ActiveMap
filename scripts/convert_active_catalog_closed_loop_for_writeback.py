#!/usr/bin/env python3
"""Convert recurrent active-catalog traces to executable vector-writeback rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from activemap.agent.identifiers import public_task_id
from activemap.models import EditOperation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_trace(
    trace: dict[str, Any],
    *,
    split: str = "val",
    evidence_mode: str = "all",
) -> dict[str, Any]:
    expected_test = split == "test"
    if trace.get("split") != split or trace.get("test_assets_read") is not expected_test:
        label = "validation" if split == "val" else "test"
        raise ValueError(f"closed-loop writeback trace is not audited {label} evidence")
    prediction = EditOperation(str(trace["predicted_edit"]))
    target = EditOperation(str(trace["target_edit"]))
    source_selected = [str(item) for item in trace["selected_evidence_ids"]]
    if not source_selected:
        raise ValueError("writeback requires at least the initial visual evidence")
    if evidence_mode == "all":
        selected = source_selected
    elif evidence_mode == "first":
        selected = source_selected[:1]
    elif evidence_mode == "last":
        selected = source_selected[-1:]
    else:
        raise ValueError(f"unsupported writeback evidence mode: {evidence_mode}")
    return {
        "task_id": public_task_id(str(trace["source_episode"])),
        "aoi_id": str(trace["aoi_id"]),
        "budget": float(trace["budget"]),
        "target": "REJECT" if target == EditOperation.KEEP else f"COMMIT:{target.value}",
        "prediction": (
            "REJECT" if prediction == EditOperation.KEEP else f"COMMIT:{prediction.value}"
        ),
        "selected_evidence_ids": selected,
        "source_selected_evidence_ids": source_selected,
        "writeback_evidence_mode": evidence_mode,
        "semantic_tool_called": int(trace.get("tool_calls", 0)) > 0,
        "semantic_tool_cost": float(trace.get("tool_cost", 0.0)),
        "policy_utility": float(trace["quality_cost_utility"]),
        "spent_cost": float(trace["spent_cost"]),
        "source_example_id": str(trace["sample_id"]),
        "source_policy": str(trace["policy"]),
        "split": split,
        "test_assets_read": expected_test,
    }


def convert_file(
    source: Path,
    output: Path,
    *,
    split: str = "val",
    frozen_test: bool = False,
    evidence_mode: str = "all",
) -> dict[str, Any]:
    test_assets_read = split == "test"
    if test_assets_read:
        if not frozen_test:
            raise PermissionError("test conversion requires --frozen-test")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    elif frozen_test:
        raise ValueError("--frozen-test is valid only for the test split")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    traces = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not traces:
        raise ValueError("no closed-loop traces")
    rows = [
        convert_trace(trace, split=split, evidence_mode=evidence_mode)
        for trace in traces
    ]
    identities = [(row["task_id"], row["budget"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate closed-loop writeback identity")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "active-catalog-closed-loop-writeback-v1",
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
        "record_count": len(rows),
        "aoi_count": len({row["aoi_id"] for row in rows}),
        "policies": sorted({row["source_policy"] for row in rows}),
        "evidence_mode": evidence_mode,
        "source_multi_evidence_rows": sum(
            len(row["source_selected_evidence_ids"]) > 1 for row in rows
        ),
        "mean_writeback_evidence_count": sum(
            len(row["selected_evidence_ids"]) for row in rows
        )
        / len(rows),
        "split": split,
        "test_assets_read": test_assets_read,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--evidence-mode", choices=("all", "first", "last"), default="all"
    )
    parser.add_argument("--frozen-test", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            convert_file(
                args.source,
                args.output,
                split=args.split,
                frozen_test=args.frozen_test,
                evidence_mode=args.evidence_mode,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
