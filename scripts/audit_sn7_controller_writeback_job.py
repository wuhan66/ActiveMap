#!/usr/bin/env python3
"""Audit one SN7 controller/writeback job before accepting completion."""

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


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _closed_identities(path: Path) -> tuple[set[tuple[str, float]], int]:
    identities = set()
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("split") != "val"
                or row.get("test_assets_read") is not False
                or row.get("policy") != "edit_utility"
            ):
                raise ValueError(f"invalid closed-loop row at line {line_number}")
            identity = (str(row["sample_id"]), float(row["budget"]))
            if identity in identities:
                raise ValueError(f"duplicate closed-loop identity: {identity}")
            identities.add(identity)
            count += 1
    return identities, count


def _writeback_identities(path: Path) -> tuple[set[tuple[str, float]], int]:
    identities = set()
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "val" or row.get("test_assets_read") is not False:
                raise ValueError(f"invalid writeback row at line {line_number}")
            if not row.get("mask_artifact"):
                raise ValueError(f"missing mask artifact receipt at line {line_number}")
            identity = (str(row["source_example_id"]), float(row["budget"]))
            if identity in identities:
                raise ValueError(f"duplicate writeback identity: {identity}")
            identities.add(identity)
            count += 1
    return identities, count


def audit(
    job_root: Path,
    *,
    expected_rows: int,
    protocol_name: str,
    translation_pixels: int,
    morphology: str,
    morphology_pixels: int,
    corruption_seed: int,
) -> dict[str, Any]:
    closed_summary_path = job_root / "closed_loop" / "summary.json"
    closed_trace_path = job_root / "closed_loop" / "edit_utility.jsonl"
    writeback_summary_path = job_root / "writeback" / "summary.json"
    writeback_rows_path = job_root / "writeback" / "writeback.jsonl"
    for path in (
        closed_summary_path,
        closed_trace_path,
        writeback_summary_path,
        writeback_rows_path,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing job artifact: {path}")

    closed_summary = _json(closed_summary_path)
    writeback_summary = _json(writeback_summary_path)
    if (
        closed_summary.get("schema_version")
        != "active-catalog-closed-loop-baselines-v1"
        or closed_summary.get("split") != "val"
        or closed_summary.get("sample_count") != expected_rows
        or closed_summary["protocol"].get("test_assets_read") is not False
    ):
        raise ValueError("invalid closed-loop summary protocol")
    protocol = writeback_summary.get("protocol", {})
    corruption = protocol.get("prior_input_corruption", {})
    expected_corruption = {
        "translation_pixels": translation_pixels,
        "morphology": morphology,
        "morphology_pixels": morphology_pixels,
        "corruption_seed": corruption_seed,
        "scope": "model_input_only",
    }
    if (
        writeback_summary.get("sample_count") != expected_rows
        or protocol.get("name") != protocol_name
        or protocol.get("test_assets_read") is not False
        or corruption != expected_corruption
        or sum(item["sample_count"] for item in writeback_summary.get("budgets", []))
        != expected_rows
    ):
        raise ValueError("invalid writeback summary protocol")

    closed_ids, closed_count = _closed_identities(closed_trace_path)
    writeback_ids, writeback_count = _writeback_identities(writeback_rows_path)
    if closed_count != expected_rows or writeback_count != expected_rows:
        raise ValueError("job row count mismatch")
    if closed_ids != writeback_ids:
        raise ValueError("closed-loop and writeback identities differ")

    return {
        "schema_version": "sn7-controller-writeback-job-audit-v1",
        "passed": True,
        "job_root": str(job_root.resolve()),
        "row_count": expected_rows,
        "identity_count": len(closed_ids),
        "protocol_name": protocol_name,
        "prior_input_corruption": expected_corruption,
        "sha256": {
            "closed_loop_summary": _sha256(closed_summary_path),
            "closed_loop_trace": _sha256(closed_trace_path),
            "writeback_summary": _sha256(writeback_summary_path),
            "writeback_rows": _sha256(writeback_rows_path),
        },
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-rows", type=int, default=6369)
    parser.add_argument("--protocol-name", required=True)
    parser.add_argument("--translation-pixels", type=int, default=0)
    parser.add_argument("--morphology", choices=("none", "erode", "dilate"), default="none")
    parser.add_argument("--morphology-pixels", type=int, default=0)
    parser.add_argument("--corruption-seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(
        args.job_root,
        expected_rows=args.expected_rows,
        protocol_name=args.protocol_name,
        translation_pixels=args.translation_pixels,
        morphology=args.morphology,
        morphology_pixels=args.morphology_pixels,
        corruption_seed=args.corruption_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
