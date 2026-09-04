#!/usr/bin/env python3
"""Independently verify a controller-intervention paper evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = {
    "summary.json",
    "claim_gate.json",
    "table.csv",
    "table.md",
    "table.tex",
    "intervention_summary.png",
    "intervention_summary.pdf",
    "intervention_summary.svg",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def audit(bundle: Path) -> dict[str, Any]:
    if not bundle.is_dir():
        raise FileNotFoundError(bundle)
    manifest_path = bundle / "manifest.json"
    manifest = _load(manifest_path)
    if manifest.get("schema_version") != "sn7-controller-intervention-manifest-v1":
        raise ValueError("unexpected manifest schema")
    if manifest.get("test_assets_read") is not False:
        raise ValueError("manifest is not validation-only")

    registered = manifest.get("files")
    if not isinstance(registered, dict):
        raise ValueError("manifest files must be an object")
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != set(registered):
        raise ValueError(
            f"manifest coverage mismatch: missing={sorted(actual - set(registered))}, "
            f"extra={sorted(set(registered) - actual)}"
        )
    if not REQUIRED_FILES <= actual:
        raise ValueError(f"required assets missing: {sorted(REQUIRED_FILES - actual)}")

    for relative, metadata in registered.items():
        path = bundle / relative
        if int(metadata["bytes"]) != path.stat().st_size:
            raise ValueError(f"size mismatch: {relative}")
        if str(metadata["sha256"]) != _sha256(path):
            raise ValueError(f"hash mismatch: {relative}")

    summary = _load(bundle / "summary.json")
    claim_gate = _load(bundle / "claim_gate.json")
    if summary.get("protocol", {}).get("test_assets_read") is not False:
        raise ValueError("summary is not validation-only")
    if claim_gate.get("test_assets_read") is not False:
        raise ValueError("claim gate is not validation-only")
    summary_conditions = [str(row["condition"]) for row in summary.get("rows", [])]
    gate_conditions = [str(row["condition"]) for row in claim_gate.get("checks", [])]
    if not summary_conditions or summary_conditions != gate_conditions:
        raise ValueError("summary and claim-gate conditions differ")

    input_count = 0
    for row in summary["rows"]:
        for metadata in row.get("inputs", {}).values():
            relative = str(metadata["bundle_path"])
            path = bundle / relative
            if not path.is_file() or relative not in registered:
                raise ValueError(f"unregistered bundled input: {relative}")
            if str(metadata["sha256"]) != _sha256(path):
                raise ValueError(f"bundled input hash mismatch: {relative}")
            input_count += 1
    if input_count != 3 * len(summary_conditions):
        raise ValueError("each condition must bundle exactly three aggregate inputs")

    return {
        "schema_version": "sn7-controller-intervention-bundle-audit-v1",
        "passed": True,
        "condition_count": len(summary_conditions),
        "input_count": input_count,
        "manifest_file_count": len(registered),
        "all_conditions_strict": bool(claim_gate.get("all_conditions_strict")),
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
