#!/usr/bin/env python3
"""Audit whether active-catalog rollout traces use identical evaluation support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.compare_active_catalog_closed_loop import load_rows


def audit_support(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = load_rows(reference_path)
    candidate = load_rows(candidate_path)
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    common_keys = reference_keys & candidate_keys
    metadata_mismatches = []
    for key in sorted(common_keys):
        reference_row = reference[key]
        candidate_row = candidate[key]
        fields = {
            name: {
                "reference": reference_row.get(name),
                "candidate": candidate_row.get(name),
            }
            for name in ("aoi_id", "target_edit", "budget")
            if reference_row.get(name) != candidate_row.get(name)
        }
        if fields:
            metadata_mismatches.append({"key": list(key), "fields": fields})
    return {
        "schema_version": "active-catalog-trace-support-audit-v1",
        "identical_support": reference_keys == candidate_keys,
        "identical_metadata": not metadata_mismatches,
        "reference_records": len(reference_keys),
        "candidate_records": len(candidate_keys),
        "common_records": len(common_keys),
        "reference_only_count": len(reference_keys - candidate_keys),
        "candidate_only_count": len(candidate_keys - reference_keys),
        "reference_only": [
            list(key) for key in sorted(reference_keys - candidate_keys)
        ],
        "candidate_only": [
            list(key) for key in sorted(candidate_keys - reference_keys)
        ],
        "metadata_mismatches": metadata_mismatches,
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_support(args.reference, args.candidate)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
