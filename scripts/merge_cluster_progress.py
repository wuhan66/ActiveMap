#!/usr/bin/env python3
"""Merge server snapshots and enforce one authoritative frozen-test lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse(specifications: list[str]) -> dict[str, Path]:
    result = {}
    for specification in specifications:
        label, separator, raw_path = specification.partition("=")
        if not separator or not label or not raw_path or label in result:
            raise ValueError("snapshots must use unique LABEL=PATH values")
        result[label] = Path(raw_path)
    return result


def merge(snapshot_paths: dict[str, Path]) -> dict[str, Any]:
    if len(snapshot_paths) < 2:
        raise ValueError("joint progress requires at least two cluster snapshots")
    snapshots = {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in sorted(snapshot_paths.items())
    }
    if any(
        row.get("schema_version") != "activemap-cluster-progress-v1"
        for row in snapshots.values()
    ):
        raise ValueError("invalid cluster progress schema")
    cluster_ids = [str(row["cluster_id"]) for row in snapshots.values()]
    if len(set(cluster_ids)) != len(cluster_ids):
        raise ValueError("cluster_id values must be unique")
    authoritative = [
        label for label, row in snapshots.items() if row.get("role") == "authoritative"
    ]
    if len(authoritative) != 1:
        raise ValueError("joint progress requires exactly one authoritative cluster")
    violations = []
    for label, row in snapshots.items():
        if row.get("role") == "validation_debug" and row.get("frozen_test_ledgers"):
            violations.append(f"{label}: validation_debug contains a frozen-test ledger")
        if row.get("role") == "validation_debug" and row.get("test_access_allowed"):
            violations.append(f"{label}: validation_debug incorrectly allows test access")
    code_hashes = {str(row["code_fingerprint"]) for row in snapshots.values()}
    registry_hashes = {str(row["registry_sha256"]) for row in snapshots.values()}
    protocol_valid = all(bool(row.get("protocol_valid")) for row in snapshots.values())
    in_sync = len(code_hashes) == 1 and len(registry_hashes) == 1
    return {
        "schema_version": "activemap-unified-progress-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_cluster": authoritative[0],
        "cluster_count": len(snapshots),
        "code_in_sync": len(code_hashes) == 1,
        "registry_in_sync": len(registry_hashes) == 1,
        "protocol_valid_everywhere": protocol_valid,
        "test_lineage_valid": not violations,
        "joint_validation_allowed": in_sync and protocol_valid and not violations,
        "frozen_test_allowed": (
            in_sync
            and protocol_valid
            and not violations
            and bool(snapshots[authoritative[0]].get("ready_for_frozen_test"))
        ),
        "violations": violations,
        "clusters": snapshots,
        "source_snapshots": [
            {"label": label, "path": str(path.resolve()), "sha256": _sha256(path)}
            for label, path in sorted(snapshot_paths.items())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--snapshot", action="append", required=True)
    args = parser.parse_args()
    payload = merge(_parse(args.snapshot))
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["joint_validation_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
