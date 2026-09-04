#!/usr/bin/env python3
"""Authorize a validation job against a merged dual-server progress ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.snapshot_cluster_progress import code_fingerprint  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_joint_progress(
    ledger_path: Path,
    project_root: Path,
    *,
    cluster_id: str,
    expected_role: str,
) -> dict[str, Any]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("schema_version") != "activemap-unified-progress-v1":
        raise PermissionError("invalid unified progress ledger schema")
    if not ledger.get("joint_validation_allowed"):
        raise PermissionError("unified progress ledger does not allow joint validation")
    if ledger.get("violations"):
        raise PermissionError("unified progress ledger contains protocol violations")
    clusters = ledger.get("clusters", {})
    cluster = clusters.get(cluster_id)
    if not isinstance(cluster, dict):
        raise PermissionError(f"cluster {cluster_id!r} is absent from unified progress")
    if cluster.get("role") != expected_role:
        raise PermissionError("cluster role differs from the requested validation role")
    current_hash, current_count = code_fingerprint(project_root)
    if current_hash != cluster.get("code_fingerprint"):
        raise PermissionError("current code differs from the unified progress snapshot")
    if current_count != int(cluster.get("code_file_count", -1)):
        raise PermissionError("current code file count differs from the progress snapshot")
    if expected_role == "validation_debug":
        if cluster.get("test_access_allowed"):
            raise PermissionError("validation-debug cluster unexpectedly allows test access")
        if cluster.get("frozen_test_ledgers"):
            raise PermissionError("validation-debug cluster contains frozen-test provenance")
    return {
        "ledger": str(ledger_path.resolve()),
        "ledger_sha256": _sha256(ledger_path),
        "cluster_id": cluster_id,
        "role": expected_role,
        "code_fingerprint": current_hash,
        "registry_sha256": cluster["registry_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument(
        "--expected-role",
        required=True,
        choices=("authoritative", "validation_debug"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            assert_joint_progress(
                args.ledger,
                args.project_root,
                cluster_id=args.cluster_id,
                expected_role=args.expected_role,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
