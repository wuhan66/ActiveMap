#!/usr/bin/env python3
"""Refuse reported training until dataset audits and manual QC are approved."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.audit_paper_experiment_registry import audit_registry

DATASET_GATES = {"sn7_audit", "muno21_audit", "inria_audit", "manual_qc_approval"}


def training_readiness(registry_path: Path, storage_root: Path) -> dict[str, object]:
    report = audit_registry(registry_path, storage_root)
    artifacts = {str(item["id"]): item for item in report["artifacts"]}
    missing_specs = sorted(DATASET_GATES - set(artifacts))
    failed = sorted(
        artifact_id
        for artifact_id in DATASET_GATES & set(artifacts)
        if not bool(artifacts[artifact_id]["ready"])
    )
    return {
        "schema_version": "activemap-training-readiness-v1",
        "ready": bool(report["protocol_valid"]) and not missing_specs and not failed,
        "protocol_valid": bool(report["protocol_valid"]),
        "required_gates": sorted(DATASET_GATES),
        "missing_gate_specs": missing_specs,
        "failed_gates": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("storage_root", type=Path)
    args = parser.parse_args()
    result = training_readiness(args.registry, args.storage_root)
    print(json.dumps(result, indent=2))
    if not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
