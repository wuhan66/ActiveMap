#!/usr/bin/env python3
"""Verify hash-bound artifacts in the SN7 Step-0 frozen registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifacts(registry: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield from registry.get("shared_artifacts", [])
    for seed, components in registry.get("seed_artifacts", {}).items():
        for name, artifact in components.items():
            yield {"id": f"seed{seed}_{name}", **artifact}
    yield from registry.get("validation_evidence", [])
    yield from registry.get("test_manifest_artifacts", [])
    yield from registry.get("code_artifacts", [])


def audit(registry_path: Path, storage_root: Path) -> dict[str, Any]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != "sn7-step0-frozen-registry-v2":
        raise ValueError("unexpected registry schema")
    results = []
    project_root = registry_path.resolve().parents[2]
    for artifact in _artifacts(registry):
        rendered = (
            str(artifact["path"])
            .replace("${STORAGE_ROOT}", str(storage_root))
            .replace("${PROJECT_ROOT}", str(project_root))
        )
        path = Path(rendered)
        exists = path.is_file() and path.stat().st_size > 0
        actual = _sha256(path) if exists else None
        expected = str(artifact["sha256"])
        results.append(
            {
                "id": artifact.get("id"),
                "path": str(path),
                "exists": exists,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "hash_matches": actual == expected,
            }
        )
    artifacts_ready = all(item["hash_matches"] for item in results)
    test_gate = registry.get("test_gate", {})
    return {
        "schema_version": "sn7-step0-frozen-registry-audit-v2",
        "artifacts_ready": artifacts_ready,
        "artifact_count": len(results),
        "test_gate_ready": bool(test_gate.get("ready")),
        "ready_for_frozen_test": artifacts_ready and bool(test_gate.get("ready")),
        "blockers": list(test_gate.get("blockers", [])),
        "artifacts": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("storage_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.registry, args.storage_root)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
