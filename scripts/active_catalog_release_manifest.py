#!/usr/bin/env python3
"""Build and verify a local-mother release manifest for cluster experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACKED_ROOTS = ("src", "scripts", "configs", "tests", "docs")
TRACKED_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".json", ".md"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_files(project_root: Path) -> list[Path]:
    files = sorted(
        path
        for root in TRACKED_ROOTS
        for path in (project_root / root).rglob("*")
        if path.is_file()
        and path.suffix.lower() in TRACKED_SUFFIXES
        and "__pycache__" not in path.parts
    )
    if not files:
        raise ValueError(f"no tracked files under {project_root}")
    return files


def fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        relative = str(row["path"]).encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(str(row["sha256"])))
    return digest.hexdigest()


def parse_declarations(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        name, separator, digest = value.partition("=")
        if (
            not separator
            or not name
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
            or name in result
        ):
            raise ValueError("declared hashes must use unique NAME=64_HEX_SHA256 values")
        result[name] = digest.lower()
    return result


def build_manifest(
    project_root: Path,
    *,
    declared_artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    rows = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in tracked_files(project_root)
    ]
    code_fingerprint = fingerprint(rows)
    return {
        "schema_version": "active-catalog-release-manifest-v1",
        "release_id": f"active-catalog-{code_fingerprint[:16]}",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_fingerprint": code_fingerprint,
        "file_count": len(rows),
        "tracked_roots": list(TRACKED_ROOTS),
        "files": rows,
        "declared_artifact_sha256": dict(sorted((declared_artifacts or {}).items())),
        "resource_contract": {
            "maximum_gpus_per_server": 4,
            "default_gpus_per_process": 1,
            "refuse_occupied_gpu": True,
            "minimum_free_disk_gib": 80,
        },
        "test_assets_read": False,
    }


def verify_manifest(project_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != "active-catalog-release-manifest-v1":
        raise ValueError("unsupported release manifest schema")
    project_root = project_root.resolve()
    expected_rows = manifest.get("files")
    if not isinstance(expected_rows, list) or not expected_rows:
        raise ValueError("release manifest contains no files")
    missing = []
    size_mismatches = []
    hash_mismatches = []
    actual_rows = []
    for expected in expected_rows:
        relative = str(expected["path"])
        path = project_root / Path(relative)
        if not path.is_file():
            missing.append(relative)
            continue
        actual = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        actual_rows.append(actual)
        if actual["bytes"] != int(expected["bytes"]):
            size_mismatches.append(relative)
        if actual["sha256"] != str(expected["sha256"]):
            hash_mismatches.append(relative)
    actual_fingerprint = fingerprint(actual_rows) if not missing else None
    expected_fingerprint = str(manifest["code_fingerprint"])
    passed = (
        not missing
        and not size_mismatches
        and not hash_mismatches
        and actual_fingerprint == expected_fingerprint
    )
    return {
        "schema_version": "active-catalog-release-verification-v1",
        "release_id": manifest["release_id"],
        "passed": passed,
        "expected_file_count": len(expected_rows),
        "verified_file_count": len(actual_rows),
        "expected_code_fingerprint": expected_fingerprint,
        "actual_code_fingerprint": actual_fingerprint,
        "missing": missing,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
        "test_assets_read": False,
    }


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("project_root", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("--declare-sha", action="append", default=[])
    verify = subparsers.add_parser("verify")
    verify.add_argument("project_root", type=Path)
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        payload = build_manifest(
            args.project_root,
            declared_artifacts=parse_declarations(args.declare_sha),
        )
        atomic_write(args.output, payload)
    else:
        payload = verify_manifest(
            args.project_root,
            json.loads(args.manifest.read_text(encoding="utf-8")),
        )
        if args.output is not None:
            if args.output.exists():
                raise FileExistsError(f"refusing to overwrite {args.output}")
            atomic_write(args.output, payload)
    print(json.dumps(payload, indent=2))
    if args.command == "verify" and not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
