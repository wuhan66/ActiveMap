#!/usr/bin/env python3
"""Run one frozen paper-test command and persist an immutable access ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.audit_paper_experiment_registry import audit_registry


def audit_frozen_registry(registry: Path, storage_root: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(registry.read_text(encoding="utf-8"))
    if payload.get("schema_version") == "sn7-step0-frozen-registry-v2":
        from scripts.audit_sn7_step0_frozen_registry_v2 import audit

        return audit(registry, storage_root)
    return audit_registry(registry, storage_root)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_ledger(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def run_frozen_test(
    registry: Path,
    storage_root: Path,
    ledger: Path,
    command: list[str],
    *,
    purpose: str,
) -> int:
    if os.environ.get("ACTIVEMAP_DISABLE_FROZEN_TEST") == "1":
        raise PermissionError("frozen test execution is disabled on this cluster role")
    if ledger.exists():
        raise FileExistsError(f"frozen test ledger already exists: {ledger}")
    if not command:
        raise ValueError("frozen test command must not be empty")
    report = audit_frozen_registry(registry, storage_root)
    if not report["ready_for_frozen_test"]:
        blockers = list(report.get("blockers", []))
        blockers.extend(
            item["id"]
            for item in report["artifacts"]
            if not item.get("ready", item.get("hash_matches", False))
        )
        raise ValueError(f"paper test gate is closed; blocking artifacts: {blockers}")

    artifact_hashes = {
        item["id"]: _sha256(Path(item["path"])) for item in report["artifacts"]
    }
    started = datetime.now(timezone.utc).isoformat()
    authorization_token = secrets.token_urlsafe(32)
    payload: dict[str, Any] = {
        "schema_version": "activemap-frozen-test-access-v1",
        "status": "started",
        "started_at_utc": started,
        "purpose": purpose,
        "registry": str(registry.resolve()),
        "registry_sha256": _sha256(registry),
        "storage_root": str(storage_root.resolve()),
        "artifact_sha256": artifact_hashes,
        "command": command,
        "authorization_sha256": hashlib.sha256(
            authorization_token.encode()
        ).hexdigest(),
        "preflight": report,
    }
    _write_ledger(ledger, payload)

    environment = os.environ.copy()
    environment["ACTIVEMAP_FROZEN_TEST"] = "1"
    environment["ACTIVEMAP_FROZEN_TEST_LEDGER"] = str(ledger.resolve())
    environment["ACTIVEMAP_FROZEN_TEST_TOKEN"] = authorization_token
    try:
        result = subprocess.run(command, check=False, env=environment)
    except BaseException as exc:
        payload["status"] = "launcher_error"
        payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        payload["error"] = f"{type(exc).__name__}: {exc}"
        _write_ledger(ledger, payload)
        raise
    payload["status"] = "complete" if result.returncode == 0 else "command_failed"
    payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["returncode"] = result.returncode
    _write_ledger(ledger, payload)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("storage_root", type=Path)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--confirm-frozen", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.confirm_frozen:
        raise SystemExit("Refusing test access without explicit --confirm-frozen")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    raise SystemExit(
        run_frozen_test(
            args.registry,
            args.storage_root,
            args.ledger,
            command,
            purpose=args.purpose,
        )
    )


if __name__ == "__main__":
    main()
