#!/usr/bin/env python3
"""Export the cap20 recovery without modifying the frozen v2 reporter.

The recovery changes only the pre-existing episode-support cap.  This adapter
validates that boundary, then delegates every metric and table computation to
the hash-registered frozen reporter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import export_sn7_step0_frozen_test_tables as frozen_reporter


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_recovery_ledger(
    ledger: dict[str, Any], ledger_path: Path, registry_path: Path
) -> None:
    if ledger.get("schema_version") != "activemap-frozen-test-access-v1":
        raise ValueError("unexpected frozen-test ledger schema")
    if ledger.get("status") != "complete" or int(ledger.get("returncode", -1)) != 0:
        raise ValueError("frozen-test ledger is not complete and successful")
    if ledger.get("registry_sha256") != _sha256(registry_path):
        raise ValueError("ledger registry hash differs from the supplied registry")

    command = " ".join(map(str, ledger.get("command", [])))
    launchers = (
        "run_sn7_step0_frozen_test_v3_cap20.sh",
        "resume_sn7_step0_frozen_test_v3_cap20_after_split_fix.sh",
    )
    if not any(launcher in command for launcher in launchers):
        raise ValueError("ledger command is not the cap20 recovery launcher")

    registry_text = registry_path.read_text(encoding="utf-8")
    required_markers = (
        "schema_version: sn7-step0-frozen-registry-v2",
        "status: validation_promoted_test_ready_protocol_recovery",
        "max_per_operation: 20",
        "recovery_basis:",
    )
    if any(marker not in registry_text for marker in required_markers):
        raise ValueError("registry is not the audited cap20 recovery contract")
    if not ledger_path.is_file():
        raise ValueError("frozen-test ledger does not exist")


def export(
    registry_path: Path,
    ledger_path: Path,
    run_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    original_validator = frozen_reporter._validate_ledger
    frozen_reporter._validate_ledger = _validate_recovery_ledger
    try:
        manifest = frozen_reporter.export(
            registry_path, ledger_path, run_root, output_dir
        )
    finally:
        frozen_reporter._validate_ledger = original_validator

    manifest["protocol_recovery"] = "v3_cap20_launcher_only"
    manifest["metric_reporter_sha256"] = _sha256(
        Path(frozen_reporter.__file__).resolve()
    )
    manifest["compatibility_adapter_sha256"] = _sha256(Path(__file__).resolve())
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            export(args.registry, args.ledger, args.run_root, args.output_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
