#!/usr/bin/env python3
"""Snapshot one ActiveMap server without opening datasets or test assets."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_paper_experiment_registry import audit_registry  # noqa: E402
from scripts.active_catalog_release_manifest import verify_manifest  # noqa: E402

TRACKED_ROOTS = ("src", "scripts", "configs", "tests")
TRACKED_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".json"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_fingerprint(project_root: Path) -> tuple[str, int]:
    files = sorted(
        path
        for root in TRACKED_ROOTS
        for path in (project_root / root).rglob("*")
        if path.is_file()
        and path.suffix in TRACKED_SUFFIXES
        and "__pycache__" not in path.parts
    )
    if not files:
        raise ValueError(f"no tracked code files under {project_root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(project_root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest(), len(files)


def _command(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else ""


def _gpus() -> list[dict[str, Any]]:
    output = _command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mib": int(parts[2]),
                "memory_total_mib": int(parts[3]),
                "utilization_percent": int(parts[4]),
            }
        )
    return rows


def _active_jobs(project_root: Path, storage_root: Path) -> list[dict[str, Any]]:
    output = _command(
        ["ps", "-u", getpass.getuser(), "-o", "pid=,etime=,args="]
    )
    needles = (str(project_root), str(storage_root), "activemap")
    rows = []
    for line in output.splitlines():
        if "snapshot_cluster_progress.py" in line or not any(item in line for item in needles):
            continue
        parts = line.strip().split(maxsplit=2)
        if len(parts) == 3:
            rows.append({"pid": int(parts[0]), "elapsed": parts[1], "command": parts[2]})
    return rows[:100]


def _path_size(path: Path) -> int | None:
    if path.is_file():
        return path.stat().st_size
    return None


def _frozen_ledgers(storage_root: Path) -> list[dict[str, Any]]:
    root = storage_root / "artifacts" / "frozen_test"
    rows = []
    for path in sorted(root.glob("*ledger*.json")) if root.is_dir() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = "invalid"
        else:
            status = str(payload.get("status", "unknown"))
        rows.append({"path": str(path.resolve()), "status": status, "sha256": _sha256(path)})
    return rows


def _json_artifact(path: Path, field: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "bytes": _path_size(path),
    }
    if not path.is_file():
        return result
    result["sha256"] = _sha256(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        result["json_error"] = str(error)
        return result
    if field is not None:
        result["field"] = field
        result["value"] = payload.get(field)
    result["schema_version"] = payload.get("schema_version")
    return result


def _active_catalog_runs(storage_root: Path) -> list[dict[str, Any]]:
    run_root = storage_root / "runs" / "sn7_active_catalog"
    rows = []
    if not run_root.is_dir():
        return rows
    for path in sorted(run_root.rglob("run_state.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            rows.append({"path": str(path.resolve()), "status": "invalid_json"})
            continue
        rows.append(
            {
                "path": str(path.relative_to(run_root)),
                "status": payload.get("status", "unknown"),
                "pid": payload.get("pid"),
                "physical_gpu": payload.get("physical_gpu"),
                "returncode": payload.get("returncode"),
                "peak_memory_used_mib": payload.get("peak_memory_used_mib"),
                "last_resource_sample": payload.get("last_resource_sample"),
            }
        )
    return rows[:200]


def _release_status(project_root: Path) -> dict[str, Any]:
    path = project_root / "releases" / "active_catalog_current.json"
    if not path.is_file():
        return {"path": str(path.resolve()), "exists": False, "verified": False}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        verification = verify_manifest(project_root, manifest)
    except (json.JSONDecodeError, KeyError, ValueError) as error:
        return {
            "path": str(path.resolve()),
            "exists": True,
            "verified": False,
            "error": str(error),
        }
    return {
        "path": str(path.resolve()),
        "exists": True,
        "sha256": _sha256(path),
        "release_id": manifest.get("release_id"),
        "code_fingerprint": manifest.get("code_fingerprint"),
        "verified": verification["passed"],
        "verification": verification,
        "declared_artifact_sha256": manifest.get("declared_artifact_sha256", {}),
    }


def snapshot(
    project_root: Path,
    storage_root: Path,
    *,
    cluster_id: str,
    role: str,
) -> dict[str, Any]:
    registry = project_root / "configs" / "experiments" / "paper_registry.yaml"
    code_hash, code_file_count = code_fingerprint(project_root)
    audit = audit_registry(registry, storage_root)
    return {
        "schema_version": "activemap-cluster-progress-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cluster_id": cluster_id,
        "role": role,
        "hostname": platform.node(),
        "project_root": str(project_root.resolve()),
        "storage_root": str(storage_root.resolve()),
        "code_fingerprint": code_hash,
        "code_file_count": code_file_count,
        "active_catalog_release": _release_status(project_root),
        "registry_sha256": _sha256(registry),
        "protocol_valid": bool(audit["protocol_valid"]),
        "ready_for_frozen_test": bool(audit["ready_for_frozen_test"]),
        "artifact_summary": audit["artifact_summary"],
        "experiment_summary": audit["experiment_summary"],
        "artifact_gates": [
            {
                "id": row["id"],
                "ready": row["ready"],
                "reason": row["reason"],
                "path": row["path"],
            }
            for row in audit["artifacts"]
        ],
        "frozen_test_ledgers": _frozen_ledgers(storage_root),
        "gpu": _gpus(),
        "active_jobs": _active_jobs(project_root, storage_root),
        "known_files": {
            "selector_states": _path_size(
                storage_root / "processed/muno21_v2/agent/selector_states_v1.jsonl"
            ),
            "episodes_train_val": _path_size(
                storage_root / "processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
            ),
            "manual_qc": _path_size(
                storage_root / "logs/dataset_preparation/QC_APPROVED.json"
            ),
            "sn7_active_catalog_selector_states": _path_size(
                storage_root
                / "processed/sn7_v1/agent/sequential_selector_v1/selector_states_train_val.jsonl"
            ),
            "sn7_active_catalog_sft_audit": _path_size(
                storage_root
                / "processed/sn7_v1/agent/sequential_selector_v1/full/active_catalog_sft_v4/audit.json"
            ),
        },
        "active_catalog_gates": {
            "token_audit": _json_artifact(
                storage_root
                / "runs/sn7_active_catalog/token_audit/full_qwen3vl4b.json",
                "passed",
            ),
            "three_seed_aggregate": _json_artifact(
                storage_root
                / "runs/sn7_active_catalog/qwen3vl4b_three_seed/active_catalog_aoi_bootstrap.json"
            ),
            "closed_loop_promotion": _json_artifact(
                storage_root
                / "runs/sn7_active_catalog/closed_loop_writeback_qwen/promotion_uncertainty_gate.json",
                "promote",
            ),
            "writeback_summary": _json_artifact(
                storage_root
                / "runs/sn7_active_catalog/closed_loop_writeback_qwen/evaluation/summary.json"
            ),
        },
        "active_catalog_run_states": _active_catalog_runs(storage_root),
        "test_access_allowed": role == "authoritative",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("storage_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument(
        "--role", required=True, choices=("authoritative", "validation_debug")
    )
    args = parser.parse_args()
    payload = snapshot(
        args.project_root,
        args.storage_root,
        cluster_id=args.cluster_id,
        role=args.role,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    partial.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
