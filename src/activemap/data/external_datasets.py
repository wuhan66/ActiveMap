"""License-aware registry and scaffold for external research datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "activemap-external-datasets-v1"
ALLOWED_ACCESS = {"public_archive", "project_download", "terms_required"}
ALLOWED_STATUS = {"source_audit", "download_ready", "prepared", "qc_approved"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_external_dataset_registry(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("external dataset registry must be a mapping")
    return payload


def audit_external_dataset_registry(path: Path) -> dict[str, Any]:
    payload = load_external_dataset_registry(path)
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    protocol = payload.get("protocol", {})
    for field in (
        "immutable_raw_assets",
        "split_manifests_required",
        "license_acceptance_out_of_band",
        "test_labels_locked",
    ):
        if protocol.get(field) is not True:
            errors.append(f"protocol.{field} must be true")

    datasets = payload.get("datasets", [])
    ids = [str(row.get("id", "")) for row in datasets]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate dataset ids: {duplicates}")
    for row in datasets:
        dataset_id = str(row.get("id", "<missing>"))
        missing = [
            field
            for field in (
                "display_name",
                "task_family",
                "modalities",
                "adapter",
                "access",
                "source_url",
                "license_status",
                "status",
            )
            if not row.get(field)
        ]
        if missing:
            errors.append(f"{dataset_id}: missing fields {missing}")
        if row.get("access") not in ALLOWED_ACCESS:
            errors.append(f"{dataset_id}: invalid access={row.get('access')!r}")
        if row.get("status") not in ALLOWED_STATUS:
            errors.append(f"{dataset_id}: invalid status={row.get('status')!r}")
        if not isinstance(row.get("modalities"), list) or not row.get("modalities"):
            errors.append(f"{dataset_id}: modalities must be a non-empty list")
        rows.append(
            {
                "id": dataset_id,
                "display_name": row.get("display_name"),
                "task_family": row.get("task_family"),
                "access": row.get("access"),
                "license_status": row.get("license_status"),
                "status": row.get("status"),
                "adapter": row.get("adapter"),
                "source_url": row.get("source_url"),
            }
        )
    return {
        "schema_version": "activemap-external-dataset-audit-v1",
        "registry": str(path.resolve()),
        "registry_sha256": _sha256(path),
        "valid": not errors,
        "errors": errors,
        "summary": {
            "datasets": len(rows),
            "prepared": sum(row["status"] in {"prepared", "qc_approved"} for row in rows),
            "terms_required": sum(row["access"] == "terms_required" for row in rows),
        },
        "datasets": rows,
    }


def build_external_dataset_scaffold(
    registry_path: Path,
    dataset_id: str,
    output_root: Path,
) -> dict[str, Any]:
    report = audit_external_dataset_registry(registry_path)
    if not report["valid"]:
        raise ValueError(f"invalid external dataset registry: {report['errors']}")
    payload = load_external_dataset_registry(registry_path)
    dataset = next(
        (row for row in payload["datasets"] if row["id"] == dataset_id),
        None,
    )
    if dataset is None:
        raise KeyError(f"unknown external dataset: {dataset_id}")
    dataset_root = output_root / dataset_id
    if dataset_root.exists():
        raise FileExistsError(f"refusing to overwrite dataset scaffold: {dataset_root}")
    for name in ("raw", "processed", "manifests", "splits", "audit"):
        (dataset_root / name).mkdir(parents=True, exist_ok=False)
    plan = {
        "schema_version": "activemap-external-dataset-plan-v1",
        "dataset": dataset,
        "registry": {
            "path": str(registry_path.resolve()),
            "sha256": report["registry_sha256"],
        },
        "paths": {
            name: str((dataset_root / name).resolve())
            for name in ("raw", "processed", "manifests", "splits", "audit")
        },
        "automatic_download_permitted": dataset["access"] in {
            "public_archive",
            "project_download",
        },
        "license_acceptance_required": dataset["access"] == "terms_required",
        "test_labels_locked": True,
    }
    (dataset_root / "dataset_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n",
        encoding="utf-8",
    )
    return plan
