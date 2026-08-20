#!/usr/bin/env python3
"""Create a provenance-rich approval only after train/validation QC review."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

QC_DIRECTORY_NAME = "qc_train_val_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def approve_qc(
    storage_root: Path,
    output: Path,
    *,
    reviewer: str,
    notes: str,
    minimum_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    minimum_counts = minimum_counts or {"sn7": 128, "muno21": 96, "inria": 96}
    specs = {
        "sn7": (
            storage_root / "processed/sn7_v1/updater_v4_cap20/audit.json",
            storage_root / f"processed/sn7_v1/updater_v4_cap20/{QC_DIRECTORY_NAME}",
        ),
        "muno21": (
            storage_root / "processed/muno21_v2/updater/audit.json",
            storage_root / f"processed/muno21_v2/updater/{QC_DIRECTORY_NAME}",
        ),
        "inria": (
            storage_root / "processed/inria_v1/segmentation/audit.json",
            storage_root / f"processed/inria_v1/segmentation/{QC_DIRECTORY_NAME}",
        ),
    }
    datasets: dict[str, Any] = {}
    for name, (audit_path, qc_dir) in specs.items():
        if not audit_path.is_file() or not _load_json(audit_path).get("passed"):
            raise ValueError(f"{name}: audit is missing or failed: {audit_path}")
        index_path = qc_dir / "index.json"
        if not index_path.is_file():
            raise ValueError(f"{name}: QC index is missing: {index_path}")
        index = _load_json(index_path)
        if index.get("test_assets_rendered") is not False:
            raise ValueError(f"{name}: QC must explicitly prove test_assets_rendered=false")
        rendered_splits = set(index.get("rendered_split_counts", {}))
        if not rendered_splits or not rendered_splits <= {"train", "val"}:
            raise ValueError(f"{name}: invalid rendered QC splits: {sorted(rendered_splits)}")
        pngs = sorted(qc_dir.glob("*.png"))
        if len(pngs) < minimum_counts[name]:
            raise ValueError(
                f"{name}: only {len(pngs)} QC images; need {minimum_counts[name]}"
            )
        image_manifest = [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in pngs
        ]
        manifest_digest = hashlib.sha256(
            json.dumps(image_manifest, sort_keys=True).encode("utf-8")
        ).hexdigest()
        datasets[name] = {
            "audit_path": str(audit_path),
            "audit_sha256": _sha256(audit_path),
            "qc_index_path": str(index_path),
            "qc_index_sha256": _sha256(index_path),
            "qc_image_count": len(pngs),
            "qc_manifest_sha256": manifest_digest,
            "rendered_split_counts": index["rendered_split_counts"],
            "test_assets_rendered": False,
        }
    approval = {
        "schema_version": "activemap-dataset-qc-approval-v1",
        "approved": True,
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": reviewer,
        "notes": notes,
        "scope": "train_validation_qc_only",
        "test_assets_reviewed": False,
        "datasets": datasets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(approval, indent=2) + "\n", encoding="utf-8")
    partial.replace(output)
    return approval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("storage_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--approve", action="store_true")
    args = parser.parse_args()
    if not args.approve:
        raise SystemExit("Refusing to approve without explicit --approve")
    approval = approve_qc(
        args.storage_root,
        args.output,
        reviewer=args.reviewer,
        notes=args.notes,
    )
    print(json.dumps(approval, indent=2))


if __name__ == "__main__":
    main()
