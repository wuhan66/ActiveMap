#!/usr/bin/env python3
"""Re-hash and audit an SN7 paper evidence manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_tree(path: Path) -> tuple[int, str]:
    images = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    digest = hashlib.sha256()
    for image in images:
        digest.update(str(image.relative_to(path)).encode("utf-8"))
        digest.update(_sha256(image).encode("ascii"))
    return len(images), digest.hexdigest()


def audit(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in {
        "sn7-paper-evidence-manifest-v1",
        "sn7-paper-evidence-manifest-v2",
    }:
        raise ValueError("unexpected evidence manifest schema")
    if (
        manifest.get("status") != "qc_approved"
        or manifest.get("split") != "val"
        or manifest.get("test_assets_read") is not False
    ):
        raise ValueError("evidence manifest is not approved validation-only evidence")

    rows = []
    for label, artifact in manifest.get("artifacts", {}).items():
        path = Path(artifact["path"])
        kind = artifact["kind"]
        exists = path.is_dir() if kind == "image_dir" else path.is_file()
        actual_hash = None
        actual_count = None
        if exists:
            if kind == "image_dir":
                actual_count, actual_hash = _image_tree(path)
            else:
                actual_hash = _sha256(path)
                if kind == "jsonl":
                    with path.open(encoding="utf-8") as handle:
                        actual_count = sum(1 for line in handle if line.strip())
        expected_count = artifact.get(
            "image_count", artifact.get("record_count")
        )
        hash_matches = actual_hash == artifact.get("sha256")
        count_matches = expected_count is None or actual_count == expected_count
        rows.append(
            {
                "id": label,
                "kind": kind,
                "path": str(path),
                "exists": exists,
                "hash_matches": hash_matches,
                "count_matches": count_matches,
                "expected_sha256": artifact.get("sha256"),
                "actual_sha256": actual_hash,
                "expected_count": expected_count,
                "actual_count": actual_count,
            }
        )
    ready = (
        len(rows) == int(manifest.get("artifact_count", -1))
        and bool(rows)
        and all(
            row["exists"] and row["hash_matches"] and row["count_matches"]
            for row in rows
        )
    )
    return {
        "schema_version": "sn7-paper-evidence-audit-v1",
        "ready": ready,
        "artifact_count": len(rows),
        "manifest_sha256": _sha256(manifest_path),
        "test_assets_read": False,
        "artifacts": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.manifest)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
