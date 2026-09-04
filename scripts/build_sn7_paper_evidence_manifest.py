#!/usr/bin/env python3
"""Validate and fingerprint the validation-only SN7 paper evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("value must be LABEL=/path/to/artifact")
    return label, Path(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contains_test_read(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("test_assets_read") is True:
            return True
        return any(_contains_test_read(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_test_read(item) for item in value)
    return False


def _read_json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if _contains_test_read(value):
        raise ValueError(f"test asset access detected in {path}")
    return value


def _audit_jsonl(path: Path, expected_records: int) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty JSONL artifact: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_records:
        raise ValueError(
            f"{path} has {len(rows)} records, expected {expected_records}"
        )
    for index, row in enumerate(rows, start=1):
        if row.get("split") not in (None, "val"):
            raise ValueError(f"{path}:{index} is not validation-only")
        if row.get("test_assets_read") is not False:
            raise ValueError(f"{path}:{index} lacks test_assets_read=false")
    return {
        "record_count": len(rows),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def build_manifest(
    json_paths: dict[str, Path],
    jsonl_paths: dict[str, Path],
    image_dirs: dict[str, Path],
    *,
    expected_records: int,
    minimum_images: int,
    file_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    file_paths = file_paths or {}
    labels = [*json_paths, *jsonl_paths, *image_dirs, *file_paths]
    if len(labels) != len(set(labels)):
        raise ValueError("artifact labels must be unique")

    artifacts: dict[str, Any] = {}
    for label, path in json_paths.items():
        value = _read_json(path)
        artifacts[label] = {
            "kind": "json",
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "schema_version": (
                value.get("schema_version") if isinstance(value, dict) else None
            ),
        }
    for label, path in jsonl_paths.items():
        artifacts[label] = {
            "kind": "jsonl",
            "path": str(path.resolve()),
            **_audit_jsonl(path, expected_records),
        }
    for label, path in image_dirs.items():
        if not path.is_dir():
            raise ValueError(f"missing image directory: {path}")
        images = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if len(images) < minimum_images:
            raise ValueError(
                f"{path} has {len(images)} images, expected at least {minimum_images}"
            )
        digest = hashlib.sha256()
        for image in images:
            digest.update(str(image.relative_to(path)).encode("utf-8"))
            digest.update(_sha256(image).encode("ascii"))
        artifacts[label] = {
            "kind": "image_dir",
            "path": str(path.resolve()),
            "image_count": len(images),
            "sha256": digest.hexdigest(),
        }
    for label, path in file_paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty file artifact: {path}")
        artifacts[label] = {
            "kind": "file",
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "suffix": path.suffix.lower(),
        }
    return {
        "schema_version": (
            "sn7-paper-evidence-manifest-v2"
            if file_paths
            else "sn7-paper-evidence-manifest-v1"
        ),
        "status": "qc_approved",
        "split": "val",
        "expected_records_per_jsonl": expected_records,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "test_assets_read": False,
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# SN7 Paper Evidence Manifest",
        "",
        f"Status: **{manifest['status']}**",
        f"Split: `{manifest['split']}`",
        f"Artifacts: {manifest['artifact_count']}",
        "",
        "| Label | Kind | Records/images | SHA256 |",
        "| --- | --- | ---: | --- |",
    ]
    for label, row in manifest["artifacts"].items():
        count = row.get("record_count", row.get("image_count", "-"))
        lines.append(
            f"| {label} | {row['kind']} | {count} | `{row['sha256'][:12]}` |"
        )
    lines.extend(["", "Validation-only checks passed; test assets were not read.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--json", action="append", type=_parse_labeled_path, default=[])
    parser.add_argument("--jsonl", action="append", type=_parse_labeled_path, default=[])
    parser.add_argument(
        "--image-dir", action="append", type=_parse_labeled_path, default=[]
    )
    parser.add_argument("--file", action="append", type=_parse_labeled_path, default=[])
    parser.add_argument("--expected-records", type=int, default=512)
    parser.add_argument("--minimum-images", type=int, default=1)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest = build_manifest(
        dict(args.json),
        dict(args.jsonl),
        dict(args.image_dir),
        expected_records=args.expected_records,
        minimum_images=args.minimum_images,
        file_paths=dict(args.file),
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "manifest.md").write_text(
        render_markdown(manifest), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
