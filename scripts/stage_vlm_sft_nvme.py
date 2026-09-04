#!/usr/bin/env python3
"""Stage validation-safe visual SFT JSONL and referenced images on local NVMe."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_reference(row: dict[str, Any], source_jsonl: Path) -> Path:
    if row.get("split") not in {"train", "val"}:
        raise ValueError("NVMe staging accepts only train and validation rows")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("visual SFT row lacks messages")
    content = messages[1].get("content")
    if not isinstance(content, list):
        raise ValueError("visual SFT user content must be a list")
    blocks = [block for block in content if block.get("type") == "image"]
    if len(blocks) != 1 or not isinstance(blocks[0].get("image"), str):
        raise ValueError("visual SFT row must contain exactly one image path")
    path = Path(blocks[0]["image"])
    return path if path.is_absolute() else (source_jsonl.parent / path).resolve()


def _read_rows(path: Path, expected_split: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        if row.get("split") != expected_split:
            raise ValueError(f"unexpected split in {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"empty visual SFT split: {path}")
    return rows


def _destination(source: Path, image_root: Path) -> Path:
    identity = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:24]
    suffix = source.suffix.lower() or ".image"
    return image_root / f"{identity}{suffix}"


def _copy_image(pair: tuple[Path, Path]) -> dict[str, Any]:
    source, destination = pair
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
    shutil.copystat(source, destination)
    if source.stat().st_size != destination.stat().st_size:
        raise OSError(f"staged image size mismatch: {source}")
    return {
        "source": str(source.resolve()),
        "staged": str(destination.resolve()),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def stage_dataset(
    source_root: Path,
    output_root: Path,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if output_root.exists():
        raise FileExistsError(f"refusing existing stage root: {output_root}")
    partial = output_root.with_name(output_root.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"refusing partial stage root: {partial}")
    source_paths = {
        split: source_root / f"{split}.jsonl" for split in ("train", "val")
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    rows_by_split = {
        split: _read_rows(path, split) for split, path in source_paths.items()
    }
    sources = sorted(
        {
            image_reference(row, source_paths[split]).resolve()
            for split, rows in rows_by_split.items()
            for row in rows
        }
    )
    image_root = partial / "images"
    copy_mapping = {source: _destination(source, image_root) for source in sources}
    final_image_root = output_root / "images"
    final_mapping = {
        source: _destination(source, final_image_root) for source in sources
    }
    if len(set(copy_mapping.values())) != len(copy_mapping):
        raise RuntimeError("staged image-name collision")

    partial.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        catalog = list(executor.map(_copy_image, copy_mapping.items()))
    for row in catalog:
        source = Path(row["source"])
        row["staged"] = str(final_mapping[source])

    split_summary = {}
    for split, rows in rows_by_split.items():
        source_path = source_paths[split]
        output = partial / f"{split}.jsonl"
        with output.open("x", encoding="utf-8") as handle:
            for row in rows:
                source = image_reference(row, source_path).resolve()
                for block in row["messages"][1]["content"]:
                    if block.get("type") == "image":
                        block["image"] = str(final_mapping[source].resolve())
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        split_summary[split] = {
            "records": len(rows),
            "source": str(source_path.resolve()),
            "source_sha256": sha256_file(source_path),
            "staged_sha256": sha256_file(output),
        }

    catalog_path = partial / "image_catalog.jsonl"
    catalog_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in catalog),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "visual-sft-nvme-stage-v1",
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "splits": split_summary,
        "unique_images": len(catalog),
        "image_bytes": sum(int(row["bytes"]) for row in catalog),
        "image_catalog_sha256": sha256_file(catalog_path),
        "test_assets_read": False,
    }
    (partial / "stage_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    partial.replace(output_root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            stage_dataset(args.source_root, args.output_root, workers=args.workers),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
