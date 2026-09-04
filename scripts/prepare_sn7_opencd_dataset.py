#!/usr/bin/env python3
"""Export the frozen SN7 image-map manifest to Open-CD's paired PNG layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

ALLOWED_SPLITS = {"train", "val"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("empty manifest")
    unsupported = sorted({str(row["split"]) for row in rows} - ALLOWED_SPLITS)
    if unsupported:
        raise ValueError(f"manifest contains forbidden splits: {unsupported}")
    identities = [str(row["sample_id"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate sample_id in manifest")
    return rows


def _rgb(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 3:
        raise ValueError(f"expected RGB array, got {value.shape}")
    if value.shape[0] in {1, 3, 4}:
        value = np.moveaxis(value[:3], 0, -1)
    elif value.shape[-1] in {1, 3, 4}:
        value = value[..., :3]
    else:
        raise ValueError(f"cannot infer channels from {value.shape}")
    if value.shape[-1] == 1:
        value = np.repeat(value, 3, axis=-1)
    value = value.astype(np.float32)
    if value.max(initial=0.0) <= 1.0:
        value *= 255.0
    return np.clip(np.rint(value), 0, 255).astype(np.uint8)


def _mask(path: Path) -> np.ndarray:
    value = np.asarray(np.load(path), dtype=np.float32).squeeze()
    if value.ndim != 2:
        raise ValueError(f"expected mask at {path}, got {value.shape}")
    return value >= 0.5


def _stem(index: int, sample_id: str) -> str:
    suffix = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]
    return f"{index:06d}_{suffix}"


def _write_png(path: Path, value: np.ndarray, mode: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(value, mode=mode).save(temporary, format="PNG")
    os.replace(temporary, path)


def export_dataset(manifest: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(output.name + ".partial")
    staging.mkdir(parents=True, exist_ok=True)
    rows = _read_manifest(manifest)
    root = manifest.parent
    for split in sorted(ALLOWED_SPLITS):
        for directory in ("A", "B", "label"):
            (staging / split / directory).mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for index, row in enumerate(tqdm(rows, desc="export Open-CD SN7", unit="sample")):
        split = str(row["split"])
        sample_id = str(row["sample_id"])
        stem = _stem(index, sample_id)
        prior_path = _resolve(root, row["prior_mask_path"])
        image_path = _resolve(root, row["image_path"])
        target_path = _resolve(root, row["target_mask_path"])
        valid_path = _resolve(root, row.get("valid_mask_path"))
        assert prior_path is not None and image_path is not None
        assert target_path is not None
        destinations = {
            "A": staging / split / "A" / f"{stem}.png",
            "B": staging / split / "B" / f"{stem}.png",
            "label": staging / split / "label" / f"{stem}.png",
        }
        if not all(path.exists() for path in destinations.values()):
            prior = _mask(prior_path)
            target = _mask(target_path)
            valid = (
                _mask(valid_path)
                if valid_path is not None
                else np.ones_like(prior, dtype=bool)
            )
            image = _rgb(np.load(image_path))
            if prior.shape != target.shape or valid.shape != prior.shape:
                raise ValueError(f"mask shape mismatch for {sample_id}")
            if image.shape[:2] != prior.shape:
                raise ValueError(f"image-mask shape mismatch for {sample_id}")
            prior_rgb = np.repeat(prior[..., None], 3, axis=-1).astype(np.uint8) * 255
            label = np.logical_xor(prior, target).astype(np.uint8)
            label[~valid] = 255
            _write_png(destinations["A"], prior_rgb, "RGB")
            _write_png(destinations["B"], image, "RGB")
            _write_png(destinations["label"], label, "L")
        records.append(
            {
                "sample_id": sample_id,
                "aoi_id": row["aoi_id"],
                "split": split,
                "edit_type": row["edit_type"],
                "stem": stem,
            }
        )
        counts[f"{split}:{row['edit_type']}"] += 1

    identity_path = staging / "samples.jsonl"
    identity_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "sn7-opencd-export-v1",
        "source_manifest": str(manifest),
        "source_manifest_sha256": _sha256(manifest),
        "sample_count": len(records),
        "counts_by_split_edit": dict(sorted(counts.items())),
        "layout": "split/{A=prior_rgb,B=current_rgb,label=prior_xor_target}",
        "label_values": [0, 1, 255],
        "ignore_index": 255,
        "test_assets_read": False,
        "identity_sha256": _sha256(identity_path),
    }
    (staging / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(staging, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = export_dataset(args.manifest, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
