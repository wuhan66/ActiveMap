#!/usr/bin/env python3
"""Materialize a path-only SN7 test manifest and audit AOI isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_aois(path: Path) -> set[str]:
    aois = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                aois.add(str(json.loads(line)["aoi_id"]))
    if not aois:
        raise ValueError(f"episode file has no AOIs: {path}")
    return aois


def audit(
    split_manifest: Path,
    train_episodes: Path,
    val_episodes: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    frame = pd.read_parquet(split_manifest)
    required = {"aoi_id", "split"}
    if missing := required - set(frame.columns):
        raise ValueError(f"split manifest lacks columns: {sorted(missing)}")
    split_aois = {
        split: set(frame.loc[frame["split"] == split, "aoi_id"].astype(str))
        for split in ("train", "val", "test")
    }
    if any(not values for values in split_aois.values()):
        raise ValueError("train, val, and test must all contain AOIs")
    if (
        split_aois["train"] & split_aois["val"]
        or split_aois["train"] & split_aois["test"]
        or split_aois["val"] & split_aois["test"]
    ):
        raise ValueError("split manifest contains AOI leakage")
    observed = {
        "train": _episode_aois(train_episodes),
        "val": _episode_aois(val_episodes),
    }
    wrong_train = observed["train"] - split_aois["train"]
    wrong_val = observed["val"] - split_aois["val"]
    test_overlap = (observed["train"] | observed["val"]) & split_aois["test"]
    if wrong_train or wrong_val or test_overlap:
        raise ValueError(
            "current episode AOIs conflict with reconstructed split: "
            f"wrong_train={sorted(wrong_train)}, wrong_val={sorted(wrong_val)}, "
            f"test_overlap={sorted(test_overlap)}"
        )

    output_root.mkdir(parents=True)
    test_manifest = output_root / "sn7_test_path_manifest.parquet"
    test_aois = output_root / "test_aoi_ids.txt"
    test_frame = frame.loc[frame["split"] == "test"].copy()
    test_frame.to_parquet(test_manifest, index=False)
    test_aois.write_text(
        "".join(f"{value}\n" for value in sorted(split_aois["test"])),
        encoding="utf-8",
    )
    result = {
        "schema_version": "sn7-frozen-split-audit-v1",
        "passed": True,
        "split_aoi_counts": {
            split: len(values) for split, values in split_aois.items()
        },
        "split_row_counts": {
            split: int((frame["split"] == split).sum())
            for split in ("train", "val", "test")
        },
        "observed_episode_aoi_counts": {
            split: len(values) for split, values in observed.items()
        },
        "test_overlap_with_existing_train_val": 0,
        "test_manifest": {
            "path": str(test_manifest.resolve()),
            "sha256": _sha256(test_manifest),
            "rows": len(test_frame),
        },
        "test_aoi_ids": {
            "path": str(test_aois.resolve()),
            "sha256": _sha256(test_aois),
            "count": len(split_aois["test"]),
        },
        "sources": {
            "split_manifest": {
                "path": str(split_manifest.resolve()),
                "sha256": _sha256(split_manifest),
            },
            "train_episodes": {
                "path": str(train_episodes.resolve()),
                "sha256": _sha256(train_episodes),
            },
            "val_episodes": {
                "path": str(val_episodes.resolve()),
                "sha256": _sha256(val_episodes),
            },
        },
        "label_geometries_read": False,
        "test_assets_read": False,
    }
    (output_root / "audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("train_episodes", type=Path)
    parser.add_argument("val_episodes", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                args.split_manifest,
                args.train_episodes,
                args.val_episodes,
                args.output_root,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
