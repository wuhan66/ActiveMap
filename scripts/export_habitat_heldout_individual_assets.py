#!/usr/bin/env python3
"""Expose all held-out Habitat RGB and occupancy assets without composition."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SEEDS = tuple(range(20270862, 20270870))
POLICIES = ("all", "unknown15", "novelty")


def _preserve_source(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _occupancy_png(source: Path, destination: Path) -> None:
    array = np.asarray(np.load(source), dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected a two-dimensional occupancy map: {source}")
    Image.fromarray(np.where(array >= 0.5, 0, 255).astype(np.uint8), mode="L").save(destination)


def export(source_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    index: list[dict[str, Any]] = []
    for seed in SEEDS:
        case_dir = output_root / "cases" / f"seed{seed}"
        case_dir.mkdir(parents=True)
        policy_assets: dict[str, Any] = {}
        for policy in POLICIES:
            source_dir = source_root / f"{policy}_seed{seed}"
            occupancy_source = source_dir / "committed_occupancy.npy"
            if not occupancy_source.is_file():
                raise FileNotFoundError(occupancy_source)
            policy_dir = case_dir / policy
            raw_destination = policy_dir / "committed_occupancy.npy"
            png_destination = policy_dir / "committed_occupancy.png"
            preservation = _preserve_source(occupancy_source, raw_destination)
            _occupancy_png(occupancy_source, png_destination)
            policy_assets[policy] = {
                "occupancy_array": str(raw_destination.relative_to(case_dir)),
                "occupancy_image": str(png_destination.relative_to(case_dir)),
                "preservation": preservation,
                "source": str(occupancy_source),
            }
        # The matched acquire-all replay supplies the shared RGB observations.
        rgb_source_dir = source_root / f"all_seed{seed}"
        rgb_files = sorted(rgb_source_dir.glob("rgb_*.png"))
        if not rgb_files:
            raise FileNotFoundError(f"no RGB frames in {rgb_source_dir}")
        rgb_assets = []
        for source in rgb_files:
            destination = case_dir / "rgb" / source.name
            rgb_assets.append(
                {
                    "asset": str(destination.relative_to(case_dir)),
                    "preservation": _preserve_source(source, destination),
                    "source": str(source),
                }
            )
        row = {
            "schema_version": "habitat-heldout-individual-assets-v1",
            "dataset": "Habitat RGB-D occupancy mapping",
            "split": "heldout",
            "test_assets_read": False,
            "trajectory_seed": seed,
            "folder": str(case_dir.relative_to(output_root)),
            "rgb_frames": rgb_assets,
            "policy_assets": policy_assets,
            "composition": "none; raw RGB frames and each policy occupancy map are individual files",
            "scope": "development portability visualization; shared motion and frozen acquisition gates",
        }
        (case_dir / "manifest.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        index.append(row)
    with (output_root / "asset_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in index:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "schema_version": "habitat-heldout-individual-assets-v1",
        "dataset": "Habitat RGB-D occupancy mapping",
        "split": "heldout",
        "test_assets_read": False,
        "trajectory_count": len(index),
        "policies": list(POLICIES),
        "composition": "none; all RGB and policy-map inputs are separate files",
        "scope": "development portability only; no cross-domain superiority claim",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.source_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
