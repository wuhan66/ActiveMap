#!/usr/bin/env python3
"""Render a deterministic, edit-stratified QC grid for an RSPrompter COCO export."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image


def _select_images(images: list[dict], count: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for image in sorted(images, key=lambda item: str(item["activemap_sample_id"])):
        grouped[str(image["edit_type"])].append(image)
    selected: list[dict] = []
    order = ["KEEP", "ADD", "DELETE", "RESHAPE"]
    while len(selected) < min(count, len(images)):
        changed = False
        for edit in order:
            bucket = grouped[edit]
            if bucket:
                selected.append(bucket.pop(0))
                changed = True
                if len(selected) == min(count, len(images)):
                    break
        if not changed:
            break
    return selected


def _decode_rle(segmentation: dict) -> np.ndarray:
    height, width = (int(value) for value in segmentation["size"])
    values: list[int] = []
    foreground = 0
    for count in segmentation["counts"]:
        values.extend([foreground] * int(count))
        foreground = 1 - foreground
    if len(values) != height * width:
        raise ValueError("RLE does not match declared image size")
    return np.asarray(values, dtype=bool).reshape((height, width), order="F")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("split", choices=("train", "val"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    if args.count <= 0 or args.columns <= 0:
        raise ValueError("count and columns must be positive")
    audit = json.loads((args.dataset_dir / "audit.json").read_text(encoding="utf-8"))
    if not audit.get("valid") or audit.get("test_assets_read") is not False:
        raise ValueError("QC requires a valid, test-free audited export")
    payload = json.loads(
        (args.dataset_dir / "annotations" / f"{args.split}.json").read_text(
            encoding="utf-8"
        )
    )
    selected = _select_images(payload["images"], args.count)
    annotations: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)

    rows = math.ceil(len(selected) / args.columns)
    fig, axes = plt.subplots(
        rows,
        args.columns,
        figsize=(4 * args.columns, 4 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, image_record in zip(axes.flat, selected, strict=False):
        image_path = args.dataset_dir / "images" / args.split / image_record["file_name"]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        axis.imshow(image)
        combined = np.zeros(image.shape[:2], dtype=bool)
        for annotation in annotations[int(image_record["id"])]:
            combined |= _decode_rle(annotation["segmentation"])
        axis.imshow(
            combined,
            cmap=ListedColormap([(0.0, 0.0, 0.0, 0.0), (0.0, 0.78, 0.78, 0.5)]),
            vmin=0,
            vmax=1,
        )
        if np.any(combined) and not np.all(combined):
            axis.contour(combined, levels=[0.5], colors=["#00f28d"], linewidths=0.7)
        axis.set_title(
            f"{image_record['edit_type']} | {len(annotations[int(image_record['id'])])} masks\n"
            f"{image_record['activemap_sample_id']}",
            fontsize=8,
        )
        axis.set_axis_off()
    for axis in axes.flat[len(selected) :]:
        axis.set_axis_off()
    fig.suptitle(
        f"MUNO21 RSPrompter road COCO QC | {args.split} | test assets not read",
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "images": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
