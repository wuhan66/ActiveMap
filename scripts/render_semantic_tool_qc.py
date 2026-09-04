#!/usr/bin/env python3
"""Render acquired image, prior, semantic evidence, change, and target panels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from activemap.agent.identifiers import public_evidence_id, public_task_id
from activemap.agent.tool_belief_data import PostAcquisitionToolPairExample
from activemap.geo_tools.raster import _read
from activemap.models import EpisodeRecord
from activemap.updater_records import UpdaterSample, load_updater_samples


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _align(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask).squeeze()
    if array.shape != shape:
        array = ndimage.zoom(
            array,
            (shape[0] / array.shape[0], shape[1] / array.shape[1]),
            order=0,
            prefilter=False,
        )
    if array.shape != shape:
        raise ValueError(f"could not align {array.shape} to {shape}")
    return array >= 0.5


def _rgb(image: np.ndarray) -> np.ndarray:
    output = np.moveaxis(np.asarray(image[:3], dtype=np.float32), 0, -1)
    for channel in range(3):
        values = output[..., channel]
        low, high = np.nanpercentile(values, (2, 98))
        output[..., channel] = np.clip((values - low) / max(high - low, 1e-6), 0, 1)
    return np.nan_to_num(output)


def _overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[float, ...]) -> np.ndarray:
    output = rgb.copy()
    output[mask] = 0.55 * output[mask] + 0.45 * np.asarray(color)
    return output


def _select(
    rows: list[PostAcquisitionToolPairExample], count: int
) -> list[PostAcquisitionToolPairExample]:
    by_operation: dict[str, list[PostAcquisitionToolPairExample]] = defaultdict(list)
    for row in rows:
        by_operation[row.gt_edit.value].append(row)
    selected = []
    while len(selected) < min(count, len(rows)):
        changed = False
        for operation in ("KEEP", "ADD", "DELETE", "RESHAPE"):
            if by_operation[operation]:
                selected.append(by_operation[operation].pop(0))
                changed = True
                if len(selected) == count:
                    break
        if not changed:
            break
    return selected


def render(
    rows: list[PostAcquisitionToolPairExample],
    episodes: dict[str, EpisodeRecord],
    samples: dict[str, UpdaterSample],
    output: Path,
    *,
    count: int,
) -> None:
    selected = _select(rows, count)
    if not selected:
        raise ValueError("no semantic examples to render")
    figure, axes = plt.subplots(len(selected), 5, figsize=(15, 3 * len(selected)))
    axes = np.asarray(axes).reshape(len(selected), 5)
    for row_index, row in enumerate(selected):
        if row.semantic_result is None or not row.semantic_result.artifacts:
            raise ValueError(f"missing semantic artifact: {row.example_id}")
        episode, sample = episodes[row.task_id], samples[row.task_id]
        evidence = next(
            item
            for item in episode.evidence_catalog
            if public_evidence_id(item.evidence_id) == row.evidence_id
        )
        prior_raw = np.load(sample.prior_mask_path)
        shape = tuple(int(value) for value in np.asarray(prior_raw).squeeze().shape)
        x0, y0, x1, y1 = evidence.region
        image, _, _ = _read(
            Path(evidence.image_path),
            {
                "pixel_window": [x0, y0, x1 - x0, y1 - y0],
                "out_size": [shape[0], shape[1]],
            },
            force_bands=[1, 2, 3],
        )
        rgb = _rgb(image)
        prior = _align(prior_raw, shape)
        semantic = _align(np.load(row.semantic_result.artifacts[0]), shape)
        target = _align(np.load(sample.target_mask_path), shape)
        add, remove = semantic & ~prior, prior & ~semantic
        change = _overlay(_overlay(rgb, add, (0.0, 0.9, 0.7)), remove, (1.0, 0.2, 0.2))
        panels = (
            (rgb, "Acquired image"),
            (_overlay(rgb, prior, (1.0, 0.75, 0.0)), "Editable prior"),
            (_overlay(rgb, semantic, (0.0, 0.8, 0.9)), "SAM-Road evidence"),
            (change, "ADD / REMOVE"),
            (_overlay(rgb, target, (0.2, 1.0, 0.3)), "Target"),
        )
        for axis, (panel, title) in zip(axes[row_index], panels, strict=True):
            axis.imshow(panel)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        outputs = row.semantic_result.outputs
        axes[row_index, 0].set_ylabel(
            f"{row.gt_edit.value}\nIoU(prior)={outputs['prior_iou']:.3f}\n"
            f"add={outputs['add_fraction']:.3f} remove={outputs['remove_fraction']:.3f}",
            fontsize=8,
        )
    figure.suptitle("ActiveMap independent semantic-tool QC", fontsize=13)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("semantic_jsonl", type=Path)
    parser.add_argument("episodes_jsonl", type=Path)
    parser.add_argument("updater_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    rows = _read_jsonl(args.semantic_jsonl, PostAcquisitionToolPairExample)
    task_ids = {row.task_id for row in rows}
    episodes = {
        public_task_id(row.episode_id): row
        for row in _read_jsonl(args.episodes_jsonl, EpisodeRecord)
        if public_task_id(row.episode_id) in task_ids
    }
    samples = {
        public_task_id(f"{row.sample_id}__temporal"): row
        for row in load_updater_samples(args.updater_manifest)
        if public_task_id(f"{row.sample_id}__temporal") in task_ids
    }
    if task_ids != set(episodes) or task_ids != set(samples):
        raise ValueError("semantic rows, episodes, and updater samples differ")
    render(rows, episodes, samples, args.output, count=args.count)
    print(json.dumps({"rows": len(rows), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
