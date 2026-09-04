#!/usr/bin/env python3
"""Render a scientific QC panel for a model-segmentation tool smoke result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from activemap.geo_tools.raster import _read
from activemap.models import EpisodeRecord
from activemap.updater_records import load_updater_samples


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = left | right
    return float(np.sum(left & right) / max(np.sum(union), 1))


def _rgb(image: np.ndarray) -> np.ndarray:
    array = np.moveaxis(image[:3], 0, -1).astype(np.float32)
    low, high = np.percentile(array, (2, 98))
    return np.clip((array - low) / max(high - low, 1e-6), 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("updater_manifest", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("split") != "val" or report.get("test_assets_read") is not False:
        raise ValueError("renderer accepts only audited validation smoke reports")
    sample = next(
        item
        for item in load_updater_samples(args.updater_manifest, split="val")
        if item.sample_id == report["sample_id"]
    )
    episodes = {}
    with args.episodes.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                episode = EpisodeRecord.model_validate_json(line)
                if episode.split == "test":
                    raise ValueError("renderer input must not contain test episodes")
                episodes[episode.episode_id] = episode
    episode = episodes[report["episode_id"]]
    evidence = next(
        item for item in episode.evidence_catalog if item.evidence_id == report["evidence_id"]
    )
    x_min, y_min, x_max, y_max = evidence.region
    image, _, _ = _read(
        Path(evidence.image_path),
        {
            "pixel_window": [x_min, y_min, x_max - x_min, y_max - y_min],
            "out_size": [512, 512],
        },
        force_bands=[1, 2, 3],
    )
    prior = np.asarray(np.load(sample.prior_mask_path)).squeeze() >= 0.5
    target = np.asarray(np.load(sample.target_mask_path)).squeeze() >= 0.5
    prediction = np.asarray(
        np.load(report["result"]["artifacts"][0]), dtype=np.uint8
    ).squeeze() > 0
    metrics = {
        "prior_target_iou": _iou(prior, target),
        "prediction_target_iou": _iou(prediction, target),
        "prediction_prior_iou": _iou(prediction, prior),
    }

    overlays = [
        (prior, "Old editable road", "#ffbf00"),
        (prediction, "Model segmentation", "#00c8c8"),
        (target, "Committed target", "#28c76f"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
    axes[0].imshow(_rgb(image))
    axes[0].set_title("2019 evidence")
    for axis, (mask, title, color) in zip(axes[1:4], overlays, strict=True):
        axis.imshow(_rgb(image))
        axis.imshow(mask, cmap=ListedColormap(["none", color]), alpha=0.72, vmin=0, vmax=1)
        axis.set_title(title)
    axes[4].imshow(_rgb(image))
    axes[4].imshow(prior, cmap=ListedColormap(["none", "#ffbf00"]), alpha=0.45)
    axes[4].contour(prediction, levels=[0.5], colors=["#00c8c8"], linewidths=1.0)
    axes[4].contour(target, levels=[0.5], colors=["#28c76f"], linewidths=1.0)
    axes[4].set_title(
        "QC overlay\n"
        f"prior-target {metrics['prior_target_iou']:.3f} | "
        f"pred-target {metrics['prediction_target_iou']:.3f}"
    )
    for axis in axes:
        axis.set_axis_off()
    fig.suptitle(
        f"{sample.sample_id} | GT {report['gt_edit']} | "
        f"pred-prior IoU {metrics['prediction_prior_iou']:.3f}",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **metrics}, indent=2))


if __name__ == "__main__":
    main()
