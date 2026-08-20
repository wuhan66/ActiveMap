"""Render deterministic updater-crop contact sheets for label quality control."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from activemap.updater_records import UpdaterSample, load_updater_samples


def _rgb(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("QC image must have three dimensions")
    if values.shape[0] in {1, 3, 4}:
        values = np.moveaxis(values[:3], 0, -1)
    if values.shape[-1] == 1:
        values = np.repeat(values, 3, axis=-1)
    if values.shape[-1] < 3:
        raise ValueError("QC image must provide at least one or three channels")
    if values.max(initial=0.0) <= 1.0:
        values *= 255.0
    return np.clip(values[..., :3], 0, 255).astype(np.uint8)


def _mask(path: str | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.ones(shape, dtype=np.float32)
    values = np.load(path).astype(np.float32).squeeze()
    if values.shape != shape:
        raise ValueError(f"QC mask shape mismatch: expected {shape}, got {values.shape}")
    return np.clip(values, 0.0, 1.0)


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = image.astype(np.float32).copy()
    selected = mask >= 0.5
    output[selected] = 0.55 * output[selected] + 0.45 * np.asarray(color, dtype=np.float32)
    return np.clip(output, 0, 255).astype(np.uint8)


def render_updater_sample(sample: UpdaterSample, output: Path) -> None:
    image = _rgb(np.load(sample.image_path))
    prior_image = (
        _rgb(np.load(sample.prior_image_path))
        if sample.prior_image_path is not None
        else None
    )
    if prior_image is not None and prior_image.shape != image.shape:
        raise ValueError(
            f"QC temporal image shape mismatch for {sample.sample_id}: "
            f"{prior_image.shape} vs {image.shape}"
        )
    height, width = image.shape[:2]
    prior = _mask(sample.prior_mask_path, (height, width))
    target = _mask(sample.target_mask_path, (height, width))
    valid = _mask(sample.valid_mask_path, (height, width))
    panels = (
        [
            prior_image,
            _overlay(prior_image, prior, (40, 120, 255)),
            image,
            _overlay(image, target, (30, 220, 90)),
            _overlay(image, 1.0 - valid, (240, 55, 55)),
        ]
        if prior_image is not None
        else [
            image,
            _overlay(image, prior, (40, 120, 255)),
            _overlay(image, target, (30, 220, 90)),
            _overlay(image, 1.0 - valid, (240, 55, 55)),
        ]
    )
    header = 28
    canvas = Image.new("RGB", (width * len(panels), height + header), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    labels = (
        ("image t-1", "prior", "image t", "target", "invalid")
        if prior_image is not None
        else ("image", "prior", "target", "invalid")
    )
    for index, (panel, label) in enumerate(zip(panels, labels, strict=True)):
        canvas.paste(Image.fromarray(panel), (index * width, header))
        draw.text((index * width + 5, 7), label, fill=(20, 20, 20))
    draw.text(
        (width * len(panels) - 150, 7),
        sample.edit_type.value,
        fill=(20, 20, 20),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def render_updater_qc(
    samples_path: Path,
    output_dir: Path,
    *,
    count: int = 32,
    seed: int = 20260710,
    sample_ids: set[str] | None = None,
    splits: set[str] | None = None,
) -> dict[str, object]:
    samples = load_updater_samples(samples_path)
    source_split_counts = Counter(sample.split for sample in samples)
    if splits is not None:
        unknown = splits - {"train", "val", "test"}
        if unknown:
            raise ValueError(f"unknown QC splits: {sorted(unknown)}")
        samples = [sample for sample in samples if sample.split in splits]
        if not samples:
            raise ValueError(f"no updater samples found for QC splits={sorted(splits)}")
    if sample_ids is not None:
        by_id = {sample.sample_id: sample for sample in samples}
        missing = sample_ids - set(by_id)
        if missing:
            raise ValueError(f"QC sample IDs not found: {sorted(missing)}")
        selected = [by_id[sample_id] for sample_id in sorted(sample_ids)]
    else:
        rng = np.random.default_rng(seed)
        selected_indices = rng.choice(
            len(samples), size=min(count, len(samples)), replace=False
        )
        selected = [samples[int(index)] for index in sorted(selected_indices)]
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for sample in selected:
        output = output_dir / f"{sample.sample_id}.png"
        render_updater_sample(sample, output)
        files.append(output.name)
    summary: dict[str, object] = {
        "source": str(samples_path.resolve()),
        "total_samples": len(samples),
        "rendered": len(selected),
        "seed": seed if sample_ids is None else None,
        "requested_sample_ids": sorted(sample_ids) if sample_ids is not None else None,
        "allowed_splits": sorted(splits) if splits is not None else None,
        "source_split_counts": dict(sorted(source_split_counts.items())),
        "eligible_split_counts": dict(sorted(Counter(sample.split for sample in samples).items())),
        "rendered_split_counts": dict(sorted(Counter(sample.split for sample in selected).items())),
        "test_assets_rendered": any(sample.split == "test" for sample in selected),
        "operation_counts": dict(
            sorted(Counter(sample.edit_type.value for sample in samples).items())
        ),
        "files": files,
    }
    (output_dir / "index.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
