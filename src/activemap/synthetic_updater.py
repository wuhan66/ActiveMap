"""Generate a deterministic image/prior/target smoke dataset for the updater."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import numpy as np

from activemap.models import EditOperation
from activemap.updater_records import UpdaterSample


def _square_mask(size: int, x: int, y: int, width: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float32)
    mask[y : y + width, x : x + width] = 1.0
    return mask


def generate_updater_smoke_dataset(
    output_dir: Path,
    *,
    sample_count: int = 160,
    image_size: int = 32,
    seed: int = 20260710,
) -> Path:
    rng = np.random.default_rng(seed)
    array_dir = output_dir / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "updater_samples.jsonl"
    operations = list(EditOperation)
    aoi_count = ceil(sample_count / len(operations))
    with manifest_path.open("w", encoding="utf-8") as handle:
        for index in range(sample_count):
            operation = operations[index % len(operations)]
            width = int(rng.integers(5, 10))
            x = int(rng.integers(3, image_size - width - 3))
            y = int(rng.integers(3, image_size - width - 3))
            target = _square_mask(image_size, x, y, width)
            prior = target.copy()
            if operation == EditOperation.ADD:
                prior.fill(0.0)
            elif operation == EditOperation.DELETE:
                target.fill(0.0)
            elif operation == EditOperation.RESHAPE:
                shift_x = int(rng.choice([-3, -2, 2, 3]))
                shift_y = int(rng.choice([-3, -2, 2, 3]))
                prior = _square_mask(image_size, x + shift_x, y + shift_y, width)
            target_box: np.ndarray = np.asarray(
                np.asarray([x, y, x + width, y + width], dtype=np.float32) / image_size,
                dtype=np.float32,
            )
            prior_box: np.ndarray
            if operation == EditOperation.ADD:
                prior_box = np.zeros(4, dtype=np.float32)
            elif operation == EditOperation.DELETE:
                target_box = np.zeros(4, dtype=np.float32)
                prior_box = np.asarray(
                    np.asarray([x, y, x + width, y + width], dtype=np.float32)
                    / image_size,
                    dtype=np.float32,
                )
            elif operation == EditOperation.RESHAPE:
                prior_box = np.asarray(
                    np.asarray(
                        [x + shift_x, y + shift_y, x + shift_x + width, y + shift_y + width],
                        dtype=np.float32,
                    )
                    / image_size,
                    dtype=np.float32,
                )
            else:
                prior_box = target_box.copy()
            geometry_delta = np.concatenate([target_box, target_box - prior_box])

            image = np.stack(
                [
                    np.clip(target + rng.normal(0, 0.08, target.shape), 0, 1),
                    np.clip(0.5 * target + rng.normal(0.25, 0.08, target.shape), 0, 1),
                    np.clip(rng.normal(0.35, 0.10, target.shape), 0, 1),
                ],
                axis=0,
            ).astype(np.float32)
            stem = f"sample-{index:05d}"
            image_path = array_dir / f"{stem}-image.npy"
            prior_path = array_dir / f"{stem}-prior.npy"
            target_path = array_dir / f"{stem}-target.npy"
            np.save(image_path, image)
            np.save(prior_path, prior)
            np.save(target_path, target)
            aoi_index = index // len(operations)
            ratio = aoi_index / aoi_count
            split = "train" if ratio < 0.70 else "val" if ratio < 0.85 else "test"
            sample = UpdaterSample(
                sample_id=stem,
                aoi_id=f"smoke-aoi-{aoi_index:03d}",
                split=split,
                image_path=image_path.relative_to(output_dir).as_posix(),
                prior_mask_path=prior_path.relative_to(output_dir).as_posix(),
                target_mask_path=target_path.relative_to(output_dir).as_posix(),
                edit_type=operation,
                geometry_delta=geometry_delta.tolist(),
                object_id=f"object-{index:05d}",
                clear_fraction=1.0,
                quality_source="synthetic",
            )
            handle.write(sample.model_dump_json() + "\n")
    return manifest_path
