"""Run a trained updater over every episode candidate to supervise the selector."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from affine import Affine
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from tqdm.auto import tqdm

try:
    import h5py
except ImportError:  # pragma: no cover - exercised only in minimal non-ML installs.
    h5py = None

from activemap.agent.writeback import apply_typed_binary_edit, effective_operation_from_masks
from activemap.data.muno21 import rasterize_muno_road_mask
from activemap.data.prior_input_corruption import (
    deterministic_prior_translation,
    morph_prior_no_wrap,
)
from activemap.data.raster_masks import rasterize_geometry_mask
from activemap.evaluation.episode_utility import UTILITY_PROFILES
from activemap.features import EVIDENCE_DIM, HYPOTHESIS_DIM, STATE_DIM
from activemap.geometry import geometry_features
from activemap.models import EditOperation, EpisodeRecord, EvidenceItem
from activemap.nn.operation_selector import (
    OperationSelector,
    OperationSelectorConfig,
    operation_selector_inputs,
)
from activemap.nn.updater import (
    PriorConditionedUNet,
    UpdaterConfig,
    operation_probabilities,
)
from activemap.selector_records import SelectorSample
from activemap.synthetic import write_selector_samples
from activemap.training.selector import resolve_device

EDIT_ORDER = list(EditOperation)
ORACLE_INPUT_CACHE_SCHEMA = "activemap-oracle-input-cache-v1"


def _require_h5py() -> Any:
    if h5py is None:
        raise RuntimeError(
            "HDF5 input caching requires h5py. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        )
    return h5py


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_episodes(
    episodes_path: Path,
    *,
    asset_root_maps: tuple[tuple[Path, Path], ...],
    splits: tuple[str, ...],
) -> list[EpisodeRecord]:
    requested_splits = set(splits)
    return remap_episode_assets(
        [
            episode
            for episode in load_episodes(episodes_path)
            if episode.split in requested_splits
        ],
        asset_root_maps,
    )


@dataclass
class _CachedOracleInputs:
    """Read-only HDF5 candidate tensors aligned with a fixed episode manifest."""

    handle: Any
    episode_offsets: np.ndarray
    episode_index: dict[str, int]

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        episodes_path: Path,
        episodes: list[EpisodeRecord],
        image_size: int,
        image_channels: int,
        temporal_pair_input: bool,
    ) -> _CachedOracleInputs:
        h5 = _require_h5py()
        handle = h5.File(
            path,
            "r",
            libver="latest",
            swmr=True,
            rdcc_nbytes=128 * 1024 * 1024,
            rdcc_nslots=10_007,
        )
        try:
            if str(handle.attrs.get("schema", "")) != ORACLE_INPUT_CACHE_SCHEMA:
                raise ValueError(f"unsupported oracle input cache schema: {path}")
            if int(handle.attrs["image_size"]) != image_size:
                raise ValueError(
                    f"cache image size {handle.attrs['image_size']} does not match {image_size}"
                )
            if int(handle.attrs["image_channels"]) != image_channels:
                raise ValueError(
                    "cache image channel count does not match the updater checkpoint"
                )
            if bool(handle.attrs.get("temporal_pair_input", False)) != temporal_pair_input:
                raise ValueError(
                    "cache temporal input mode does not match the updater checkpoint"
                )
            if str(handle.attrs.get("source_manifest_sha256", "")) != _file_sha256(
                episodes_path
            ):
                raise ValueError("cache manifest hash does not match the requested episodes")
            cached_ids = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in handle["episode_ids"][:]
            ]
            expected_ids = [episode.episode_id for episode in episodes]
            if cached_ids != expected_ids:
                raise ValueError(
                    "cache episode order does not match the requested manifest; rebuild the cache"
                )
            offsets = np.asarray(handle["episode_offsets"][:], dtype=np.int64)
            if offsets.shape != (len(episodes) + 1,) or int(offsets[0]) != 0:
                raise ValueError("invalid oracle input cache offsets")
            return cls(
                handle=handle,
                episode_offsets=offsets,
                episode_index={episode_id: index for index, episode_id in enumerate(cached_ids)},
            )
        except Exception:
            handle.close()
            raise

    def close(self) -> None:
        self.handle.close()

    def read(self, episode: EpisodeRecord) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[tuple[int, int]],
    ]:
        index = self.episode_index.get(episode.episode_id)
        if index is None:
            raise KeyError(f"episode missing from oracle input cache: {episode.episode_id}")
        start, stop = (int(value) for value in self.episode_offsets[index : index + 2])
        if stop - start != len(episode.evidence_catalog):
            raise ValueError(
                f"cache candidate count does not match episode {episode.episode_id}"
            )
        shapes = np.asarray(self.handle["raster_shapes"][start:stop], dtype=np.int32)
        return (
            np.asarray(self.handle["images"][start:stop], dtype=np.float32),
            np.asarray(self.handle["priors"][start:stop], dtype=np.float32),
            np.asarray(self.handle["targets"][start:stop], dtype=np.float32),
            np.asarray(self.handle["valid_masks"][start:stop], dtype=np.float32),
            [(int(width), int(height)) for width, height in shapes],
        )


def _operation_errors(
    target: EditOperation, prediction: EditOperation
) -> tuple[bool, bool, bool]:
    false_edit = target == EditOperation.KEEP and prediction != target
    missed_edit = target != EditOperation.KEEP and prediction == EditOperation.KEEP
    wrong_edit = (
        target != EditOperation.KEEP
        and prediction != EditOperation.KEEP
        and prediction != target
    )
    return false_edit, missed_edit, wrong_edit


@lru_cache(maxsize=4)
def _load_cached_rgb_jpeg(path: str) -> Image.Image:
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as source:
            return source.convert("RGB").copy()
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


def load_episodes(path: Path, *, split: str | None = None) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = EpisodeRecord.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid episode at line {line_number}") from exc
            if split is None or record.split == split:
                records.append(record)
    if not records:
        raise ValueError(f"no episodes found for split={split!r} in {path}")
    return records


def _remap_asset_path(
    value: str | None, mappings: tuple[tuple[Path, Path], ...]
) -> str | None:
    if value is None:
        return None
    original = Path(value)
    for source, target in mappings:
        try:
            return str(target / original.relative_to(source))
        except ValueError:
            continue
    return value


def remap_episode_assets(
    episodes: list[EpisodeRecord], mappings: tuple[tuple[Path, Path], ...]
) -> list[EpisodeRecord]:
    if not mappings:
        return episodes
    return [
        episode.model_copy(
            update={
                "evidence_catalog": [
                    item.model_copy(
                        update={
                            "image_path": _remap_asset_path(item.image_path, mappings),
                            "udm_path": _remap_asset_path(item.udm_path, mappings),
                            "prior_image_path": _remap_asset_path(
                                item.prior_image_path, mappings
                            ),
                            "prior_udm_path": _remap_asset_path(item.prior_udm_path, mappings),
                        }
                    )
                    for item in episode.evidence_catalog
                ]
            }
        )
        for episode in episodes
    ]


def _month_index(timestamp: str) -> int:
    year, month = (int(value) for value in timestamp.replace("-", "_").split("_")[:2])
    return year * 12 + month


def _geometry(record: Any) -> BaseGeometry | None:
    return shape(record.model_dump(mode="json")) if record is not None else None


def _normalize_image(array: np.ndarray) -> np.ndarray:
    output = array.astype(np.float32)
    maximum = float(output.max(initial=0.0))
    if maximum > 1.0:
        denominator = 255.0 if maximum <= 255.0 else 10000.0 if maximum <= 10000 else maximum
        output /= denominator
    return np.clip(np.nan_to_num(output), 0.0, 1.0)


def _read_candidate(
    item: EvidenceItem,
    *,
    prior_geometry: BaseGeometry | None,
    target_geometry: BaseGeometry | None,
    image_size: int,
    image_channels: int,
    temporal_pair_input: bool = False,
    road_width_source_pixels: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    if temporal_pair_input and image_channels != 6:
        raise ValueError("temporal_pair_input requires six image channels")
    source_channels = image_channels // 2 if temporal_pair_input else image_channels
    x_min, y_min, x_max, y_max = item.region
    window = Window(x_min, y_min, x_max - x_min, y_max - y_min)
    if Path(item.image_path).suffix.lower() in {".jpg", ".jpeg"} and item.udm_path is None:
        if temporal_pair_input:
            raise ValueError(
                "paired temporal oracle inputs require georeferenced raster evidence; "
                f"JPEG evidence is unsupported for {item.evidence_id}"
            )
        source = _load_cached_rgb_jpeg(item.image_path)
        crop = source.crop((x_min, y_min, x_max, y_max)).resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
        image = np.moveaxis(np.asarray(crop, dtype=np.float32), -1, 0)
        while image.shape[0] < source_channels:
            image = np.concatenate([image, image[-1:]], axis=0)
        image = image[:source_channels]
        transform = Affine.translation(x_min, y_min) * Affine.scale(
            float(window.width) / image_size,
            float(window.height) / image_size,
        )
        if road_width_source_pixels is None:
            prior = rasterize_geometry_mask(prior_geometry, transform, image_size)
            target = rasterize_geometry_mask(target_geometry, transform, image_size)
        else:
            prior = rasterize_muno_road_mask(
                prior_geometry, transform, image_size, road_width_source_pixels
            )
            target = rasterize_muno_road_mask(
                target_geometry, transform, image_size, road_width_source_pixels
            )
        normalized_image = _normalize_image(image)
        valid = np.any(normalized_image > 1e-6, axis=0).astype(np.float32)
        return normalized_image, prior, target, valid, source.size
    with rasterio.open(item.image_path) as dataset:
        indexes = list(range(1, min(dataset.count, source_channels) + 1))
        image = dataset.read(
            indexes,
            window=window,
            out_shape=(len(indexes), image_size, image_size),
            boundless=True,
            fill_value=0,
            resampling=Resampling.bilinear,
        )
        while image.shape[0] < source_channels:
            image = np.concatenate([image, image[-1:]], axis=0)
        transform = dataset.window_transform(window) * Affine.scale(
            float(window.width) / image_size,
            float(window.height) / image_size,
        )
        raster_bounds = window_bounds(window, dataset.transform)
        raster_shape = (dataset.width, dataset.height)
        current_crs = dataset.crs
    if road_width_source_pixels is None:
        prior = rasterize_geometry_mask(prior_geometry, transform, image_size)
        target = rasterize_geometry_mask(target_geometry, transform, image_size)
    else:
        prior = rasterize_muno_road_mask(
            prior_geometry, transform, image_size, road_width_source_pixels
        )
        target = rasterize_muno_road_mask(
            target_geometry, transform, image_size, road_width_source_pixels
        )
    normalized_image = _normalize_image(image[:source_channels])
    valid = np.any(normalized_image > 1e-6, axis=0).astype(np.float32)
    if item.udm_path is not None and Path(item.udm_path).is_file():
        with rasterio.open(item.udm_path) as udm:
            udm_window = from_bounds(*raster_bounds, transform=udm.transform)
            values = udm.read(
                1,
                window=udm_window,
                out_shape=(image_size, image_size),
                boundless=True,
                fill_value=255,
                resampling=Resampling.nearest,
            )
            valid = ((values == 0) & (valid >= 0.5)).astype(np.float32)
    if temporal_pair_input:
        if item.prior_image_path is None:
            raise ValueError(
                "paired temporal oracle inputs require prior_image_path; "
                f"missing from {item.evidence_id}"
            )
        with rasterio.open(item.prior_image_path) as prior_dataset:
            if prior_dataset.crs != current_crs:
                raise ValueError(
                    "paired temporal oracle inputs require matching raster CRS: "
                    f"{item.prior_image_path} vs {item.image_path}"
                )
            prior_window = from_bounds(*raster_bounds, transform=prior_dataset.transform)
            prior_indexes = list(range(1, min(prior_dataset.count, source_channels) + 1))
            prior_image = prior_dataset.read(
                prior_indexes,
                window=prior_window,
                out_shape=(len(prior_indexes), image_size, image_size),
                boundless=True,
                fill_value=0,
                resampling=Resampling.bilinear,
            )
        while prior_image.shape[0] < source_channels:
            prior_image = np.concatenate([prior_image, prior_image[-1:]], axis=0)
        normalized_prior_image = _normalize_image(prior_image[:source_channels])
        prior_valid = np.any(normalized_prior_image > 1e-6, axis=0).astype(np.float32)
        if item.prior_udm_path is not None and Path(item.prior_udm_path).is_file():
            with rasterio.open(item.prior_udm_path) as prior_udm:
                prior_udm_window = from_bounds(*raster_bounds, transform=prior_udm.transform)
                prior_values = prior_udm.read(
                    1,
                    window=prior_udm_window,
                    out_shape=(image_size, image_size),
                    boundless=True,
                    fill_value=255,
                    resampling=Resampling.nearest,
                )
                prior_valid = ((prior_values == 0) & (prior_valid >= 0.5)).astype(np.float32)
        normalized_image = np.concatenate((normalized_prior_image, normalized_image), axis=0)
        valid = valid * prior_valid
    return normalized_image, prior, target, valid, raster_shape


def build_selector_oracle_input_cache(
    episodes_path: Path,
    output: Path,
    *,
    image_size: int,
    image_channels: int,
    temporal_pair_input: bool = False,
    asset_root_maps: tuple[tuple[Path, Path], ...] = (),
    splits: tuple[str, ...] = ("train", "val"),
    chunk_candidates: int = 16,
    compression: str = "lzf",
    max_episodes: int | None = None,
    frozen_test: bool = False,
) -> dict[str, Any]:
    """Materialize raw counterfactual inputs into an immutable HDF5 shard.

    The cache contains only standardised imagery and binary raster inputs that the
    uncached oracle would construct at runtime. Geometry, candidate metadata, and
    all utility/writeback logic remain in the normal manifest-driven code path.
    """
    if image_size < 16 or image_channels < 1:
        raise ValueError("image_size must be at least 16 and image_channels must be positive")
    if temporal_pair_input and image_channels != 6:
        raise ValueError("temporal_pair_input requires image_channels=6")
    if chunk_candidates < 1:
        raise ValueError("chunk_candidates must be positive")
    if compression not in {"lzf", "gzip", "none"}:
        raise ValueError("compression must be one of lzf, gzip, none")
    requested_splits = set(splits)
    if not requested_splits or not requested_splits <= {"train", "val", "test"}:
        raise ValueError("oracle cache splits must be train, val, or test")
    test_requested = "test" in requested_splits
    if test_requested:
        if requested_splits != {"test"}:
            raise ValueError("frozen oracle-cache construction must be test-only")
        if not frozen_test:
            raise PermissionError("test oracle-cache construction requires --frozen-test")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    elif frozen_test:
        raise ValueError("--frozen-test is valid only for the test split")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing oracle input cache: {output}")
    partial = output.with_name(f"{output.name}.partial")
    if partial.exists():
        raise FileExistsError(f"refusing to overwrite partial oracle input cache: {partial}")

    episodes = _selected_episodes(
        episodes_path,
        asset_root_maps=asset_root_maps,
        splits=splits,
    )
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise ValueError(f"no episodes found for splits={sorted(requested_splits)}")
    candidate_count = sum(len(episode.evidence_catalog) for episode in episodes)
    if candidate_count < 1:
        raise ValueError("cannot build an oracle input cache without evidence candidates")
    h5 = _require_h5py()
    output.parent.mkdir(parents=True, exist_ok=True)
    h5_compression = None if compression == "none" else compression
    compression_kwargs: dict[str, Any] = {"compression": h5_compression}
    if compression == "gzip":
        compression_kwargs["compression_opts"] = 1
    image_chunk = (
        min(chunk_candidates, candidate_count),
        image_channels,
        image_size,
        image_size,
    )
    mask_chunk = (min(chunk_candidates, candidate_count), image_size, image_size)
    offsets = np.zeros(len(episodes) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(
        [len(episode.evidence_catalog) for episode in episodes], dtype=np.int64
    )

    try:
        with h5.File(partial, "w", libver="latest") as handle:
            handle.attrs["schema"] = ORACLE_INPUT_CACHE_SCHEMA
            handle.attrs["image_size"] = image_size
            handle.attrs["image_channels"] = image_channels
            handle.attrs["temporal_pair_input"] = temporal_pair_input
            handle.attrs["source_manifest_sha256"] = _file_sha256(episodes_path)
            handle.attrs["source_manifest"] = str(episodes_path.resolve())
            handle.attrs["splits"] = ",".join(sorted(requested_splits))
            handle.attrs["compression"] = compression
            handle.attrs["test_assets_read"] = test_requested
            images = handle.create_dataset(
                "images",
                shape=(candidate_count, image_channels, image_size, image_size),
                dtype=np.float32,
                chunks=image_chunk,
                **compression_kwargs,
            )
            priors = handle.create_dataset(
                "priors",
                shape=(candidate_count, image_size, image_size),
                dtype=np.uint8,
                chunks=mask_chunk,
                **compression_kwargs,
            )
            targets = handle.create_dataset(
                "targets",
                shape=(candidate_count, image_size, image_size),
                dtype=np.uint8,
                chunks=mask_chunk,
                **compression_kwargs,
            )
            valid_masks = handle.create_dataset(
                "valid_masks",
                shape=(candidate_count, image_size, image_size),
                dtype=np.uint8,
                chunks=mask_chunk,
                **compression_kwargs,
            )
            raster_shapes = handle.create_dataset(
                "raster_shapes", shape=(candidate_count, 2), dtype=np.int32
            )
            string_dtype = h5.string_dtype(encoding="utf-8")
            handle.create_dataset(
                "episode_ids",
                data=[episode.episode_id for episode in episodes],
                dtype=string_dtype,
            )
            handle.create_dataset("episode_offsets", data=offsets, dtype=np.int64)

            cursor = 0
            image_buffer: list[np.ndarray] = []
            prior_buffer: list[np.ndarray] = []
            target_buffer: list[np.ndarray] = []
            valid_buffer: list[np.ndarray] = []
            shape_buffer: list[tuple[int, int]] = []

            def flush_candidate_buffer() -> None:
                nonlocal cursor
                if not image_buffer:
                    return
                stop = cursor + len(image_buffer)
                images[cursor:stop] = np.stack(image_buffer)
                priors[cursor:stop] = np.stack(prior_buffer)
                targets[cursor:stop] = np.stack(target_buffer)
                valid_masks[cursor:stop] = np.stack(valid_buffer)
                raster_shapes[cursor:stop] = np.asarray(shape_buffer, dtype=np.int32)
                cursor = stop
                image_buffer.clear()
                prior_buffer.clear()
                target_buffer.clear()
                valid_buffer.clear()
                shape_buffer.clear()

            for episode in tqdm(episodes, desc="Cache oracle inputs"):
                prior_geometry = _geometry(episode.prior_geometry)
                target_geometry = _geometry(episode.target_geometry)
                road_width = episode.metadata.get("road_width_source_pixels")
                for item in episode.evidence_catalog:
                    image, prior, target, valid, raster_shape = _read_candidate(
                        item,
                        prior_geometry=prior_geometry,
                        target_geometry=target_geometry,
                        image_size=image_size,
                        image_channels=image_channels,
                        temporal_pair_input=temporal_pair_input,
                        road_width_source_pixels=(
                            float(road_width) if road_width is not None else None
                        ),
                    )
                    image_buffer.append(image)
                    prior_buffer.append(np.asarray(prior >= 0.5, dtype=np.uint8))
                    target_buffer.append(np.asarray(target >= 0.5, dtype=np.uint8))
                    valid_buffer.append(np.asarray(valid >= 0.5, dtype=np.uint8))
                    shape_buffer.append(raster_shape)
                    if len(image_buffer) == chunk_candidates:
                        flush_candidate_buffer()
            flush_candidate_buffer()
            if cursor != candidate_count:
                raise RuntimeError("oracle input cache wrote an unexpected candidate count")
            handle.flush()
            handle.swmr_mode = True
        os.replace(partial, output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return {
        "cache": str(output.resolve()),
        "episodes": len(episodes),
        "candidates": candidate_count,
        "image_size": image_size,
        "image_channels": image_channels,
        "temporal_pair_input": temporal_pair_input,
        "compression": compression,
        "max_episodes": max_episodes,
        "source_manifest_sha256": _file_sha256(episodes_path),
        "test_assets_read": test_requested,
    }


def _iou(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    prediction_mask = prediction >= 0.5
    target_mask = target >= 0.5
    valid_mask = valid >= 0.5
    intersection = np.sum(prediction_mask & target_mask & valid_mask)
    union = np.sum((prediction_mask | target_mask) & valid_mask)
    return float(intersection / union) if union else 1.0


def _evidence_features(
    item: EvidenceItem,
    *,
    anchor_timestamp: str,
    max_temporal_distance: int,
    raster_shape: tuple[int, int],
    hypothesis_uncertainty: float,
    normalized_cost: float,
    observed_clear_fraction: float,
) -> list[float]:
    values = np.zeros(EVIDENCE_DIM, dtype=np.float32)
    values[0] = observed_clear_fraction
    values[1] = 1.0 - observed_clear_fraction
    delta = _month_index(item.timestamp) - _month_index(anchor_timestamp)
    denominator = max(max_temporal_distance, 1)
    values[2] = float(delta / denominator)
    values[3] = float(abs(delta) / denominator)
    scale_index = 0 if item.scale <= 1 else 1 if item.scale <= 2 else 2
    values[4 + scale_index] = 1.0
    width, height = raster_shape
    x_min, y_min, x_max, y_max = item.region
    values[7:11] = [
        (x_min + x_max) * 0.5 / max(width, 1),
        (y_min + y_max) * 0.5 / max(height, 1),
        (x_max - x_min) / max(width, 1),
        (y_max - y_min) / max(height, 1),
    ]
    values[11] = hypothesis_uncertainty
    values[12] = normalized_cost
    return values.tolist()


def _base_selector_sample(
    episode: EpisodeRecord,
    model: PriorConditionedUNet,
    device: torch.device,
    *,
    image_size: int,
    cost_weight: float,
    false_edit_weight: float,
    utility_mode: str = "proxy",
    utility_profile: str = "balanced",
    writeback_threshold: float = 0.5,
    writeback_delta_margin: float = 0.0,
    operation_selector: OperationSelector | None = None,
    operation_update_threshold: float = 0.5,
    initial_evidence_strategy: str = "updater_confidence",
    prior_input_translation_pixels: int = 0,
    prior_input_morphology: str = "none",
    prior_input_morphology_pixels: int = 0,
    corruption_seed: int = 0,
    input_cache: _CachedOracleInputs | None = None,
    candidate_workers: int = 1,
) -> SelectorSample:
    if utility_mode not in {"proxy", "executable"}:
        raise ValueError("utility_mode must be proxy or executable")
    if utility_profile not in UTILITY_PROFILES:
        raise ValueError(f"unknown utility profile: {utility_profile}")
    if initial_evidence_strategy not in {"updater_confidence", "min_cost"}:
        raise ValueError(f"unknown initial evidence strategy: {initial_evidence_strategy}")
    if candidate_workers < 1:
        raise ValueError("candidate_workers must be at least one")
    prior_geometry = _geometry(episode.prior_geometry)
    target_geometry = _geometry(episode.target_geometry)
    if input_cache is None:
        images: list[np.ndarray] | np.ndarray = []
        priors: list[np.ndarray] | np.ndarray = []
        targets: list[np.ndarray] | np.ndarray = []
        valid_masks: list[np.ndarray] | np.ndarray = []
        raster_shapes: list[tuple[int, int]] = []
        road_width_source_pixels = episode.metadata.get("road_width_source_pixels")

        def read_item(item: EvidenceItem) -> tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]
        ]:
            return _read_candidate(
                item,
                prior_geometry=prior_geometry,
                target_geometry=target_geometry,
                image_size=image_size,
                image_channels=model.config.image_channels,
                temporal_pair_input=model.config.temporal_pair_input,
                road_width_source_pixels=(
                    float(road_width_source_pixels)
                    if road_width_source_pixels is not None
                    else None
                ),
            )

        # Candidate reading and rasterization dominates real SN7 state construction.
        # ``map`` preserves manifest order, so this changes throughput, not the oracle.
        if candidate_workers == 1 or len(episode.evidence_catalog) == 1:
            candidate_inputs = map(read_item, episode.evidence_catalog)
        else:
            with ThreadPoolExecutor(max_workers=candidate_workers) as executor:
                candidate_inputs = list(executor.map(read_item, episode.evidence_catalog))
        for image, prior, target, valid, raster_shape in candidate_inputs:
            images.append(image)
            priors.append(prior)
            targets.append(target)
            valid_masks.append(valid)
            raster_shapes.append(raster_shape)
    else:
        images, priors, targets, valid_masks, raster_shapes = input_cache.read(episode)
    model_priors = []
    prior_input_shift = (0, 0)
    for prior in priors:
        corrupted, shift = deterministic_prior_translation(
            prior,
            identity=episode.episode_id,
            max_pixels=prior_input_translation_pixels,
            seed=corruption_seed,
        )
        corrupted = morph_prior_no_wrap(
            corrupted,
            operation=prior_input_morphology,
            pixels=prior_input_morphology_pixels,
        )
        model_priors.append(corrupted)
        if prior_input_shift != (0, 0) and prior_input_shift != shift:
            raise RuntimeError("episode prior-input translation changed across evidence")
        prior_input_shift = shift
    image_array = np.asarray(images, dtype=np.float32)
    prior_array = np.asarray(model_priors, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    valid_array = np.asarray(valid_masks, dtype=np.float32)
    image_tensor = torch.from_numpy(image_array).to(device)
    prior_tensor = torch.from_numpy(prior_array[:, None]).to(device)
    with torch.no_grad():
        outputs = model(image_tensor, prior_tensor)
        segmentation = torch.sigmoid(outputs["segmentation_logits"]).cpu().numpy()[:, 0]
        updater_probabilities = operation_probabilities(outputs, prior_tensor)
        if operation_selector is None:
            probabilities = updater_probabilities.cpu().numpy()
        else:
            spatial, context = operation_selector_inputs(
                outputs,
                prior_tensor,
                spatial_size=operation_selector.config.spatial_size,
            )
            probabilities = torch.softmax(operation_selector(spatial, context), dim=1).cpu().numpy()
        updater_confidence = torch.sigmoid(outputs["confidence_logits"]).cpu().numpy()
        geometry_deltas = outputs["geometry_delta"].cpu().numpy()

    target_index = EDIT_ORDER.index(episode.gt_edit.op)
    if operation_selector is None:
        predicted_indices = np.argmax(probabilities, axis=1)
    else:
        predicted_indices = np.where(
            1.0 - probabilities[:, 0] >= operation_update_threshold,
            np.argmax(probabilities[:, 1:], axis=1) + 1,
            0,
        )
    confidence = (
        updater_confidence
        if operation_selector is None
        else probabilities[np.arange(len(probabilities)), predicted_indices]
    )
    segmentation_ious = np.asarray(
        [
            _iou(prediction, target, valid)
            for prediction, target, valid in zip(
                segmentation, target_array, valid_array, strict=True
            )
        ]
    )
    prior_ious = np.asarray(
        [
            _iou(prior, target, valid)
            for prior, target, valid in zip(priors, target_array, valid_array, strict=True)
        ]
    )
    quality_after = 0.5 * segmentation_ious + 0.5 * (predicted_indices == target_index)
    stop_correct = float(episode.gt_edit.op == EditOperation.KEEP)
    quality_before = 0.5 * prior_ious + 0.5 * stop_correct
    # Deployment cannot condition candidate risk on the unknown ground-truth edit.
    false_edit_risks = 1.0 - np.max(probabilities, axis=1)
    observed_clear = np.asarray(
        [float(np.mean(valid >= 0.5)) for valid in valid_array], dtype=np.float32
    )
    costs = np.asarray(
        [
            item.cost + 0.25 * (1.0 - float(clear_fraction))
            for item, clear_fraction in zip(
                episode.evidence_catalog, observed_clear, strict=True
            )
        ],
        dtype=np.float32,
    )
    proxy_utilities = (
        quality_after
        - quality_before
        - cost_weight * costs
        - false_edit_weight * false_edit_risks
    )
    executable_outcomes: list[dict[str, Any]] = []
    profile = UTILITY_PROFILES[utility_profile]
    for index, (prior, target, valid) in enumerate(
        zip(priors, target_array, valid_array, strict=True)
    ):
        operation = EDIT_ORDER[int(predicted_indices[index])]
        committed = apply_typed_binary_edit(
            prior,
            segmentation[index],
            operation,
            threshold=writeback_threshold,
            delta_margin=writeback_delta_margin,
        )
        effective_operation = effective_operation_from_masks(prior, committed, valid)
        false_edit, missed_edit, wrong_edit = _operation_errors(
            episode.gt_edit.op, effective_operation
        )
        final_iou = _iou(committed, target, valid)
        quality_gain = final_iou - float(prior_ious[index])
        terminal_score = (
            quality_gain
            - profile.false_edit * float(false_edit)
            - profile.missed_edit * float(missed_edit)
            - profile.wrong_edit * float(wrong_edit)
        )
        executable_outcomes.append(
            {
                "predicted_operation": operation.value,
                "effective_operation": effective_operation.value,
                "prior_raster_iou": float(prior_ious[index]),
                "final_raster_iou": float(final_iou),
                "quality_gain": float(quality_gain),
                "false_edit": bool(false_edit),
                "missed_edit": bool(missed_edit),
                "wrong_edit": bool(wrong_edit),
                "terminal_score_before_cost": float(terminal_score),
            }
        )
    utilities = (
        np.asarray(
            [row["terminal_score_before_cost"] for row in executable_outcomes],
            dtype=np.float32,
        )
        if utility_mode == "executable"
        else proxy_utilities
    )
    if initial_evidence_strategy == "updater_confidence":
        initial_index = min(
            range(len(costs)),
            key=lambda index: (float(costs[index]), -float(confidence[index]), index),
        )
    else:
        # Keep the direct observation fixed when comparing replacement updaters.
        initial_index = min(range(len(costs)), key=lambda index: (float(costs[index]), index))
    initial_probability = probabilities[initial_index]
    entropy = float(
        -np.sum(initial_probability * np.log(np.clip(initial_probability, 1e-7, 1.0)))
        / np.log(len(EDIT_ORDER))
    )
    initial_edit = EDIT_ORDER[int(predicted_indices[initial_index])]
    hypothesis = np.zeros(HYPOTHESIS_DIM, dtype=np.float32)
    hypothesis[:4] = initial_probability
    if prior_geometry is not None:
        hypothesis[4:12] = geometry_features(prior_geometry)
    hypothesis[12] = float(np.mean(predicted_indices != predicted_indices[initial_index]))
    hypothesis[13] = float(confidence[initial_index])
    hypothesis[14] = float(np.mean([np.mean(prior >= 0.5) for prior in priors]))
    hypothesis[15] = float(min(len(costs) / 32.0, 1.0))
    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[0] = 1.0
    state[2:6] = initial_probability
    state[7] = float(confidence[initial_index])
    anchor_timestamp = episode.anchor_timestamp or episode.evidence_catalog[-1].timestamp
    temporal_distances = [
        abs(_month_index(item.timestamp) - _month_index(anchor_timestamp))
        for item in episode.evidence_catalog
    ]
    evidence_features = [
        _evidence_features(
            item,
            anchor_timestamp=anchor_timestamp,
            max_temporal_distance=max(temporal_distances, default=1),
            raster_shape=raster_shape,
            hypothesis_uncertainty=entropy,
            normalized_cost=float(cost / max(float(np.max(costs)), 1e-6)),
            observed_clear_fraction=float(clear_fraction),
        )
        for item, raster_shape, clear_fraction, cost in zip(
            episode.evidence_catalog,
            raster_shapes,
            observed_clear,
            costs,
            strict=True,
        )
    ]
    return SelectorSample(
        sample_id=episode.episode_id,
        split=episode.split,
        edit_type=initial_edit,
        hypothesis_features=hypothesis.tolist(),
        state_features=state.tolist(),
        evidence_ids=[item.evidence_id for item in episode.evidence_catalog],
        evidence_features=evidence_features,
        evidence_costs=costs.tolist(),
        false_edit_risks=false_edit_risks.tolist(),
        oracle_utilities=utilities.tolist(),
        stop_utility=0.0,
        false_edit_penalty_weight=false_edit_weight,
        metadata={
            "source_episode": episode.episode_id,
            "aoi_id": episode.aoi_id,
            "gt_edit": episode.gt_edit.op.value,
            "initial_evidence_id": episode.evidence_catalog[initial_index].evidence_id,
            "cost_weight": cost_weight,
            "quality": (
                "executable_raster_iou_gain"
                if utility_mode == "executable"
                else "0.5*raster_iou+0.5*edit_correct"
            ),
            "utility_mode": utility_mode,
            "utility_profile": utility_profile,
            "writeback_threshold": writeback_threshold,
            "writeback_delta_margin": writeback_delta_margin,
            "image_size": image_size,
            "executable_outcomes": {
                item.evidence_id: executable_outcomes[index]
                for index, item in enumerate(episode.evidence_catalog)
            },
            "operation_selector_enabled": operation_selector is not None,
            "operation_update_threshold": operation_update_threshold,
            "initial_evidence_strategy": initial_evidence_strategy,
            "prior_input_corruption": {
                "translation_pixels": prior_input_translation_pixels,
                "morphology": prior_input_morphology,
                "morphology_pixels": prior_input_morphology_pixels,
                "corruption_seed": corruption_seed,
                "shift_y": prior_input_shift[0],
                "shift_x": prior_input_shift[1],
                "scope": "model_input_only",
            },
            "evidence_predictions": {
                item.evidence_id: {
                    "edit_probabilities": probabilities[index].tolist(),
                    "gated_edit": EDIT_ORDER[int(predicted_indices[index])].value,
                    "confidence": float(confidence[index]),
                    "geometry_delta": geometry_deltas[index].tolist(),
                }
                for index, item in enumerate(episode.evidence_catalog)
            },
        },
    )


def _fused_prediction(
    base: SelectorSample, selected: list[int]
) -> tuple[np.ndarray, float, np.ndarray, float]:
    predictions = base.metadata.get("evidence_predictions", {})
    rows = [predictions[base.evidence_ids[index]] for index in selected]
    probabilities = np.asarray([row["edit_probabilities"] for row in rows], dtype=np.float32)
    confidences = np.asarray([row["confidence"] for row in rows], dtype=np.float32)
    geometry = np.asarray([row["geometry_delta"] for row in rows], dtype=np.float32)
    weights = np.clip(confidences, 0.05, 1.0)
    fused_probability = np.average(probabilities, axis=0, weights=weights)
    fused_probability /= max(float(fused_probability.sum()), 1e-7)
    fused_geometry = np.average(geometry, axis=0, weights=weights)
    entropy = float(
        -np.sum(fused_probability * np.log(np.clip(fused_probability, 1e-7, 1.0)))
        / np.log(len(EDIT_ORDER))
    )
    agreement = 1.0 - float(
        np.mean(np.argmax(probabilities, axis=1) != int(np.argmax(fused_probability)))
    )
    fused_confidence = float(
        np.clip(
            0.5 * np.average(confidences, weights=weights) + 0.5 * agreement,
            0.0,
            1.0,
        )
    )
    return fused_probability, fused_confidence, fused_geometry, entropy


def _expand_budget_states(
    base: SelectorSample,
    *,
    budgets: tuple[float, ...],
    cost_weight: float,
    max_steps: int | None,
) -> list[SelectorSample]:
    costs = np.asarray(base.evidence_costs, dtype=np.float32)
    risks = np.asarray(base.false_edit_risks, dtype=np.float32)
    initial_utilities = np.asarray(base.oracle_utilities, dtype=np.float32)
    utility_mode = str(base.metadata.get("utility_mode", "proxy"))
    if utility_mode == "executable":
        profile_name = str(base.metadata.get("utility_profile", "balanced"))
        profile = UTILITY_PROFILES[profile_name]
        terminal_scores = initial_utilities
    else:
        penalties = cost_weight * costs + base.false_edit_penalty_weight * risks
        quality_gains = initial_utilities + penalties
    expanded: list[SelectorSample] = []
    for budget in budgets:
        if budget <= 0:
            raise ValueError("selector budgets must be positive")
        initial_id = str(base.metadata["initial_evidence_id"])
        initial_index = base.evidence_ids.index(initial_id)
        available = [index for index in range(len(costs)) if index != initial_index]
        selected: list[int] = [initial_index]
        remaining = float(budget)
        current_gain = (
            float(terminal_scores[initial_index])
            if utility_mode == "executable"
            else max(float(quality_gains[initial_index]), 0.0)
        )
        step = 0
        while available and (max_steps is None or step < max_steps):
            affordable = [index for index in available if float(costs[index]) <= remaining]
            if not affordable:
                break
            if utility_mode == "executable":
                utilities = np.asarray(
                    [
                        float(terminal_scores[index])
                        - current_gain
                        - profile.cost * min(float(costs[index]) / budget, 1.0)
                        for index in affordable
                    ],
                    dtype=np.float32,
                )
            else:
                utilities = np.asarray(
                    [
                        max(float(quality_gains[index]) - current_gain, 0.0)
                        - float(penalties[index])
                        for index in affordable
                    ],
                    dtype=np.float32,
                )
            state = list(base.state_features)
            state[0] = float(np.clip(remaining / budget, 0.0, 1.0))
            state[1] = float(np.clip((budget - remaining) / budget, 0.0, 1.0))
            state[6] = float(len(selected) / max(len(costs), 1))
            hypothesis = list(base.hypothesis_features)
            fused_probability, fused_confidence, _, entropy = _fused_prediction(
                base, selected
            )
            hypothesis[:4] = fused_probability.tolist()
            hypothesis[12] = entropy
            hypothesis[13] = fused_confidence
            state[2:6] = fused_probability.tolist()
            state[7] = current_gain
            suffix = str(budget).replace(".", "p")
            metadata = dict(base.metadata)
            metadata.update(
                {
                    "budget": budget,
                    "oracle_step": step,
                    "selected_evidence_ids": [base.evidence_ids[index] for index in selected],
                    "initial_evidence_cost_excluded": True,
                }
            )
            expanded.append(
                base.model_copy(
                    update={
                        "sample_id": f"{base.sample_id}__b{suffix}__s{step}",
                        "hypothesis_features": hypothesis,
                        "state_features": state,
                        "evidence_ids": [base.evidence_ids[index] for index in affordable],
                        "evidence_features": [
                            base.evidence_features[index] for index in affordable
                        ],
                        "evidence_costs": [base.evidence_costs[index] for index in affordable],
                        "false_edit_risks": [
                            base.false_edit_risks[index] for index in affordable
                        ],
                        "oracle_utilities": utilities.tolist(),
                        "metadata": metadata,
                    }
                )
            )
            best_local = int(np.argmax(utilities))
            if float(utilities[best_local]) <= base.stop_utility:
                break
            selected_index = affordable[best_local]
            selected.append(selected_index)
            available.remove(selected_index)
            remaining -= float(costs[selected_index])
            current_gain = (
                max(current_gain, float(terminal_scores[selected_index]))
                if utility_mode == "executable"
                else max(current_gain, float(quality_gains[selected_index]))
            )
            step += 1
    return expanded


def build_selector_oracle_samples(
    checkpoint_path: Path,
    episodes_path: Path,
    output: Path,
    *,
    device: str = "auto",
    image_size: int = 128,
    cost_weight: float = 0.18,
    false_edit_weight: float = 0.35,
    utility_mode: str = "proxy",
    utility_profile: str = "balanced",
    writeback_threshold: float = 0.5,
    writeback_delta_margin: float = 0.0,
    asset_root_maps: tuple[tuple[Path, Path], ...] = (),
    budgets: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
    max_steps: int | None = None,
    operation_selector_checkpoint: Path | None = None,
    operation_update_threshold: float = 0.5,
    initial_evidence_strategy: str = "updater_confidence",
    prior_input_translation_pixels: int = 0,
    prior_input_morphology: str = "none",
    prior_input_morphology_pixels: int = 0,
    corruption_seed: int = 0,
    splits: tuple[str, ...] = ("train", "val"),
    frozen_test: bool = False,
    input_cache: Path | None = None,
    max_episodes: int | None = None,
    candidate_workers: int = 1,
) -> dict[str, Any]:
    if prior_input_translation_pixels < 0:
        raise ValueError("prior_input_translation_pixels must be non-negative")
    if candidate_workers < 1:
        raise ValueError("candidate_workers must be at least one")
    morph_prior_no_wrap(
        np.zeros((1, 1), dtype=np.float32),
        operation=prior_input_morphology,
        pixels=prior_input_morphology_pixels,
    )
    requested_splits = set(splits)
    if not requested_splits or not requested_splits <= {"train", "val", "test"}:
        raise ValueError("selector oracle splits must be train, val, or test")
    test_requested = "test" in requested_splits
    if test_requested:
        if requested_splits != {"test"}:
            raise ValueError("frozen selector-oracle construction must be test-only")
        if not frozen_test:
            raise PermissionError("test selector-oracle construction requires --frozen-test")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
        if output.exists() or output.with_suffix(".summary.json").exists():
            raise FileExistsError(f"refusing to overwrite frozen test states: {output}")
    elif frozen_test:
        raise ValueError("--frozen-test is valid only for the test split")
    target_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    model = PriorConditionedUNet(UpdaterConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(target_device).eval()
    operation_selector = None
    if operation_selector_checkpoint is not None:
        selector_checkpoint = torch.load(
            operation_selector_checkpoint,
            map_location=target_device,
            weights_only=False,
        )
        operation_selector = OperationSelector(
            OperationSelectorConfig(**selector_checkpoint["model_config"])
        )
        operation_selector.load_state_dict(selector_checkpoint["state_dict"])
        operation_selector.to(target_device).eval()
    episodes = _selected_episodes(
        episodes_path,
        asset_root_maps=asset_root_maps,
        splits=splits,
    )
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise ValueError(f"no episodes found for splits={sorted(requested_splits)}")
    progress_path = output.with_suffix(".progress.json")
    base_samples = []
    cache = (
        _CachedOracleInputs.open(
            input_cache,
            episodes_path=episodes_path,
            episodes=episodes,
            image_size=image_size,
            image_channels=model.config.image_channels,
            temporal_pair_input=model.config.temporal_pair_input,
        )
        if input_cache is not None
        else None
    )
    try:
        for index, episode in enumerate(tqdm(episodes, desc="Counterfactual episodes"), start=1):
            base_samples.append(
                _base_selector_sample(
                    episode,
                    model,
                    target_device,
                    image_size=image_size,
                    cost_weight=cost_weight,
                    false_edit_weight=false_edit_weight,
                    utility_mode=utility_mode,
                    utility_profile=utility_profile,
                    writeback_threshold=writeback_threshold,
                    writeback_delta_margin=writeback_delta_margin,
                    operation_selector=operation_selector,
                    operation_update_threshold=operation_update_threshold,
                    initial_evidence_strategy=initial_evidence_strategy,
                    prior_input_translation_pixels=prior_input_translation_pixels,
                    prior_input_morphology=prior_input_morphology,
                    prior_input_morphology_pixels=prior_input_morphology_pixels,
                    corruption_seed=corruption_seed,
                    input_cache=cache,
                    candidate_workers=candidate_workers,
                )
            )
            if index == 1 or index % 25 == 0 or index == len(episodes):
                progress_path.write_text(
                    json.dumps(
                        {
                            "status": "running" if index < len(episodes) else "inference_complete",
                            "episodes_processed": index,
                            "episodes_total": len(episodes),
                            "operation_selector_enabled": operation_selector is not None,
                            "utility_mode": utility_mode,
                            "input_cache": str(input_cache) if input_cache is not None else None,
                            "test_assets_read": test_requested,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
    finally:
        if cache is not None:
            cache.close()
    samples = [
        state
        for base in base_samples
        for state in _expand_budget_states(
            base,
            budgets=budgets,
            cost_weight=cost_weight,
            max_steps=max_steps,
        )
    ]
    write_selector_samples(samples, output)
    summary = {
        "episodes": len(episodes),
        "samples": len(samples),
        "checkpoint": str(checkpoint_path.resolve()),
        "output": str(output.resolve()),
        "device": str(target_device),
        "image_size": image_size,
        "cost_weight": cost_weight,
        "false_edit_weight": false_edit_weight,
        "utility_mode": utility_mode,
        "utility_profile": utility_profile,
        "writeback_threshold": writeback_threshold,
        "writeback_delta_margin": writeback_delta_margin,
        "asset_root_maps": [
            {"source": str(source), "target": str(target)}
            for source, target in asset_root_maps
        ],
        "budgets": list(budgets),
        "max_steps": max_steps,
        "operation_selector_checkpoint": (
            str(operation_selector_checkpoint.resolve())
            if operation_selector_checkpoint is not None
            else None
        ),
        "operation_update_threshold": operation_update_threshold,
        "initial_evidence_strategy": initial_evidence_strategy,
        "prior_input_translation_pixels": prior_input_translation_pixels,
        "prior_input_morphology": prior_input_morphology,
        "prior_input_morphology_pixels": prior_input_morphology_pixels,
        "corruption_seed": corruption_seed,
        "prior_input_corruption_scope": "model_input_only",
        "input_cache": str(input_cache.resolve()) if input_cache is not None else None,
        "max_episodes": max_episodes,
        "splits": sorted(requested_splits),
        "test_assets_read": test_requested,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    progress_path.write_text(
        json.dumps({"status": "complete", **summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
