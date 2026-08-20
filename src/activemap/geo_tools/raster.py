"""Raster inspection, processing, segmentation, and terrain tools."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy import ndimage

from activemap.geo_tools.records import GeoToolCall, GeoToolName, GeoToolResult


def _window(parameters: dict[str, Any]) -> Window | None:
    values = parameters.get("pixel_window")
    if values is None:
        return None
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("pixel_window must be [column, row, width, height]")
    column, row, width, height = (int(value) for value in values)
    if width <= 0 or height <= 0:
        raise ValueError("pixel_window width and height must be positive")
    return Window(column, row, width, height)


def _bands(parameters: dict[str, Any], count: int) -> list[int]:
    values = parameters.get("bands")
    bands = list(range(1, count + 1)) if values is None else [int(value) for value in values]
    if not bands or any(value < 1 or value > count for value in bands):
        raise ValueError(f"bands must be within 1..{count}")
    return bands


def _read_uncached(
    path: Path,
    parameters: dict[str, Any],
    *,
    force_bands: list[int] | None = None,
    force_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        if _window(parameters) is not None:
            raise ValueError("pixel_window is unsupported for aligned NPY patches")
        array = np.asarray(np.load(path), dtype=np.float32)
        if array.ndim != 3:
            raise ValueError("NPY raster must have three dimensions")
        if array.shape[0] in {1, 3, 4}:
            data = array
        elif array.shape[-1] in {1, 3, 4}:
            data = np.moveaxis(array, -1, 0)
        else:
            raise ValueError(f"cannot infer NPY raster channel axis: {array.shape}")
        bands = force_bands or _bands(parameters, data.shape[0])
        data = data[np.asarray(bands) - 1]
        out_size = parameters.get("out_size")
        if force_shape is not None:
            height, width = force_shape
        elif out_size is not None:
            if not isinstance(out_size, list) or len(out_size) != 2:
                raise ValueError("out_size must be [height, width]")
            height, width = (int(value) for value in out_size)
        else:
            height, width = data.shape[-2:]
        if height <= 0 or width <= 0:
            raise ValueError("output height and width must be positive")
        if data.shape[-2:] != (height, width):
            scale = (1.0, height / data.shape[-2], width / data.shape[-1])
            data = ndimage.zoom(data, scale, order=1, prefilter=False)
        valid = np.all(np.isfinite(data), axis=0)
        return data, valid, {
            "bands": bands,
            "shape": list(data.shape),
            "crs": None,
            "transform": None,
            "source_dtype": str(array.dtype),
            "aligned_patch": True,
        }
    with rasterio.open(path) as dataset:
        bands = force_bands or _bands(parameters, dataset.count)
        window = _window(parameters)
        out_size = parameters.get("out_size")
        if force_shape is not None:
            height, width = force_shape
        elif out_size is not None:
            if not isinstance(out_size, list) or len(out_size) != 2:
                raise ValueError("out_size must be [height, width]")
            height, width = (int(value) for value in out_size)
        elif window is not None:
            height, width = int(window.height), int(window.width)
        else:
            height, width = dataset.height, dataset.width
        if height <= 0 or width <= 0:
            raise ValueError("output height and width must be positive")
        masked = dataset.read(
            bands,
            window=window,
            out_shape=(len(bands), height, width),
            boundless=window is not None,
            masked=True,
            resampling=Resampling.bilinear,
        )
        data = np.asarray(masked.astype(np.float32).filled(np.nan), dtype=np.float32)
        valid = ~np.any(np.ma.getmaskarray(masked), axis=0) & np.all(np.isfinite(data), axis=0)
        metadata = {
            "bands": bands,
            "shape": list(data.shape),
            "crs": str(dataset.crs) if dataset.crs else None,
            "transform": list(dataset.transform)[:6],
            "source_dtype": dataset.dtypes[0],
        }
    return data, valid, metadata


@lru_cache(maxsize=64)
def _cached_read(
    path: str,
    parameters_json: str,
    force_bands: tuple[int, ...] | None,
    force_shape: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    return _read_uncached(
        Path(path),
        json.loads(parameters_json),
        force_bands=list(force_bands) if force_bands is not None else None,
        force_shape=force_shape,
    )


def _read(
    path: Path,
    parameters: dict[str, Any],
    *,
    force_bands: list[int] | None = None,
    force_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read a bounded cached raster window and isolate mutable caller state."""

    data, valid, metadata = _cached_read(
        str(path.resolve()),
        json.dumps(parameters, sort_keys=True, separators=(",", ":")),
        tuple(force_bands) if force_bands is not None else None,
        force_shape,
    )
    return data.copy(), valid.copy(), copy.deepcopy(metadata)


def _stats(values: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    selected = values[..., valid]
    finite = selected[np.isfinite(selected)]
    if finite.size == 0:
        raise ValueError("raster selection has no valid finite pixels")
    return {
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
    }


class RasterCropTool:
    name = GeoToolName.RASTER_CROP
    cost = 0.05

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def run(self, call: GeoToolCall) -> GeoToolResult:
        data, valid, metadata = _read(Path(call.inputs["image_path"]), call.parameters)
        self.output_root.mkdir(parents=True, exist_ok=True)
        output = self.output_root / f"{call.call_id}.npy"
        np.save(output, data)
        return GeoToolResult(
            call_id=call.call_id,
            tool=self.name,
            success=True,
            outputs={
                **metadata,
                "valid_fraction": float(np.mean(valid)),
                "stats": _stats(data, valid),
            },
            artifacts=[str(output.resolve())],
            cost=self.cost,
        )


class ImageQualityTool:
    name = GeoToolName.IMAGE_QUALITY
    cost = 0.03

    def run(self, call: GeoToolCall) -> GeoToolResult:
        data, valid, metadata = _read(Path(call.inputs["image_path"]), call.parameters)
        finite = data[:, valid]
        if finite.size == 0:
            raise ValueError("raster selection has no valid finite pixels")
        low = np.percentile(finite, 1, axis=1)[:, None, None]
        high = np.percentile(finite, 99, axis=1)[:, None, None]
        scale = np.maximum(high - low, 1e-6)
        normalized = np.clip((data - low) / scale, 0.0, 1.0)
        gray = np.nanmean(normalized, axis=0)
        gradient_y, gradient_x = np.gradient(np.nan_to_num(gray))
        sharpness = np.mean(np.square(gradient_x[valid]) + np.square(gradient_y[valid]))
        saturation = np.any((normalized <= 0.005) | (normalized >= 0.995), axis=0)
        return GeoToolResult(
            call_id=call.call_id,
            tool=self.name,
            success=True,
            outputs={
                **metadata,
                "valid_fraction": float(np.mean(valid)),
                "saturation_fraction": float(np.mean(saturation[valid])),
                "sharpness": float(sharpness),
                "stats": _stats(data, valid),
            },
            cost=self.cost,
        )


class SpectralIndexTool:
    name = GeoToolName.SPECTRAL_INDEX
    cost = 0.10
    PRESETS = {
        "NDVI": ("nir", "red"),
        "NDWI": ("green", "nir"),
        "NDBI": ("swir", "nir"),
    }

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def run(self, call: GeoToolCall) -> GeoToolResult:
        preset = str(call.parameters.get("index", "CUSTOM")).upper()
        band_map = {
            str(key).lower(): int(value)
            for key, value in call.parameters.get("band_map", {}).items()
        }
        if preset in self.PRESETS:
            positive_name, negative_name = self.PRESETS[preset]
            missing = [name for name in (positive_name, negative_name) if name not in band_map]
            if missing:
                raise ValueError(f"{preset} requires band_map entries: {missing}")
            positive_band, negative_band = band_map[positive_name], band_map[negative_name]
        else:
            positive_band = int(call.parameters["positive_band"])
            negative_band = int(call.parameters["negative_band"])
        data, valid, metadata = _read(
            Path(call.inputs["image_path"]),
            call.parameters,
            force_bands=[positive_band, negative_band],
        )
        denominator = data[0] + data[1]
        index = np.divide(
            data[0] - data[1],
            denominator,
            out=np.full_like(denominator, np.nan),
            where=np.abs(denominator) > 1e-6,
        )
        valid &= np.isfinite(index)
        self.output_root.mkdir(parents=True, exist_ok=True)
        output = self.output_root / f"{call.call_id}_{preset.lower()}.npy"
        np.save(output, index)
        return GeoToolResult(
            call_id=call.call_id,
            tool=self.name,
            success=True,
            outputs={
                **metadata,
                "index": preset,
                "valid_fraction": float(np.mean(valid)),
                "stats": _stats(index, valid),
            },
            artifacts=[str(output.resolve())],
            cost=self.cost,
        )


def _robust_unit(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.empty_like(data, dtype=np.float32)
    for index, band in enumerate(data):
        values = band[valid]
        low, high = np.percentile(values, [2, 98])
        output[index] = np.clip((band - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    return output


class TemporalChangeTool:
    name = GeoToolName.TEMPORAL_CHANGE
    cost = 0.15

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def run(self, call: GeoToolCall) -> GeoToolResult:
        before, before_valid, metadata = _read(
            Path(call.inputs["before_path"]), call.parameters
        )
        after, after_valid, _ = _read(
            Path(call.inputs["after_path"]),
            call.parameters,
            force_bands=list(metadata["bands"]),
            force_shape=before.shape[-2:],
        )
        valid = before_valid & after_valid
        score = np.mean(
            np.abs(_robust_unit(after, valid) - _robust_unit(before, valid)), axis=0
        )
        threshold = float(call.parameters.get("threshold", 0.20))
        self.output_root.mkdir(parents=True, exist_ok=True)
        output = self.output_root / f"{call.call_id}_change.npy"
        np.save(output, score)
        return GeoToolResult(
            call_id=call.call_id,
            tool=self.name,
            success=True,
            outputs={
                **metadata,
                "threshold": threshold,
                "changed_fraction": float(np.mean(score[valid] >= threshold)),
                "stats": _stats(score, valid),
            },
            artifacts=[str(output.resolve())],
            cost=self.cost,
        )


class RasterSegmentTool:
    name = GeoToolName.RASTER_SEGMENT
    cost = 0.12

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def run(self, call: GeoToolCall) -> GeoToolResult:
        band = int(call.parameters.get("band", 1))
        data, valid, metadata = _read(
            Path(call.inputs["image_path"]), call.parameters, force_bands=[band]
        )
        values = data[0]
        threshold = call.parameters.get("threshold")
        if threshold is None:
            percentile = float(call.parameters.get("percentile", 75.0))
            threshold = float(np.percentile(values[valid], percentile))
        direction = str(call.parameters.get("direction", "above"))
        mask = values >= float(threshold) if direction == "above" else values <= float(threshold)
        mask &= valid
        iterations = int(call.parameters.get("morphology_iterations", 1))
        if iterations > 0:
            mask = ndimage.binary_opening(mask, iterations=iterations)
            mask = ndimage.binary_closing(mask, iterations=iterations)
        labels, component_count = ndimage.label(mask)
        areas = np.bincount(labels.ravel())[1:]
        minimum_area = int(call.parameters.get("minimum_area", 4))
        kept = np.flatnonzero(areas >= minimum_area) + 1
        filtered = np.isin(labels, kept)
        self.output_root.mkdir(parents=True, exist_ok=True)
        output = self.output_root / f"{call.call_id}_segment.npy"
        np.save(output, filtered.astype(np.uint8))
        return GeoToolResult(
            call_id=call.call_id,
            tool=self.name,
            success=True,
            outputs={
                **metadata,
                "threshold": float(threshold),
                "direction": direction,
                "raw_component_count": int(component_count),
                "component_count": int(len(kept)),
                "foreground_fraction": float(np.mean(filtered[valid])),
            },
            artifacts=[str(output.resolve())],
            cost=self.cost,
        )


class TerrainAnalysisTool:
    name = GeoToolName.TERRAIN_ANALYSIS
    cost = 0.15

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def run(self, call: GeoToolCall) -> GeoToolResult:
        data, valid, metadata = _read(
            Path(call.inputs["dem_path"]), call.parameters, force_bands=[1]
        )
        elevation = data[0]
        transform = metadata["transform"]
        x_resolution = abs(float(transform[0]))
        y_resolution = abs(float(transform[4]))
        gradient_y, gradient_x = np.gradient(
            np.nan_to_num(elevation, nan=float(np.nanmedian(elevation[valid]))),
            y_resolution,
            x_resolution,
        )
        slope = np.degrees(np.arctan(np.hypot(gradient_x, gradient_y)))
        aspect = (np.degrees(np.arctan2(-gradient_x, gradient_y)) + 360.0) % 360.0
        azimuth = np.radians(float(call.parameters.get("sun_azimuth", 315.0)))
        altitude = np.radians(float(call.parameters.get("sun_altitude", 45.0)))
        slope_radians = np.radians(slope)
        aspect_radians = np.radians(aspect)
        hillshade = 255.0 * np.clip(
            np.sin(altitude) * np.cos(slope_radians)
            + np.cos(altitude) * np.sin(slope_radians) * np.cos(azimuth - aspect_radians),
            0.0,
            1.0,
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        output = self.output_root / f"{call.call_id}_terrain.npz"
        np.savez_compressed(output, slope=slope, aspect=aspect, hillshade=hillshade)
        return GeoToolResult(
            call_id=call.call_id,
            tool=self.name,
            success=True,
            outputs={
                **metadata,
                "slope_degrees": _stats(slope, valid),
                "aspect_degrees": _stats(aspect, valid),
                "hillshade": _stats(hillshade, valid),
            },
            artifacts=[str(output.resolve())],
            cost=self.cost,
        )
