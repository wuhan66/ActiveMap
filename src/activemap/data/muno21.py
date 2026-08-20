"""MUNO21 road-graph scenarios converted to typed ActiveMap updater samples."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine
from PIL import Image, ImageDraw
from shapely.geometry import LineString, MultiLineString, Point, box, mapping
from shapely.geometry.base import BaseGeometry

from activemap.data.progress import write_progress
from activemap.models import (
    CandidateHypothesis,
    EditOperation,
    EditRecord,
    EpisodeRecord,
    EvidenceItem,
    GeoJSONGeometry,
)
from activemap.updater_records import UpdaterSample


@dataclass(frozen=True)
class MunoGraph:
    vertices: np.ndarray
    edges: tuple[tuple[int, int], ...]

    def lines(self, bounds: tuple[int, int, int, int] | None = None) -> list[LineString]:
        seen: set[tuple[int, int]] = set()
        output = []
        for source, destination in self.edges:
            key = (min(source, destination), max(source, destination))
            if source == destination or key in seen:
                continue
            seen.add(key)
            source_coordinate = self.vertices[source]
            destination_coordinate = self.vertices[destination]
            if bounds is not None:
                left, top, right, bottom = bounds
                if (
                    max(source_coordinate[0], destination_coordinate[0]) < left
                    or min(source_coordinate[0], destination_coordinate[0]) > right
                    or max(source_coordinate[1], destination_coordinate[1]) < top
                    or min(source_coordinate[1], destination_coordinate[1]) > bottom
                ):
                    continue
            output.append(LineString([source_coordinate.tolist(), destination_coordinate.tolist()]))
        return output


def read_muno_graph(path: Path) -> MunoGraph:
    vertices: list[tuple[float, float]] = []
    edges: list[tuple[int, int]] = []
    reading_vertices = True
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            reading_vertices = False
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"invalid MUNO21 graph line {line_number}: {raw_line!r}")
        if reading_vertices:
            vertices.append((float(parts[0]), float(parts[1])))
        else:
            source, destination = int(parts[0]), int(parts[1])
            if source >= len(vertices) or destination >= len(vertices):
                raise ValueError(f"MUNO21 edge references missing vertex at line {line_number}")
            edges.append((source, destination))
    if not vertices:
        raise ValueError(f"MUNO21 graph has no vertices: {path}")
    return MunoGraph(np.asarray(vertices, dtype=np.float64), tuple(edges))


def muno_tags_to_edit(tags: list[str]) -> EditOperation:
    values = {value.lower() for value in tags}
    has_add = bool(values & {"constructed", "was_missing"})
    has_delete = bool(values & {"bulldozed", "deconstructed", "was_incorrect"})
    if has_add and has_delete:
        return EditOperation.RESHAPE
    if has_add:
        return EditOperation.ADD
    if has_delete:
        return EditOperation.DELETE
    return EditOperation.KEEP


def _point(payload: dict[str, Any]) -> tuple[float, float]:
    x_value = payload["X"] if "X" in payload else payload.get("x")
    y_value = payload["Y"] if "Y" in payload else payload.get("y")
    if x_value is None or y_value is None:
        raise ValueError(f"invalid MUNO21 point: {payload}")
    return float(x_value), float(y_value)


def _change_lines(annotation: dict[str, Any], *, deleted: bool) -> list[LineString]:
    output = []
    cluster = annotation.get("Cluster") or {}
    for change in cluster.get("Changes") or []:
        if not isinstance(change, dict):
            continue
        if bool(change.get("Deleted", False)) != deleted:
            continue
        for segment in change.get("Segments") or []:
            if not isinstance(segment, dict):
                continue
            start = segment.get("Start", segment.get("start"))
            end = segment.get("End", segment.get("end"))
            if start is None or end is None:
                continue
            line = LineString([_point(start), _point(end)])
            if not line.is_empty and line.length > 0:
                output.append(line)
    return output


def _change_groups(annotation: dict[str, Any]) -> list[BaseGeometry]:
    groups: list[BaseGeometry] = []
    cluster = annotation.get("Cluster") or {}
    for change in cluster.get("Changes") or []:
        if not isinstance(change, dict):
            continue
        lines: list[LineString] = []
        for segment in change.get("Segments") or []:
            if not isinstance(segment, dict):
                continue
            start = segment.get("Start", segment.get("start"))
            end = segment.get("End", segment.get("end"))
            if start is None or end is None:
                continue
            line = LineString([_point(start), _point(end)])
            if not line.is_empty and line.length > 0:
                lines.append(line)
        groups.extend(lines)
    return groups


def _clip_lines(lines: list[LineString], bounds: tuple[int, int, int, int]) -> list[LineString]:
    clipping = box(*bounds)
    output: list[LineString] = []
    for line in lines:
        clipped = line.intersection(clipping)
        if clipped.is_empty:
            continue
        if clipped.geom_type == "LineString":
            output.append(clipped)
        elif clipped.geom_type == "MultiLineString":
            output.extend(list(clipped.geoms))
        elif clipped.geom_type == "GeometryCollection":
            output.extend(part for part in clipped.geoms if part.geom_type == "LineString")
    return output


def _multiline(lines: list[LineString]) -> MultiLineString | None:
    return MultiLineString([line.coords for line in lines]) if lines else None


def _normalized_bounds(
    geometry: BaseGeometry | None, bounds: tuple[int, int, int, int]
) -> np.ndarray:
    if geometry is None or geometry.is_empty:
        return np.zeros(4, dtype=np.float32)
    left, top, right, bottom = bounds
    min_x, min_y, max_x, max_y = geometry.bounds
    return np.asarray(
        [
            (min_x - left) / max(right - left, 1),
            (min_y - top) / max(bottom - top, 1),
            (max_x - left) / max(right - left, 1),
            (max_y - top) / max(bottom - top, 1),
        ],
        dtype=np.float32,
    )


def rasterize_muno_road_mask(
    geometry: BaseGeometry | None,
    transform: Affine,
    image_size: int,
    road_width_pixels: float,
) -> np.ndarray:
    output = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(output)
    inverse = ~transform
    source_per_output_pixel = (abs(transform.a) + abs(transform.e)) / 2.0
    width = max(1, round(road_width_pixels / max(source_per_output_pixel, 1e-6)))
    lines = []
    if geometry is not None and not geometry.is_empty:
        if geometry.geom_type == "LineString":
            lines = [geometry]
        elif geometry.geom_type == "MultiLineString":
            lines = list(geometry.geoms)
    for line in lines:
        pixels = [inverse * (x, y) for x, y in line.coords]
        if len(pixels) >= 2:
            draw.line(pixels, fill=255, width=width, joint="curve")
    return np.asarray(output, dtype=np.float32) / 255.0


def _latest_image(image_dir: Path, label: str) -> Path:
    candidates = list(image_dir.glob(f"{label}_*.jpg"))
    if not candidates:
        raise FileNotFoundError(f"no MUNO21 NAIP image found for {label} in {image_dir}")

    def key(path: Path) -> tuple[int, str]:
        years = [int(value) for value in re.findall(r"(?:19|20)\d{2}", path.stem)]
        return (max(years) if years else -1, path.name)

    return max(candidates, key=key)


def _validation_regions(train_regions: set[str], seed: int) -> set[str]:
    count = max(1, round(len(train_regions) * 0.2))
    ranked = sorted(
        train_regions,
        key=lambda region: hashlib.sha1(f"{seed}:{region}".encode()).hexdigest(),
    )
    return set(ranked[:count])


def _scenario_crop_windows(
    annotation: dict[str, Any],
    *,
    padding: int,
    max_source_crop_size: int,
) -> list[list[int]]:
    window = [int(value) for value in annotation["Cluster"]["Window"]]
    padded_width = window[2] - window[0] + 2 * padding
    padded_height = window[3] - window[1] + 2 * padding
    if padded_width <= max_source_crop_size and padded_height <= max_source_crop_size:
        return [
            [
                window[0] - padding,
                window[1] - padding,
                window[2] + padding,
                window[3] + padding,
            ]
        ]

    groups = _change_groups(annotation)
    if not groups:
        center_x = (window[0] + window[2]) / 2.0
        center_y = (window[1] + window[3]) / 2.0
        groups = [Point(center_x, center_y)]
    half = max_source_crop_size / 2.0
    inset = min(float(padding), half / 2.0)
    usable_span = max(1.0, max_source_crop_size - 2.0 * inset)
    crops: list[list[int]] = []
    for geometry in groups:
        point_count = max(1, math.ceil(geometry.length / usable_span))
        for point_index in range(point_count):
            if geometry.length > 0:
                center = geometry.interpolate((point_index + 0.5) / point_count, normalized=True)
            else:
                center = geometry.centroid
            if any(
                left + inset <= center.x <= right - inset
                and top + inset <= center.y <= bottom - inset
                for left, top, right, bottom in crops
            ):
                continue
            crops.append(
                [
                    round(center.x - half),
                    round(center.y - half),
                    round(center.x + half),
                    round(center.y + half),
                ]
            )
    return crops


def _read_muno_crop(
    image_path: Path,
    window: list[int],
    *,
    padding: int,
    image_size: int,
    max_source_pixels: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(image_path) as source:
            width, height = source.size
            pixels = width * height
            if pixels > max_source_pixels:
                raise ValueError(
                    f"MUNO21 source image has {pixels} pixels; "
                    f"trusted limit is {max_source_pixels}: {image_path}"
                )
            bounds = (
                max(0, window[0] - padding),
                max(0, window[1] - padding),
                min(width, window[2] + padding),
                min(height, window[3] + padding),
            )
            image = np.asarray(
                source.crop(bounds)
                .convert("RGB")
                .resize((image_size, image_size), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
    return image, bounds


def build_muno21_updater(
    dataset_root: Path,
    output_dir: Path,
    *,
    image_size: int = 512,
    padding: int = 128,
    max_source_crop_size: int = 1024,
    road_width_pixels: float = 6.0,
    max_source_pixels: int = 250_000_000,
    seed: int = 20260710,
) -> dict[str, Any]:
    """Build scenario-level road edits while preserving official city test splits."""

    annotations = json.loads((dataset_root / "annotations.json").read_text(encoding="utf-8"))
    train_regions = set(json.loads((dataset_root / "train.json").read_text(encoding="utf-8")))
    test_regions = set(json.loads((dataset_root / "test.json").read_text(encoding="utf-8")))
    val_regions = _validation_regions(train_regions, seed)
    image_dir = dataset_root / "naip" / "jpg"
    graph_dir = dataset_root / "graphs" / "graphs"
    array_dir = output_dir / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    graph_cache: dict[Path, MunoGraph] = {}
    samples: list[UpdaterSample] = []
    operation_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    progress_path = output_dir / "progress.json"

    for index, annotation in enumerate(annotations):
        cluster = annotation["Cluster"]
        region = str(cluster["Region"])
        tile = cluster["Tile"]
        label = f"{region}_{tile[0]}_{tile[1]}"
        split = "test" if region in test_regions else "val" if region in val_regions else "train"
        image_path = _latest_image(image_dir, label)
        scenario_operation = muno_tags_to_edit(list(annotation.get("Tags") or []))
        prior_graph_path = graph_dir / f"{label}_2013-07-01.graph"
        graph_paths = {
            "prior": prior_graph_path,
            "target": (
                prior_graph_path
                if scenario_operation == EditOperation.KEEP
                else graph_dir / f"{label}_2020-07-01.graph"
            ),
        }
        for graph_path in graph_paths.values():
            if graph_path not in graph_cache:
                graph_cache[graph_path] = read_muno_graph(graph_path)
        source_window = [int(value) for value in cluster["Window"]]
        crop_windows = _scenario_crop_windows(
            annotation,
            padding=padding,
            max_source_crop_size=max_source_crop_size,
        )
        for patch_index, crop_window in enumerate(crop_windows):
            image, bounds = _read_muno_crop(
                image_path,
                crop_window,
                padding=0,
                image_size=image_size,
                max_source_pixels=max_source_pixels,
            )
            image = np.moveaxis(image / 255.0, -1, 0)
            added = _clip_lines(_change_lines(annotation, deleted=False), bounds)
            deleted = _clip_lines(_change_lines(annotation, deleted=True), bounds)
            if added and deleted:
                operation = EditOperation.RESHAPE
            elif added:
                operation = EditOperation.ADD
            elif deleted:
                operation = EditOperation.DELETE
            else:
                operation = scenario_operation
            if operation != EditOperation.KEEP and not added and not deleted:
                skipped["empty_change_geometry"] += 1
                continue
            prior_geometry = _multiline(
                _clip_lines(graph_cache[graph_paths["prior"]].lines(bounds), bounds)
            )
            target_geometry = _multiline(
                _clip_lines(graph_cache[graph_paths["target"]].lines(bounds), bounds)
            )
            transform = Affine(
                (bounds[2] - bounds[0]) / image_size,
                0.0,
                bounds[0],
                0.0,
                (bounds[3] - bounds[1]) / image_size,
                bounds[1],
            )
            prior = rasterize_muno_road_mask(
                prior_geometry, transform, image_size, road_width_pixels
            )
            target = rasterize_muno_road_mask(
                target_geometry, transform, image_size, road_width_pixels
            )
            stem = f"muno21-{index:06d}-p{patch_index:02d}-{region}-{operation.value.lower()}"
            paths = {
                "image": array_dir / f"{stem}-image.npy",
                "prior": array_dir / f"{stem}-prior.npy",
                "target": array_dir / f"{stem}-target.npy",
                "valid": array_dir / f"{stem}-valid.npy",
            }
            np.save(paths["image"], image)
            np.save(paths["prior"], prior)
            np.save(paths["target"], target)
            np.save(paths["valid"], np.ones((image_size, image_size), dtype=np.float32))
            target_edit_geometry = _multiline(added)
            prior_edit_geometry = _multiline(deleted)
            target_bounds = _normalized_bounds(target_edit_geometry, bounds)
            prior_bounds = _normalized_bounds(prior_edit_geometry, bounds)
            samples.append(
                UpdaterSample(
                    sample_id=stem,
                    aoi_id=region,
                    split=split,
                    image_path=paths["image"].relative_to(output_dir).as_posix(),
                    prior_mask_path=paths["prior"].relative_to(output_dir).as_posix(),
                    target_mask_path=paths["target"].relative_to(output_dir).as_posix(),
                    valid_mask_path=paths["valid"].relative_to(output_dir).as_posix(),
                    edit_type=operation,
                    geometry_delta=np.concatenate(
                        [target_bounds, target_bounds - prior_bounds]
                    ).tolist(),
                    object_id=f"muno21-scenario-{index}-patch-{patch_index}",
                    crop_transform=list(transform)[:6],
                    crs="MUNO21_IMAGE_PIXELS",
                    prior_geometry=(
                        GeoJSONGeometry.model_validate(mapping(prior_geometry))
                        if prior_geometry is not None
                        else None
                    ),
                    target_geometry=(
                        GeoJSONGeometry.model_validate(mapping(target_geometry))
                        if target_geometry is not None
                        else None
                    ),
                    clear_fraction=1.0,
                    quality_source="muno21_naip",
                    dataset_name="muno21",
                    geometry_family="polyline",
                    supervision_type="full_scene_temporal",
                    source_metadata={
                        "annotation_index": index,
                        "patch_index": patch_index,
                        "tags": annotation.get("Tags", []),
                        "years": annotation.get("Years", []),
                        "region": region,
                        "tile": tile,
                        "window": source_window,
                        "crop_bounds": list(bounds),
                        "prior_graph": str(graph_paths["prior"]),
                        "target_graph": str(graph_paths["target"]),
                        "source_image": str(image_path),
                    },
                )
            )
            operation_counts[operation.value] += 1
            split_counts[split] += 1
        write_progress(
            progress_path,
            {
                "status": "running",
                "annotations_processed": index + 1,
                "samples": len(samples),
                "operations": dict(operation_counts),
                "splits": dict(split_counts),
                "skipped": dict(skipped),
            },
        )

    manifest_path = output_dir / "updater_samples.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.model_dump_json() + "\n")
    summary = {
        "dataset": "MUNO21",
        "samples": len(samples),
        "operations": dict(sorted(operation_counts.items())),
        "splits": dict(sorted(split_counts.items())),
        "official_test_regions": sorted(test_regions),
        "derived_validation_regions": sorted(val_regions),
        "skipped": dict(sorted(skipped.items())),
        "image_size": image_size,
        "padding": padding,
        "max_source_crop_size": max_source_crop_size,
        "road_width_pixels": road_width_pixels,
        "max_source_pixels": max_source_pixels,
        "seed": seed,
        "manifest": str(manifest_path.resolve()),
        "evaluation": ["MUNO21 APLS improvement", "MUNO21 PixelF1 improvement"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_progress(progress_path, {"status": "complete", **summary})
    return summary


def _evidence_year(path: Path) -> int:
    years = [int(value) for value in re.findall(r"(?:19|20)\d{2}", path.stem)]
    if not years:
        raise ValueError(f"MUNO21 evidence image lacks a year: {path}")
    return max(years)


def build_muno21_evidence_episodes(
    updater_manifest: Path,
    output: Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
    road_width_source_pixels: float = 6.0,
    frozen_test: bool = False,
) -> dict[str, Any]:
    """Convert real multi-year MUNO21 imagery into split-safe agent episodes."""

    requested_splits = set(splits)
    if not requested_splits or not requested_splits <= {"train", "val", "test"}:
        raise ValueError("MUNO21 evidence splits must be train, val, or test")
    test_requested = "test" in requested_splits
    if test_requested:
        if requested_splits != {"test"}:
            raise ValueError("frozen test construction must be test-only")
        if not frozen_test:
            raise PermissionError("test construction requires --frozen-test")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
        if output.exists() or output.with_suffix(".summary.json").exists():
            raise FileExistsError(f"refusing to overwrite frozen test episodes: {output}")
    elif frozen_test:
        raise ValueError("--frozen-test is valid only for the test split")
    if road_width_source_pixels <= 0:
        raise ValueError("road_width_source_pixels must be positive")
    episodes: list[EpisodeRecord] = []
    evidence_histogram: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    with updater_manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                sample = UpdaterSample.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid updater sample at line {line_number}") from exc
            if sample.split not in requested_splits:
                continue
            metadata = sample.source_metadata
            region = str(metadata["region"])
            tile = metadata["tile"]
            label = f"{region}_{tile[0]}_{tile[1]}"
            source_image = Path(str(metadata["source_image"]))
            image_paths = sorted(source_image.parent.glob(f"{label}_*.jpg"), key=_evidence_year)
            if len(image_paths) < 2:
                raise ValueError(f"sample {sample.sample_id} has fewer than two temporal images")
            years = [_evidence_year(path) for path in image_paths]
            latest_year = max(years)
            earliest_year = min(years)
            temporal_span = max(latest_year - earliest_year, 1)
            bounds = tuple(int(value) for value in metadata["crop_bounds"])
            evidence = [
                EvidenceItem(
                    evidence_id=f"{sample.sample_id}__y{year}",
                    timestamp=f"{year}-01",
                    region=bounds,
                    scale=1,
                    image_path=str(path),
                    clear_fraction=1.0,
                    cost=(
                        0.25
                        if year == latest_year
                        else 1.0 + 0.5 * (latest_year - year) / temporal_span
                    ),
                )
                for path, year in zip(image_paths, years, strict=True)
            ]
            object_id = sample.object_id or sample.sample_id
            target_edit_geometry = (
                sample.target_geometry
                if sample.edit_type in {EditOperation.ADD, EditOperation.RESHAPE}
                else None
            )
            episodes.append(
                EpisodeRecord(
                    episode_id=f"{sample.sample_id}__temporal",
                    aoi_id=sample.aoi_id,
                    anchor_timestamp=f"{latest_year}-01",
                    split=sample.split,
                    source_dataset="MUNO21",
                    map_before=str(metadata["prior_graph"]),
                    target_map=str(metadata["target_graph"]),
                    prior_geometry=sample.prior_geometry,
                    target_geometry=sample.target_geometry,
                    hypothesis=CandidateHypothesis(
                        op=EditOperation.KEEP,
                        object_id=object_id,
                        source="editable_prior",
                        confidence=None,
                    ),
                    evidence_catalog=evidence,
                    gt_edit=EditRecord(
                        op=sample.edit_type,
                        object_id=object_id,
                        geometry=target_edit_geometry,
                    ),
                    is_synthetic=False,
                    derivation_version="muno21_temporal_evidence_v1",
                    metadata={
                        "geometry_family": "polyline",
                        "road_width_source_pixels": road_width_source_pixels,
                        "fixed_seed_year": latest_year,
                    },
                )
            )
            evidence_histogram[len(evidence)] += 1
            split_counts[sample.split] += 1

    if not episodes:
        raise ValueError("no MUNO21 train/validation episodes were constructed")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(episode.model_dump_json() + "\n")
    summary = {
        "dataset": "MUNO21",
        "derivation_version": "muno21_temporal_evidence_v1",
        "episodes": len(episodes),
        "splits": dict(sorted(split_counts.items())),
        "evidence_count_histogram": dict(sorted(evidence_histogram.items())),
        "fixed_seed": "latest image (cost excluded by selector oracle)",
        "additional_evidence": "earlier real NAIP years",
        "road_width_source_pixels": road_width_source_pixels,
        "test_assets_read": test_requested,
        "output": str(output.resolve()),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
