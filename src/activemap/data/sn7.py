"""Build a monthly SpaceNet 7 asset manifest without assuming one archive layout."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import rasterio

TIMESTAMP_PATTERN = re.compile(r"(?P<year>20\d{2})[_-](?P<month>0[1-9]|1[0-2])")
MOSAIC_AOI_PATTERN = re.compile(
    r"mosaic_(?P<aoi>.+?)(?:_Buildings|_UDM)?\.(?:tif|tiff|geojson|json)$",
    flags=re.IGNORECASE,
)
IGNORED_AOI_DIRS = {
    "train",
    "test",
    "test_public",
    "test_private",
    "images",
    "images_masked",
    "labels",
    "labels_match",
    "labels_match_pix",
    "udm",
    "udm_masks",
    "geojson",
}


def extract_timestamp(path: Path) -> str | None:
    match = TIMESTAMP_PATTERN.search(path.name)
    if match is None:
        return None
    return f"{match.group('year')}_{match.group('month')}"


def extract_aoi_id(path: Path, root: Path) -> str | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path

    for part in reversed(relative.parts[:-1]):
        normalized = part.lower()
        if normalized in IGNORED_AOI_DIRS:
            continue
        if TIMESTAMP_PATTERN.fullmatch(part):
            continue
        if part.startswith(("L15-", "AOI_", "AOI-")):
            return part

    filename_match = MOSAIC_AOI_PATTERN.search(path.name)
    if filename_match is not None:
        return filename_match.group("aoi")

    for part in reversed(relative.parts[:-1]):
        if part.lower() not in IGNORED_AOI_DIRS:
            return part
    return None


def classify_asset(path: Path) -> str | None:
    suffix = path.suffix.lower()
    lowered = str(path).lower()
    if suffix in {".geojson", ".json"} and "label" in lowered:
        return "label"
    if suffix in {".geojson", ".json"} and "building" in path.name.lower():
        return "label"
    if suffix not in {".tif", ".tiff"}:
        return None
    if "udm" in lowered:
        return "udm"
    return "image"


def _asset_priority(path: Path, asset_type: str) -> int:
    parent = path.parent.name.lower()
    priorities = {
        "image": {"images_masked": 0, "images": 1},
        "label": {"labels_match": 0, "labels": 1, "labels_match_pix": 2},
        "udm": {"udm_masks": 0, "udm": 0},
    }
    return priorities.get(asset_type, {}).get(parent, 10)


def _preferred_assets(paths: list[Path], asset_type: str) -> list[Path]:
    if not paths:
        return []
    best_priority = min(_asset_priority(path, asset_type) for path in paths)
    return [path for path in paths if _asset_priority(path, asset_type) == best_priority]


def _raster_metadata(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as dataset:
        transform = tuple(dataset.transform)[:6]
        return {
            "width": dataset.width,
            "height": dataset.height,
            "band_count": dataset.count,
            "dtype": dataset.dtypes[0] if dataset.dtypes else None,
            "crs": dataset.crs.to_string() if dataset.crs else None,
            "transform": list(transform),
        }


def build_sn7_manifest(
    root: Path,
    *,
    read_raster_metadata: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SpaceNet 7 root does not exist: {root}")

    grouped: dict[tuple[str, str], dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        asset_type = classify_asset(path)
        if asset_type is None:
            continue
        timestamp = extract_timestamp(path)
        aoi_id = extract_aoi_id(path, root)
        if timestamp is None or aoi_id is None:
            if strict:
                raise ValueError(f"cannot parse AOI/timestamp from asset: {path}")
            continue
        grouped[(aoi_id, timestamp)][asset_type].append(path.resolve())

    records: list[dict[str, Any]] = []
    for (aoi_id, timestamp), assets in sorted(grouped.items()):
        images = _preferred_assets(assets.get("image", []), "image")
        if not images:
            if strict:
                raise ValueError(f"no image for {aoi_id} at {timestamp}")
            continue
        if len(images) > 1:
            raise ValueError(f"multiple images for {aoi_id} at {timestamp}: {images}")

        labels = _preferred_assets(assets.get("label", []), "label")
        udms = _preferred_assets(assets.get("udm", []), "udm")
        if len(labels) > 1 or len(udms) > 1:
            raise ValueError(f"duplicate label/UDM assets for {aoi_id} at {timestamp}")

        image_path = images[0]
        record: dict[str, Any] = {
            "aoi_id": aoi_id,
            "timestamp": timestamp,
            "image_path": str(image_path),
            "label_path": str(labels[0]) if labels else None,
            "udm_path": str(udms[0]) if udms else None,
            "has_label": bool(labels),
            "has_udm": bool(udms),
            "image_variant": image_path.parent.name,
            "label_variant": labels[0].parent.name if labels else None,
        }
        if read_raster_metadata:
            try:
                record.update(_raster_metadata(image_path))
            except Exception as exc:
                if strict:
                    raise ValueError(f"failed to read raster metadata: {image_path}") from exc
                record["metadata_error"] = str(exc)
        records.append(record)

    if not records:
        raise ValueError(f"no SpaceNet 7 monthly images found under: {root}")

    frame = pd.DataFrame.from_records(records).sort_values(
        ["aoi_id", "timestamp"], ignore_index=True
    )
    if frame.duplicated(["aoi_id", "timestamp"]).any():
        raise ValueError("manifest contains duplicate AOI/timestamp rows")
    return frame
