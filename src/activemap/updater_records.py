"""Records for prior-conditioned raster updater training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from activemap.models import EditOperation, GeoJSONGeometry, StrictModel


class UpdaterSample(StrictModel):
    sample_id: str
    aoi_id: str | None = None
    split: str
    image_path: str
    prior_image_path: str | None = None
    prior_mask_path: str
    target_mask_path: str
    edit_type: EditOperation
    geometry_delta: list[float]
    valid_mask_path: str | None = None
    object_id: str | None = None
    crop_transform: list[float] | None = None
    crs: str | None = None
    prior_geometry: GeoJSONGeometry | None = None
    target_geometry: GeoJSONGeometry | None = None
    clear_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_source: str | None = None
    dataset_name: str = "legacy"
    geometry_family: str = "polygon"
    supervision_type: str = "real_temporal"
    source_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("split")
    @classmethod
    def known_split(cls, value: str) -> str:
        if value not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        return value

    @field_validator("geometry_delta")
    @classmethod
    def geometry_delta_has_eight_values(cls, value: list[float]) -> list[float]:
        if len(value) != 8:
            raise ValueError("geometry_delta must have eight values")
        return value

    @field_validator("geometry_family")
    @classmethod
    def known_geometry_family(cls, value: str) -> str:
        if value not in {"polygon", "polyline"}:
            raise ValueError("geometry_family must be polygon or polyline")
        return value

    @field_validator("supervision_type")
    @classmethod
    def known_supervision_type(cls, value: str) -> str:
        supported = {
            "real_temporal",
            "full_scene_temporal",
            "synthetic_prior",
            "single_timestamp",
        }
        if value not in supported:
            raise ValueError(f"unsupported supervision_type: {value}")
        return value

    @field_validator("crop_transform")
    @classmethod
    def crop_transform_has_six_values(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) != 6:
            raise ValueError("crop_transform must have six affine values")
        return value


def load_updater_samples(path: Path, *, split: str | None = None) -> list[UpdaterSample]:
    records: list[UpdaterSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = UpdaterSample.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid updater sample at line {line_number}") from exc
            resolved_paths: dict[str, str | None] = {}
            for field in (
                "image_path",
                "prior_image_path",
                "prior_mask_path",
                "target_mask_path",
                "valid_mask_path",
            ):
                value = getattr(record, field)
                if value is None:
                    resolved_paths[field] = None
                    continue
                candidate = Path(value)
                resolved_paths[field] = str(
                    candidate if candidate.is_absolute() else (path.parent / candidate).resolve()
                )
            record = record.model_copy(update=resolved_paths)
            if split is None or record.split == split:
                records.append(record)
    if not records:
        raise ValueError(f"no updater samples found for split={split!r} in {path}")
    return records
