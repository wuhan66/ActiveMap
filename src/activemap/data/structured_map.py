"""Neutral structured-map samples for autonomous-driving dataset adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from activemap.models import EditOperation, GeoJSONGeometry


class StructuredMapObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str
    timestamp: str
    modality: Literal[
        "camera", "camera_bundle", "lidar", "aerial", "raster_map", "vector_map"
    ]
    path: str
    cost: float = Field(gt=0.0)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class StructuredAtomicEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: EditOperation
    object_id: str
    prior_geometry: GeoJSONGeometry | None = None
    target_geometry: GeoJSONGeometry | None = None
    attributes_before: dict[str, str | int | float | bool] = Field(default_factory=dict)
    attributes_after: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_geometry_transition(self) -> StructuredAtomicEdit:
        if self.operation == EditOperation.ADD:
            if self.prior_geometry is not None or self.target_geometry is None:
                raise ValueError("ADD requires only target_geometry")
        elif self.operation == EditOperation.DELETE:
            if self.prior_geometry is None or self.target_geometry is not None:
                raise ValueError("DELETE requires only prior_geometry")
        else:
            if self.prior_geometry is None or self.target_geometry is None:
                raise ValueError(f"{self.operation.value} requires prior and target geometry")
        return self


class StructuredMapSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["activemap-structured-map-sample-v1"]
    sample_id: str
    dataset: str
    split: Literal["train", "val", "test"]
    aoi_id: str
    prior_map_path: str
    target_map_path: str
    observations: list[StructuredMapObservation]
    atomic_edits: list[StructuredAtomicEdit]
    native_sample_id: str
    test_assets_read: bool = False
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_sample(self) -> StructuredMapSample:
        observation_ids = [row.observation_id for row in self.observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observation IDs must be unique")
        if not self.observations:
            raise ValueError("structured map sample requires observations")
        if not self.atomic_edits:
            raise ValueError("structured map sample requires atomic edits")
        if self.test_assets_read != (self.split == "test"):
            raise ValueError("test_assets_read must match split")
        return self


class StructuredMapScene(BaseModel):
    """Portable input scene consumed by native HD-map dataset adapters."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["activemap-structured-map-scene-v1"]
    sample_id: str
    dataset: str
    split: Literal["train", "val", "test"]
    aoi_id: str
    native_sample_id: str
    prior_map_path: str
    target_map_path: str
    observations: list[StructuredMapObservation]
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_scene(self) -> StructuredMapScene:
        if not self.observations:
            raise ValueError("structured map scene requires observations")
        return self


def _feature_id(feature: dict[str, Any]) -> str:
    properties = feature.get("properties") or {}
    candidates = (
        feature.get("id"),
        properties.get("object_id"),
        properties.get("id"),
        properties.get("token"),
        properties.get("uuid"),
    )
    value = next((candidate for candidate in candidates if candidate not in {None, ""}), None)
    if value is None:
        raise ValueError("GeoJSON feature lacks id/object_id/token/uuid")
    return str(value)


def _scalar_attributes(feature: dict[str, Any]) -> dict[str, str | int | float | bool]:
    ignored = {"object_id", "id", "token", "uuid"}
    return {
        str(key): value
        for key, value in (feature.get("properties") or {}).items()
        if key not in ignored and isinstance(value, str | int | float | bool)
    }


def _load_feature_map(path: Path) -> dict[str, tuple[GeoJSONGeometry, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "FeatureCollection" or not isinstance(
        payload.get("features"), list
    ):
        raise ValueError(f"expected GeoJSON FeatureCollection: {path}")
    result: dict[str, tuple[GeoJSONGeometry, dict[str, Any]]] = {}
    for feature in payload["features"]:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError(f"invalid GeoJSON feature in {path}")
        object_id = _feature_id(feature)
        if object_id in result:
            raise ValueError(f"duplicate object id {object_id!r} in {path}")
        geometry = GeoJSONGeometry.model_validate(feature.get("geometry"))
        result[object_id] = (geometry, _scalar_attributes(feature))
    return result


def derive_structured_atomic_edits(
    prior_map_path: Path,
    target_map_path: Path,
    *,
    include_keep: bool = False,
) -> tuple[list[StructuredAtomicEdit], dict[str, int]]:
    """Derive typed edits by stable object identity from two vector-map snapshots."""

    prior = _load_feature_map(prior_map_path)
    target = _load_feature_map(target_map_path)
    edits: list[StructuredAtomicEdit] = []
    counts = {operation.value: 0 for operation in EditOperation}
    unchanged: list[StructuredAtomicEdit] = []
    for object_id in sorted(set(prior) | set(target)):
        before = prior.get(object_id)
        after = target.get(object_id)
        if before is None and after is not None:
            edit = StructuredAtomicEdit(
                operation=EditOperation.ADD,
                object_id=object_id,
                target_geometry=after[0],
                attributes_after=after[1],
            )
        elif before is not None and after is None:
            edit = StructuredAtomicEdit(
                operation=EditOperation.DELETE,
                object_id=object_id,
                prior_geometry=before[0],
                attributes_before=before[1],
            )
        elif before is not None and after is not None:
            same_geometry = before[0].model_dump(mode="json") == after[0].model_dump(
                mode="json"
            )
            same_attributes = before[1] == after[1]
            operation = (
                EditOperation.KEEP
                if same_geometry and same_attributes
                else EditOperation.RESHAPE
            )
            edit = StructuredAtomicEdit(
                operation=operation,
                object_id=object_id,
                prior_geometry=before[0],
                target_geometry=after[0],
                attributes_before=before[1],
                attributes_after=after[1],
            )
        else:  # pragma: no cover - exhaustive set union
            raise AssertionError(object_id)
        counts[edit.operation.value] += 1
        if edit.operation == EditOperation.KEEP:
            unchanged.append(edit)
        else:
            edits.append(edit)
    if include_keep:
        edits.extend(unchanged)
    elif not edits and unchanged:
        edits.append(unchanged[0])
    if not edits:
        raise ValueError("map pair contains no identifiable objects")
    return sorted(edits, key=lambda row: (row.object_id, row.operation.value)), counts


def _resolve_scene_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def convert_structured_map_scenes(
    manifest_path: Path,
    output_path: Path,
    *,
    allow_test: bool = False,
    include_keep: bool = False,
    check_paths: bool = True,
) -> dict[str, Any]:
    """Convert portable native scenes into the shared structured-map sample index."""

    scenes = [
        StructuredMapScene.model_validate_json(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not scenes:
        raise ValueError(f"no structured map scenes in {manifest_path}")
    samples = []
    operation_counts = {operation.value: 0 for operation in EditOperation}
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for scene in scenes:
        if scene.split == "test" and not allow_test:
            raise PermissionError("test scene conversion requires allow_test=True")
        prior_path = _resolve_scene_path(scene.prior_map_path, manifest_path)
        target_path = _resolve_scene_path(scene.target_map_path, manifest_path)
        observation_paths = [
            _resolve_scene_path(row.path, manifest_path) for row in scene.observations
        ]
        if check_paths:
            for path in (prior_path, target_path, *observation_paths):
                if not path.is_file():
                    raise FileNotFoundError(path)
        edits, counts = derive_structured_atomic_edits(
            prior_path,
            target_path,
            include_keep=include_keep,
        )
        for operation, count in counts.items():
            operation_counts[operation] += count
        split_counts[scene.split] += 1
        observations = [
            row.model_copy(update={"path": str(path)})
            for row, path in zip(scene.observations, observation_paths, strict=True)
        ]
        samples.append(
            StructuredMapSample(
                schema_version="activemap-structured-map-sample-v1",
                sample_id=scene.sample_id,
                dataset=scene.dataset,
                split=scene.split,
                aoi_id=scene.aoi_id,
                prior_map_path=str(prior_path),
                target_map_path=str(target_path),
                observations=observations,
                atomic_edits=edits,
                native_sample_id=scene.native_sample_id,
                test_assets_read=scene.split == "test",
                metadata={
                    **scene.metadata,
                    "derived_edit_count": len(edits),
                    "source_object_count": sum(counts.values()),
                },
            )
        )
    write_structured_map_jsonl(samples, output_path)
    return {
        "schema_version": "activemap-structured-map-conversion-summary-v1",
        "source_manifest": str(manifest_path.resolve()),
        "output": str(output_path.resolve()),
        "sample_count": len(samples),
        "split_counts": split_counts,
        "operation_counts": operation_counts,
        "include_keep": include_keep,
        "test_assets_read": any(scene.split == "test" for scene in scenes),
    }


def validate_structured_map_jsonl(
    path: Path,
    *,
    allow_test: bool = False,
    check_paths: bool = False,
) -> tuple[int, list[str]]:
    errors: list[str] = []
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            count += 1
            try:
                sample = StructuredMapSample.model_validate_json(line)
                if sample.split == "test" and not allow_test:
                    raise ValueError("test sample is locked")
                if check_paths:
                    for value in (
                        sample.prior_map_path,
                        sample.target_map_path,
                        *(row.path for row in sample.observations),
                    ):
                        if not Path(value).is_file():
                            raise FileNotFoundError(value)
            except Exception as exc:
                errors.append(f"{path}:{line_number}: {exc}")
    if not count:
        errors.append(f"{path}: no structured map samples")
    return count, errors


def write_structured_map_jsonl(samples: list[StructuredMapSample], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite structured map index: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for sample in sorted(samples, key=lambda row: row.sample_id):
            handle.write(
                json.dumps(sample.model_dump(mode="json"), separators=(",", ":")) + "\n"
            )
