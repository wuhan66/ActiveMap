"""Portable prediction and result contracts for third-party baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from activemap.models import EditOperation, GeoJSONGeometry


class ExternalBaselinePrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["activemap-external-baseline-prediction-v1"]
    sample_id: str
    split: Literal["train", "validation", "test"]
    dataset: str
    baseline: str
    operation: EditOperation
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    geometry: GeoJSONGeometry | None = None
    source_artifact: str
    test_assets_read: bool = False

    @model_validator(mode="after")
    def validate_edit_payload(self) -> ExternalBaselinePrediction:
        if self.test_assets_read and self.split != "test":
            raise ValueError("non-test prediction cannot report test asset access")
        if self.operation in {EditOperation.ADD, EditOperation.RESHAPE}:
            if self.geometry is None:
                raise ValueError(f"{self.operation.value} requires geometry")
        elif self.geometry is not None:
            raise ValueError(f"{self.operation.value} must not carry replacement geometry")
        return self


class ExternalBaselineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["activemap-external-baseline-result-v1"]
    run_id: str
    dataset: str
    baseline: str
    split: Literal["validation", "test"]
    seed: int
    source_commit: str
    predictions_path: str
    sample_count: int = Field(gt=0)
    metrics: dict[str, float]
    test_assets_read: bool

    @model_validator(mode="after")
    def validate_test_access(self) -> ExternalBaselineResult:
        if self.test_assets_read != (self.split == "test"):
            raise ValueError("test_assets_read must match the evaluated split")
        if not self.metrics:
            raise ValueError("external baseline result requires metrics")
        return self


class StructuredMapPrediction(BaseModel):
    """Object-level prediction shared by ArgoTweak and vector-map backends."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["activemap-structured-map-prediction-v1"]
    sample_id: str
    split: Literal["train", "val", "test"]
    dataset: str
    baseline: str
    object_id: str
    operation: EditOperation
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    geometry: GeoJSONGeometry | None = None
    attributes_after: dict[str, str | int | float | bool] = Field(default_factory=dict)
    source_artifact: str
    test_assets_read: bool = False

    @model_validator(mode="after")
    def validate_structured_edit(self) -> StructuredMapPrediction:
        if self.test_assets_read != (self.split == "test"):
            raise ValueError("test_assets_read must match split")
        if self.operation in {EditOperation.ADD, EditOperation.RESHAPE}:
            if self.geometry is None:
                raise ValueError(f"{self.operation.value} requires geometry")
        elif self.geometry is not None:
            raise ValueError(f"{self.operation.value} must not carry geometry")
        return self


def validate_prediction_jsonl(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            count += 1
            try:
                ExternalBaselinePrediction.model_validate_json(line)
            except Exception as exc:
                errors.append(f"{path}:{line_number}: {exc}")
    if not count:
        errors.append(f"{path}: no predictions")
    return count, errors


def write_prediction_jsonl(
    predictions: list[ExternalBaselinePrediction],
    path: Path,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite predictions: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for prediction in sorted(predictions, key=lambda row: row.sample_id):
            handle.write(
                json.dumps(
                    prediction.model_dump(mode="json"),
                    separators=(",", ":"),
                )
                + "\n"
            )


def validate_structured_prediction_jsonl(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    count = 0
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            count += 1
            try:
                row = StructuredMapPrediction.model_validate_json(line)
                key = (row.sample_id, row.object_id)
                if key in seen:
                    raise ValueError(f"duplicate sample/object prediction: {key}")
                seen.add(key)
            except Exception as exc:
                errors.append(f"{path}:{line_number}: {exc}")
    if not count:
        errors.append(f"{path}: no structured-map predictions")
    return count, errors


def write_structured_prediction_jsonl(
    predictions: list[StructuredMapPrediction],
    path: Path,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite predictions: {path}")
    keys = [(row.sample_id, row.object_id) for row in predictions]
    if len(keys) != len(set(keys)):
        raise ValueError("structured-map predictions contain duplicate sample/object pairs")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for prediction in sorted(predictions, key=lambda row: (row.sample_id, row.object_id)):
            handle.write(
                json.dumps(prediction.model_dump(mode="json"), separators=(",", ":")) + "\n"
            )
