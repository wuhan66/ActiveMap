"""Serializable geospatial tool calls and results."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from activemap.models import StrictModel


class GeoToolName(str, Enum):
    RASTER_CROP = "RASTER_CROP"
    IMAGE_QUALITY = "IMAGE_QUALITY"
    SPECTRAL_INDEX = "SPECTRAL_INDEX"
    TEMPORAL_CHANGE = "TEMPORAL_CHANGE"
    RASTER_SEGMENT = "RASTER_SEGMENT"
    TERRAIN_ANALYSIS = "TERRAIN_ANALYSIS"
    VECTOR_INSPECT = "VECTOR_INSPECT"


class GeoToolCall(StrictModel):
    call_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    tool: GeoToolName
    inputs: dict[str, Any]
    parameters: dict[str, Any] = Field(default_factory=dict)


class GeoToolResult(StrictModel):
    call_id: str
    tool: GeoToolName
    success: bool
    outputs: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    cost: float = Field(ge=0.0)
    error: str | None = None
