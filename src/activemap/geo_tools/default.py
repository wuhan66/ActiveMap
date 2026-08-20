"""Default geospatial toolbox with normalized compute costs."""

from __future__ import annotations

from pathlib import Path

from activemap.geo_tools.raster import (
    ImageQualityTool,
    RasterCropTool,
    RasterSegmentTool,
    SpectralIndexTool,
    TemporalChangeTool,
    TerrainAnalysisTool,
)
from activemap.geo_tools.registry import GeoToolRegistry
from activemap.geo_tools.vector import VectorInspectTool


def build_default_registry(output_root: Path) -> GeoToolRegistry:
    registry = GeoToolRegistry()
    registry.register(RasterCropTool(output_root / "crops"))
    registry.register(ImageQualityTool())
    registry.register(SpectralIndexTool(output_root / "indices"))
    registry.register(TemporalChangeTool(output_root / "change"))
    registry.register(RasterSegmentTool(output_root / "segments"))
    registry.register(TerrainAnalysisTool(output_root / "terrain"))
    registry.register(VectorInspectTool())
    return registry
