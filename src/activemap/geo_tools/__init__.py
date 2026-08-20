"""Typed, auditable raster and vector tools for the ActiveMap agent."""

from pathlib import Path

from activemap.geo_tools.records import GeoToolCall, GeoToolName, GeoToolResult
from activemap.geo_tools.registry import GeoToolRegistry


def build_default_registry(output_root: Path) -> GeoToolRegistry:
    """Load the full raster/vector toolbox only when it is requested."""
    from activemap.geo_tools.default import build_default_registry as build

    return build(output_root)

__all__ = [
    "GeoToolCall",
    "GeoToolName",
    "GeoToolRegistry",
    "GeoToolResult",
    "build_default_registry",
]
