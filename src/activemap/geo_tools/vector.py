"""Vector-map inspection tool for topology-aware agent decisions."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import geopandas as gpd

from activemap.geo_tools.records import GeoToolCall, GeoToolName, GeoToolResult


class VectorInspectTool:
    name = GeoToolName.VECTOR_INSPECT
    cost = 0.05

    def run(self, call: GeoToolCall) -> GeoToolResult:
        path = Path(call.inputs["vector_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = gpd.read_file(path)
        geometry = frame.geometry
        valid = geometry.is_valid & ~geometry.is_empty & geometry.notna()
        bounds = frame.total_bounds.tolist() if len(frame) else []
        return GeoToolResult(
            call_id=call.call_id,
            tool=self.name,
            success=True,
            outputs={
                "feature_count": len(frame),
                "valid_count": int(valid.sum()),
                "invalid_count": int((~valid).sum()),
                "geometry_types": dict(Counter(geometry.geom_type.fillna("None"))),
                "crs": str(frame.crs) if frame.crs else None,
                "bounds": bounds,
            },
            cost=self.cost,
        )
