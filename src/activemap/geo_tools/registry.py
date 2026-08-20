"""Registry and failure boundary for agent-callable geospatial tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from activemap.geo_tools.records import GeoToolCall, GeoToolName, GeoToolResult


class GeoTool(Protocol):
    name: GeoToolName
    cost: float

    def run(self, call: GeoToolCall) -> GeoToolResult: ...


class GeoToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[GeoToolName, GeoTool] = {}

    def register(self, tool: GeoTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name.value}")
        if tool.cost < 0:
            raise ValueError("tool cost must be non-negative")
        self._tools[tool.name] = tool

    def cost(self, name: GeoToolName) -> float:
        if name not in self._tools:
            raise ValueError(f"tool is not registered: {name.value}")
        return float(self._tools[name].cost)

    def execute(
        self,
        call: GeoToolCall,
        *,
        asset_paths: Mapping[str, str] | None = None,
    ) -> GeoToolResult:
        if call.tool not in self._tools:
            return GeoToolResult(
                call_id=call.call_id,
                tool=call.tool,
                success=False,
                cost=0.0,
                error=f"tool is not registered: {call.tool.value}",
            )
        tool = self._tools[call.tool]
        resolved_call = call
        evidence_id = call.inputs.get("evidence_id")
        if evidence_id is not None and "image_path" not in call.inputs:
            if asset_paths is None or str(evidence_id) not in asset_paths:
                return GeoToolResult(
                    call_id=call.call_id,
                    tool=call.tool,
                    success=False,
                    cost=float(tool.cost),
                    error=f"unknown evidence asset: {evidence_id}",
                )
            inputs = dict(call.inputs)
            inputs["image_path"] = asset_paths[str(evidence_id)]
            resolved_call = call.model_copy(update={"inputs": inputs})
        try:
            result = tool.run(resolved_call)
        except Exception as exc:
            return GeoToolResult(
                call_id=call.call_id,
                tool=call.tool,
                success=False,
                cost=float(tool.cost),
                error=f"{type(exc).__name__}: {exc}",
            )
        if result.call_id != call.call_id or result.tool != call.tool:
            raise ValueError("tool returned a result for a different call")
        return result

    def names(self) -> list[GeoToolName]:
        return sorted(self._tools, key=lambda item: item.value)
