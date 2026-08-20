"""Stable numeric encoding for heterogeneous geospatial tool results."""

from __future__ import annotations

import math
from collections.abc import Callable

from activemap.geo_tools.records import GeoToolName, GeoToolResult
from activemap.models import EditOperation

_TOOLS = list(GeoToolName)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _direct(key: str) -> Callable[[dict[str, object]], float | None]:
    return lambda outputs: _number(outputs.get(key))


def _component_count(outputs: dict[str, object]) -> float | None:
    value = _number(outputs.get("component_count"))
    return math.log1p(max(value, 0.0)) if value is not None else None


def _invalid_fraction(outputs: dict[str, object]) -> float | None:
    invalid = _number(outputs.get("invalid_count"))
    total = _number(outputs.get("feature_count"))
    if invalid is None or total is None:
        return None
    return invalid / max(total, 1.0)


_SCALARS: tuple[tuple[str, Callable[[dict[str, object]], float | None]], ...] = (
    ("valid_fraction", _direct("valid_fraction")),
    ("saturation_fraction", _direct("saturation_fraction")),
    ("sharpness", _direct("sharpness")),
    ("changed_fraction", _direct("changed_fraction")),
    ("foreground_fraction", _direct("foreground_fraction")),
    ("log_component_count", _component_count),
    ("invalid_fraction", _invalid_fraction),
)

TOOL_RESULT_FEATURE_NAMES = tuple(
    [f"tool_{tool.value}" for tool in _TOOLS]
    + ["success", "cost"]
    + [name for name, _ in _SCALARS]
    + [f"has_{name}" for name, _ in _SCALARS]
)
TOOL_RESULT_FEATURE_DIM = len(TOOL_RESULT_FEATURE_NAMES)

_SEMANTIC_GEOMETRY_SCALARS: tuple[
    tuple[str, Callable[[dict[str, object]], float | None]], ...
] = (
    ("foreground_fraction", _direct("foreground_fraction")),
    ("changed_fraction", _direct("changed_fraction")),
    ("add_fraction", _direct("add_fraction")),
    ("remove_fraction", _direct("remove_fraction")),
    ("learned_add_fraction", _direct("learned_add_fraction")),
    ("learned_remove_fraction", _direct("learned_remove_fraction")),
    ("mean_add_probability", _direct("mean_add_probability")),
    ("mean_remove_probability", _direct("mean_remove_probability")),
    ("prior_iou", _direct("prior_iou")),
    ("mean_probability", _direct("mean_probability")),
    ("log_component_count", _component_count),
)


def _edit_probability(index: int) -> Callable[[dict[str, object]], float | None]:
    def extract(outputs: dict[str, object]) -> float | None:
        raw = outputs.get("edit_probabilities")
        if not isinstance(raw, list) or len(raw) != len(EditOperation):
            return None
        values = [_number(value) for value in raw]
        if any(value is None or not 0.0 <= value <= 1.0 for value in values):
            return None
        numeric = [float(value) for value in values if value is not None]
        if abs(sum(numeric) - 1.0) > 1e-3:
            return None
        return numeric[index]

    return extract


_SEMANTIC_OPERATION_SCALARS: tuple[
    tuple[str, Callable[[dict[str, object]], float | None]], ...
] = tuple(
    (f"edit_probability_{edit.value.lower()}", _edit_probability(index))
    for index, edit in enumerate(EditOperation)
) + (
    ("update_probability", _direct("update_probability")),
    ("confidence", _direct("confidence")),
    ("uncertainty", _direct("uncertainty")),
)

_SEMANTIC_SCALARS = _SEMANTIC_GEOMETRY_SCALARS + _SEMANTIC_OPERATION_SCALARS

SEMANTIC_TOOL_FEATURE_NAMES = tuple(
    ["success", "cost"]
    + [name for name, _ in _SEMANTIC_SCALARS]
    + [f"has_{name}" for name, _ in _SEMANTIC_SCALARS]
)
SEMANTIC_TOOL_FEATURE_DIM = len(SEMANTIC_TOOL_FEATURE_NAMES)


def encode_tool_result(result: GeoToolResult) -> list[float]:
    """Encode a result without assigning semantic edit thresholds."""

    one_hot = [float(result.tool == tool) for tool in _TOOLS]
    outputs = result.outputs if result.success else {}
    values = [extractor(outputs) for _, extractor in _SCALARS]
    return (
        one_hot
        + [float(result.success), float(result.cost)]
        + [value if value is not None else 0.0 for value in values]
        + [float(value is not None) for value in values]
    )


def encode_semantic_tool_result(result: GeoToolResult) -> list[float]:
    """Encode map-relative geometry and calibrated operation evidence."""

    if result.tool != GeoToolName.RASTER_SEGMENT:
        raise ValueError("semantic tool features require RASTER_SEGMENT")
    outputs = result.outputs if result.success else {}
    values = [extractor(outputs) for _, extractor in _SEMANTIC_SCALARS]
    return (
        [float(result.success), float(result.cost)]
        + [value if value is not None else 0.0 for value in values]
        + [float(value is not None) for value in values]
    )
