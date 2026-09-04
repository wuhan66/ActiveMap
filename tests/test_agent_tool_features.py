import math

from activemap.agent.tool_features import (
    TOOL_RESULT_FEATURE_DIM,
    TOOL_RESULT_FEATURE_NAMES,
    encode_tool_result,
)
from activemap.geo_tools.records import GeoToolName, GeoToolResult


def test_tool_result_encoder_preserves_values_and_missingness() -> None:
    result = GeoToolResult(
        call_id="segment-1",
        tool=GeoToolName.RASTER_SEGMENT,
        success=True,
        outputs={
            "foreground_fraction": 0.3,
            "component_count": 4,
        },
        cost=0.12,
    )

    features = encode_tool_result(result)
    encoded = dict(zip(TOOL_RESULT_FEATURE_NAMES, features, strict=True))

    assert len(features) == TOOL_RESULT_FEATURE_DIM
    assert encoded["tool_RASTER_SEGMENT"] == 1.0
    assert encoded["success"] == 1.0
    assert encoded["foreground_fraction"] == 0.3
    assert encoded["log_component_count"] == math.log1p(4)
    assert encoded["has_foreground_fraction"] == 1.0
    assert encoded["changed_fraction"] == 0.0
    assert encoded["has_changed_fraction"] == 0.0


def test_failed_tool_result_does_not_expose_partial_outputs() -> None:
    result = GeoToolResult(
        call_id="change-failed",
        tool=GeoToolName.TEMPORAL_CHANGE,
        success=False,
        outputs={"changed_fraction": 0.9},
        cost=0.15,
        error="registration failed",
    )

    encoded = dict(
        zip(TOOL_RESULT_FEATURE_NAMES, encode_tool_result(result), strict=True)
    )

    assert encoded["tool_TEMPORAL_CHANGE"] == 1.0
    assert encoded["success"] == 0.0
    assert encoded["changed_fraction"] == 0.0
    assert encoded["has_changed_fraction"] == 0.0
