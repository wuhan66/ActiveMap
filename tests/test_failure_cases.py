from activemap.evaluation.failure_cases import select_failure_cases
from activemap.evaluation.update import UpdatePrediction
from activemap.models import EditOperation


def _record(
    sample_id: str,
    target: EditOperation,
    predicted: EditOperation,
    *,
    confidence: float,
    polygon_iou: float | None = None,
    topology_valid: bool | None = True,
) -> UpdatePrediction:
    return UpdatePrediction(
        sample_id=sample_id,
        aoi_id="aoi",
        target_edit=target,
        predicted_edit=predicted,
        confidence=confidence,
        polygon_iou=polygon_iou,
        topology_valid=topology_valid,
    )


def test_failure_selection_is_ranked_and_categorized() -> None:
    records = [
        _record("false-low", EditOperation.KEEP, EditOperation.DELETE, confidence=0.4),
        _record("false-high", EditOperation.KEEP, EditOperation.DELETE, confidence=0.9),
        _record("missed", EditOperation.DELETE, EditOperation.KEEP, confidence=0.8),
        _record("wrong", EditOperation.DELETE, EditOperation.RESHAPE, confidence=0.7),
        _record(
            "geometry",
            EditOperation.RESHAPE,
            EditOperation.RESHAPE,
            confidence=0.9,
            polygon_iou=0.1,
        ),
        _record(
            "topology",
            EditOperation.ADD,
            EditOperation.ADD,
            confidence=0.6,
            polygon_iou=0.8,
            topology_valid=False,
        ),
    ]
    cases = select_failure_cases(records, per_category=1)
    by_category = {case["failure_category"]: case for case in cases}
    assert by_category["false_edit"]["prediction"]["sample_id"] == "false-high"
    assert by_category["missed_update"]["prediction"]["sample_id"] == "missed"
    assert by_category["wrong_update_type"]["prediction"]["sample_id"] == "wrong"
    assert by_category["low_geometry_iou"]["prediction"]["sample_id"] == "geometry"
    assert by_category["topology_invalid"]["prediction"]["sample_id"] == "topology"
