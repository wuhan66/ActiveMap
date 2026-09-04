import pytest

from activemap.models import EditOperation
from scripts.build_edit_only_validation_manifest import build_manifest
from activemap.selector_records import SelectorSample


def _with_target(index: int, operation: str):
    return SelectorSample(
        sample_id=f"val-{index}",
        split="val",
        edit_type=EditOperation(operation),
        hypothesis_features=[0.0] * 16,
        state_features=[0.0] * 8,
        evidence_ids=[f"e-{index}"],
        evidence_features=[[0.0] * 13],
        evidence_costs=[1.0],
        false_edit_risks=[0.0],
        oracle_utilities=[0.0],
        metadata={
            "source_episode": f"episode-{index}",
            "aoi_id": f"aoi-{index % 2}",
            "budget": 3.0,
            "oracle_step": 0,
            "gt_edit": operation,
        },
    )


def test_edit_only_manifest_is_complete_and_label_only(tmp_path):
    samples = [_with_target(index, operation) for index, operation in enumerate(
        ("KEEP", "ADD", "DELETE", "RESHAPE")
    )]
    states = tmp_path / "states.jsonl"
    states.write_text(
        "".join(sample.model_dump_json() + "\n" for sample in samples), encoding="utf-8"
    )

    manifest = build_manifest(states)

    assert manifest["split"] == "val"
    assert manifest["test_assets_read"] is False
    assert manifest["record_count"] == 3
    assert manifest["operation_counts"] == {"ADD": 1, "DELETE": 1, "RESHAPE": 1}
    assert manifest["selection_contract"]["selection_uses_policy_outcome"] is False
    assert {row["target_edit"] for row in manifest["records"]} == {
        "ADD", "DELETE", "RESHAPE"
    }


def test_edit_only_manifest_rejects_missing_operation_support(tmp_path):
    states = tmp_path / "states.jsonl"
    states.write_text(_with_target(0, "ADD").model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks operation support"):
        build_manifest(states)
