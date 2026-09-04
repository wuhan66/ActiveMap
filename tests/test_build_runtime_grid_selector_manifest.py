from activemap.models import EditOperation
from activemap.selector_records import SelectorSample
from scripts.build_runtime_grid_selector_manifest import filter_sample_to_support


def sample() -> SelectorSample:
    return SelectorSample(
        sample_id="grid-filter-fixture",
        split="train",
        edit_type=EditOperation.ADD,
        hypothesis_features=[0.0] * 16,
        state_features=[0.0] * 8,
        evidence_ids=["same-a", "other", "same-b"],
        evidence_features=[[0.1] * 13, [0.2] * 13, [0.3] * 13],
        evidence_costs=[0.2, 0.3, 0.4],
        false_edit_risks=[0.1, 0.2, 0.3],
        oracle_utilities=[0.1, 0.8, 0.4],
        metadata={"initial_evidence_id": "direct", "source_episode": "episode"},
    )


def test_filter_relabels_when_the_original_oracle_is_not_executable_on_grid():
    filtered, audit = filter_sample_to_support(
        sample(), eligible_ids={"same-a", "same-b"}, catalog_count=4
    )

    assert filtered.evidence_ids == ["same-a", "same-b"]
    assert filtered.oracle_utilities == [0.1, 0.4]
    assert filtered.target_index() == 1
    assert audit == {
        "candidate_count_before": 3,
        "candidate_count_after": 2,
        "target_before": "ACQUIRE",
        "target_after": "ACQUIRE",
        "target_id_preserved": False,
    }
    assert filtered.metadata["candidate_support_contract"] == {
        "version": "same-runtime-grid-v1",
        "catalog_candidate_count": 4,
        "eligible_catalog_candidate_count": 2,
        "excluded_catalog_candidate_count": 1,
        "initial_evidence_id": "direct",
    }
