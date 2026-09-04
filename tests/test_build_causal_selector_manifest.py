from activemap.features import EVIDENCE_DIM, HYPOTHESIS_DIM, STATE_DIM
from activemap.models import (
    CandidateHypothesis,
    EditOperation,
    EditRecord,
    EpisodeRecord,
    EvidenceItem,
)
from activemap.selector_records import SelectorSample
from scripts.build_causal_selector_manifest import TEMPORAL_CONTRACT, filter_sample


def test_causal_selector_filter_removes_future_candidate_and_recomputes_stop():
    evidence = [
        EvidenceItem(
            evidence_id="past",
            timestamp="2019_02",
            region=(0, 0, 8, 8),
            scale=1,
            image_path="past.tif",
            clear_fraction=1.0,
            cost=1.0,
        ),
        EvidenceItem(
            evidence_id="future",
            timestamp="2019_04",
            region=(0, 0, 8, 8),
            scale=1,
            image_path="future.tif",
            clear_fraction=1.0,
            cost=1.0,
        ),
    ]
    episode = EpisodeRecord(
        episode_id="episode-1",
        aoi_id="aoi-1",
        anchor_timestamp="2019_03",
        split="train",
        source_dataset="fixture",
        map_before="before.geojson",
        target_map="after.geojson",
        prior_geometry=None,
        target_geometry=None,
        hypothesis=CandidateHypothesis(
            op=EditOperation.KEEP,
            object_id="object-1",
            geometry=None,
            source="fixture",
        ),
        evidence_catalog=evidence,
        gt_edit=EditRecord(op=EditOperation.KEEP, object_id="object-1", geometry=None),
        is_synthetic=False,
        derivation_version="fixture",
    )
    sample = SelectorSample(
        sample_id="sample-1",
        split="train",
        edit_type=EditOperation.KEEP,
        hypothesis_features=[0.0] * HYPOTHESIS_DIM,
        state_features=[0.0] * STATE_DIM,
        evidence_ids=["past", "future"],
        evidence_features=[[0.0] * EVIDENCE_DIM for _ in range(2)],
        evidence_costs=[1.0, 1.0],
        false_edit_risks=[0.0, 0.0],
        oracle_utilities=[0.1, 0.9],
        stop_utility=0.2,
        metadata={"source_episode": "episode-1"},
    )

    filtered, audit = filter_sample(sample, episode)

    assert filtered is not None
    assert filtered.evidence_ids == ["past"]
    assert audit["before_action"] == "ACQUIRE"
    assert audit["after_action"] == "STOP"
    assert filtered.metadata["candidate_temporal_contract"] == {
        "version": TEMPORAL_CONTRACT,
        "anchor_timestamp": "2019_03",
        "candidate_count_before": 2,
        "candidate_count_after": 1,
        "future_candidate_count": 1,
    }
