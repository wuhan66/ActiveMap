import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("shapely")

from activemap.models import (
    CandidateHypothesis,
    EditOperation,
    EditRecord,
    EpisodeRecord,
    EvidenceItem,
    GeoJSONGeometry,
)
from activemap.oracle.updater_counterfactual import (
    MASK_FEATURE_NAMES,
    MASK_FEATURES_SCHEMA,
    _target_free_mask_features,
)
from scripts import evaluate_online_full_controller as online


def polygon(x0, x1):
    return GeoJSONGeometry(
        type="Polygon",
        coordinates=[[[x0, 0], [x1, 0], [x1, 1], [x0, 1], [x0, 0]]],
    )


def make_episode():
    target = polygon(0, 2)
    return EpisodeRecord(
        episode_id="online-full-controller-fixture",
        aoi_id="aoi-fixture",
        anchor_timestamp="2018_02",
        split="val",
        source_dataset="fixture",
        map_before="before.geojson",
        target_map="after.geojson",
        prior_geometry=polygon(0, 1),
        target_geometry=target,
        hypothesis=CandidateHypothesis(
            op=EditOperation.ADD,
            object_id="road-1",
            geometry=target,
            source="fixture",
        ),
        evidence_catalog=[
            EvidenceItem(
                evidence_id="anchor",
                timestamp="2018_02",
                region=(0, 0, 8, 8),
                scale=1,
                image_path="anchor.jpg",
                clear_fraction=1.0,
                cost=0.2,
            ),
            EvidenceItem(
                evidence_id="later",
                timestamp="2018_03",
                region=(0, 0, 8, 8),
                scale=1,
                image_path="later.jpg",
                clear_fraction=1.0,
                cost=0.3,
            ),
        ],
        gt_edit=EditRecord(
            op=EditOperation.ADD,
            object_id="road-1",
            geometry=target,
        ),
        is_synthetic=False,
        derivation_version="fixture",
    )


class FakePredictor:
    model = SimpleNamespace(config=SimpleNamespace(image_channels=3))

    def predict(self, _image, prior):
        value = float(np.mean(prior))
        return {
            "mask_probability": np.full((8, 8), 0.85 - 0.1 * value, dtype=np.float32),
            "edit_probabilities": np.asarray([0.1, 0.8, 0.05, 0.05]),
            "confidence": 0.9 - 0.1 * value,
            "geometry_delta": np.zeros(8, dtype=np.float32),
        }


def test_always_reject_policy_never_commits_a_direct_edit():
    action = online.AlwaysRejectPolicy().act(None)

    assert action.action.value == "REJECT"


def test_safe_commit_variants_reuse_the_identical_precommit_policy():
    assert online._base_policy_name("direct_current_hypothesis_safe") == "direct_current_hypothesis"
    assert online._base_policy_name("active_forced_safe") == "active_forced"
    assert online._base_policy_name("active_selective_safe") == "active_selective"
    assert "direct_current_hypothesis_safe" in online.SAFE_COMMIT_POLICIES


def test_true_online_controller_rejects_legacy_selector_contract():
    with pytest.raises(ValueError, match="online-observable-state-v1"):
        online.require_online_observable_selector(SimpleNamespace(data_contract=None))


def test_online_writeback_must_match_the_frozen_registry_contract():
    assets = SimpleNamespace(updater_threshold=0.5, delta_margin=0.15)

    online.require_registry_writeback_protocol(
        threshold=0.5,
        delta_margin=0.15,
        assets=assets,
    )
    with pytest.raises(ValueError, match="threshold must match"):
        online.require_registry_writeback_protocol(
            threshold=0.4,
            delta_margin=0.15,
            assets=assets,
        )
    with pytest.raises(ValueError, match="delta margin must match"):
        online.require_registry_writeback_protocol(
            threshold=0.5,
            delta_margin=0.0,
            assets=assets,
        )


def test_parser_accepts_train_tool_gate_diagnostic(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_online_full_controller.py",
            str(tmp_path / "episodes.jsonl"),
            str(tmp_path / "registry.yaml"),
            "20260902",
            str(tmp_path / "output"),
            "--storage-root",
            str(tmp_path),
            "--split",
            "train",
            "--diagnostic-tool-gate-threshold-override",
            "0.125",
            "--recurrent-safe-commit",
            "--chain-start-index",
            "20",
        ],
    )

    args = online.parse_args()

    assert args.split == "train"
    assert args.diagnostic_tool_gate_threshold_override == pytest.approx(0.125)
    assert args.recurrent_safe_commit is True
    assert args.chain_start_index == 20


def test_recurrent_safe_commit_blocks_unescalated_immediate_retry():
    gate = online.SafeCommitGate(0.68, 0.99)
    memory = online.RecurrentSafeCommitMemory()
    rejected_metrics = {
        "writeback_changed": True,
        "fused_confidence": 0.670591,
        "vector_replay_iou": 1.0,
        "vector_delta_topology_valid": True,
    }

    accepted, first = online.decide_safe_commit(
        rejected_metrics,
        gate,
        memory,
        operation=EditOperation.ADD,
        step=0,
        prior_hash="unchanged-prior",
        selected_evidence_ids=["direct-0", "aux-1"],
        direct_evidence_id="direct-0",
        tool_calls=2,
        retry_confidence_margin=0.05,
    )
    assert accepted is False
    assert first["safe_commit_decision_reason"] == "base_gate_rejected"

    retry_metrics = {**rejected_metrics, "fused_confidence": 0.682721}
    accepted, retry = online.decide_safe_commit(
        retry_metrics,
        gate,
        memory,
        operation=EditOperation.ADD,
        step=1,
        prior_hash="unchanged-prior",
        selected_evidence_ids=["direct-1", "aux-1"],
        direct_evidence_id="direct-1",
        tool_calls=2,
        retry_confidence_margin=0.05,
    )
    assert accepted is False
    assert retry["safe_commit_immediate_retry"] is True
    assert retry["safe_commit_retry_blocked"] is True
    assert retry["safe_commit_decision_reason"] == "retry_hysteresis_blocked"


def test_recurrent_safe_commit_allows_retry_with_new_auxiliary_evidence():
    gate = online.SafeCommitGate(0.68, 0.99)
    memory = online.RecurrentSafeCommitMemory(
        last_step=0,
        last_operation=EditOperation.ADD,
        last_accepted=False,
        last_confidence=0.67,
        last_prior_hash="unchanged-prior",
        last_auxiliary_evidence_ids=frozenset({"aux-1"}),
        last_tool_calls=2,
    )
    metrics = {
        "writeback_changed": True,
        "fused_confidence": 0.69,
        "vector_replay_iou": 1.0,
        "vector_delta_topology_valid": True,
    }

    accepted, decision = online.decide_safe_commit(
        metrics,
        gate,
        memory,
        operation=EditOperation.ADD,
        step=1,
        prior_hash="unchanged-prior",
        selected_evidence_ids=["direct-1", "aux-1", "aux-2"],
        direct_evidence_id="direct-1",
        tool_calls=2,
        retry_confidence_margin=0.05,
    )

    assert accepted is True
    assert decision["safe_commit_new_auxiliary_evidence"] is True
    assert decision["safe_commit_decision_reason"] == "retry_evidence_escalated"


def test_parser_rejects_test_split(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_online_full_controller.py",
            str(tmp_path / "episodes.jsonl"),
            str(tmp_path / "registry.yaml"),
            "20260902",
            str(tmp_path / "output"),
            "--storage-root",
            str(tmp_path),
            "--split",
            "test",
        ],
    )

    with pytest.raises(SystemExit):
        online.parse_args()


def test_runtime_catalog_reruns_target_free_frontend_for_each_carried_prior(monkeypatch):
    episode = make_episode()
    target_arguments = []

    def fake_inputs(_episode, item, prior_geometry, **kwargs):
        target_arguments.append(kwargs["target_geometry"])
        prior_value = 0.0 if prior_geometry is None else min(prior_geometry.area / 2.0, 1.0)
        return (
            np.zeros((3, 8, 8), dtype=np.float32),
            np.full((8, 8), prior_value, dtype=np.float32),
            np.zeros((8, 8), dtype=np.float32),
            np.ones((8, 8), dtype=np.float32),
            (8, 8),
        )

    monkeypatch.setattr(online, "_runtime_candidate_inputs", fake_inputs)
    predictor = FakePredictor()
    first = online.build_runtime_catalog(
        episode,
        online._episode_geometry(episode.prior_geometry),
        predictor,
        image_size=8,
        budget=3.0,
    )
    carried = online._episode_geometry(episode.target_geometry)
    second = online.build_runtime_catalog(
        episode, carried, predictor, image_size=8, budget=3.0
    )

    assert target_arguments == [None, None, None, None]
    assert first.prior_hash != second.prior_hash
    assert first.candidate_receipt_hash != second.candidate_receipt_hash
    assert len(first.prior_hash) == 64
    assert len(first.candidate_receipt_hash) == 64
    assert first.sample.metadata["runtime_target_free"] is True
    assert first.sample.metadata["gt_edit"] == EditOperation.KEEP.value
    assert first.sample.metadata["selected_evidence_ids"] == ["anchor"]
    assert first.sample.evidence_ids == ["later"]
    assert first.sample.oracle_utilities == [0.0]
    assert first.sample.hypothesis_features[12] == pytest.approx(
        online._prediction_entropy(np.asarray([0.1, 0.8, 0.05, 0.05]))
    )
    expected_fused_confidence = (
        0.5 * first.candidates["anchor"].prediction["confidence"] + 0.5
    )
    assert first.sample.hypothesis_features[13] == pytest.approx(expected_fused_confidence)
    assert first.sample.state_features[7] == pytest.approx(expected_fused_confidence)
    assert first.sample.hypothesis_features[4:12] != second.sample.hypothesis_features[4:12]
    assert first.sample.metadata["mask_feature_contract"] == {
        "schema_version": MASK_FEATURES_SCHEMA,
        "feature_names": list(MASK_FEATURE_NAMES),
        "target_free": True,
        "scope": "candidate_local_grid",
    }
    first_prediction = first.sample.metadata["evidence_predictions"]["later"]
    second_prediction = second.sample.metadata["evidence_predictions"]["later"]
    assert first_prediction["mask_features"] == pytest.approx(
        _target_free_mask_features(
            first.candidates["later"].prediction["mask_probability"],
            first.candidates["later"].prior,
            first.candidates["later"].valid,
        )
    )
    assert first_prediction["mask_features"] != pytest.approx(
        second_prediction["mask_features"]
    )


def test_runtime_catalog_forwards_temporal_pair_contract(monkeypatch):
    episode = make_episode()
    temporal_arguments = []

    def fake_inputs(_episode, _item, _prior_geometry, **kwargs):
        temporal_arguments.append(kwargs["temporal_pair_input"])
        return (
            np.zeros((6, 8, 8), dtype=np.float32),
            np.zeros((8, 8), dtype=np.float32),
            np.zeros((8, 8), dtype=np.float32),
            np.ones((8, 8), dtype=np.float32),
            (8, 8),
        )

    predictor = FakePredictor()
    predictor.model.config.image_channels = 6
    predictor.model.config.temporal_pair_input = True
    monkeypatch.setattr(online, "_runtime_candidate_inputs", fake_inputs)
    online.build_runtime_catalog(
        episode,
        online._episode_geometry(episode.prior_geometry),
        predictor,
        image_size=8,
        budget=3.0,
    )

    assert temporal_arguments == [True, True]


def test_causal_runtime_catalog_excludes_future_evidence_before_inference(monkeypatch):
    episode = make_episode()
    inferred_ids = []

    def fake_inputs(_episode, item, _prior_geometry, **_kwargs):
        inferred_ids.append(item.evidence_id)
        return (
            np.zeros((3, 8, 8), dtype=np.float32),
            np.zeros((8, 8), dtype=np.float32),
            np.zeros((8, 8), dtype=np.float32),
            np.ones((8, 8), dtype=np.float32),
            (8, 8),
        )

    monkeypatch.setattr(online, "_runtime_candidate_inputs", fake_inputs)
    catalog = online.build_runtime_catalog(
        episode,
        online._episode_geometry(episode.prior_geometry),
        FakePredictor(),
        image_size=8,
        budget=3.0,
        causal_evidence_only=True,
    )

    assert inferred_ids == ["anchor"]
    assert catalog.excluded_future_evidence_ids == ("later",)
    assert catalog.sample.evidence_ids == []
    assert catalog.sample.metadata["runtime_temporally_causal"] is True
    assert catalog.sample.metadata["runtime_excluded_future_evidence_ids"] == [
        "later"
    ]


def test_runtime_selector_excludes_misaligned_candidates_without_using_targets():
    episode = make_episode()
    prediction = {
        "mask_probability": np.full((8, 8), 0.8, dtype=np.float32),
        "edit_probabilities": np.asarray([0.1, 0.8, 0.05, 0.05], dtype=np.float32),
        "confidence": 0.9,
        "geometry_delta": np.zeros(8, dtype=np.float32),
    }
    rows = [
        online.RuntimeCandidate(
            evidence_id="anchor",
            image=np.zeros((3, 8, 8), dtype=np.float32),
            prior=np.zeros((8, 8), dtype=np.float32),
            valid=np.ones((8, 8), dtype=np.float32),
            transform="shared-grid",
            raster_shape=(8, 8),
            prediction=prediction,
            clear_fraction=1.0,
        ),
        online.RuntimeCandidate(
            evidence_id="later",
            image=np.zeros((3, 8, 8), dtype=np.float32),
            prior=np.zeros((8, 8), dtype=np.float32),
            valid=np.ones((8, 8), dtype=np.float32),
            transform="shifted-grid",
            raster_shape=(8, 8),
            prediction=prediction,
            clear_fraction=1.0,
        ),
    ]

    sample, _, _ = online.build_runtime_selector_sample(
        episode,
        online._episode_geometry(episode.prior_geometry),
        rows,
        budget=3.0,
    )

    assert sample.evidence_ids == []
    assert sample.metadata["runtime_grid_alignment"] == {
        "direct_evidence_id": "anchor",
        "eligible_evidence_ids": [],
        "excluded_misaligned_evidence_ids": ["later"],
    }
    assert sample.metadata["runtime_terminal_only"] is True
    assert sample.metadata["runtime_target_free"] is True
    with pytest.raises(RuntimeError, match="terminal-only"):
        sample.target_index()


def test_carried_state_oracle_labels_runtime_features_after_target_free_inference(
    monkeypatch,
):
    episode = make_episode()

    def fake_inputs(_episode, item, _prior_geometry, **kwargs):
        is_candidate = item.evidence_id == "later"
        target = (
            np.ones((8, 8), dtype=np.float32)
            if kwargs["target_geometry"] is not None
            else np.zeros((8, 8), dtype=np.float32)
        )
        return (
            np.full((3, 8, 8), float(is_candidate), dtype=np.float32),
            np.zeros((8, 8), dtype=np.float32),
            target,
            np.ones((8, 8), dtype=np.float32),
            (8, 8),
        )

    class RuntimeOraclePredictor:
        model = SimpleNamespace(config=SimpleNamespace(image_channels=3))

        def predict(self, image, _prior):
            is_candidate = float(np.mean(image)) > 0.5
            return {
                "mask_probability": np.full(
                    (8, 8), 1.0 if is_candidate else 0.0, dtype=np.float32
                ),
                "edit_probabilities": np.asarray(
                    [0.01, 0.98, 0.005, 0.005]
                    if is_candidate
                    else [0.90, 0.08, 0.01, 0.01],
                    dtype=np.float32,
                ),
                "confidence": 0.99 if is_candidate else 0.20,
                "geometry_delta": np.zeros(8, dtype=np.float32),
            }

    monkeypatch.setattr(online, "_runtime_candidate_inputs", fake_inputs)
    prior_geometry = online._episode_geometry(episode.prior_geometry)
    catalog = online.build_runtime_catalog(
        episode,
        prior_geometry,
        RuntimeOraclePredictor(),
        image_size=8,
        budget=3.0,
    )

    labeled = online.build_carried_state_oracle_sample(
        catalog,
        episode,
        prior_geometry,
        output_split="train",
        policy_name="active_selective_safe",
        chain_index=2,
        step=1,
        budget=3.0,
        max_candidates=16,
        threshold=0.4,
        delta_margin=0.0,
        min_delta_component_pixels=0,
        safe_gate=None,
    )

    assert labeled is not None
    assert catalog.sample.metadata["gt_edit"] == EditOperation.KEEP.value
    assert labeled.split == "train"
    assert labeled.metadata["gt_edit"] == EditOperation.ADD.value
    assert labeled.metadata["runtime_oracle_contract"] == {
        "version": online.RUNTIME_ORACLE_CONTRACT,
        "policy": "active_selective_safe",
        "target_used_for_labels_only": True,
        "downstream": "belief_terminal_plus_safe_commit_without_tools",
        "chain_index": 2,
        "step": 1,
        "direct_outcome": labeled.metadata["runtime_oracle_contract"][
            "direct_outcome"
        ],
    }
    assert labeled.metadata["utility_mode"] == "executable"
    assert labeled.oracle_utilities[0] > labeled.stop_utility
    assert labeled.metadata["executable_outcomes"]["later"]["quality_gain"] > 0
    assert (
        labeled.metadata["executable_outcomes"]["later"]["raw_effective_operation"]
        == EditOperation.ADD.value
    )
