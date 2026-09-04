import pytest
import torch

from activemap.agent.records import AgentBelief
from activemap.agent.tool_belief_model import (
    ANCHORED_BELIEF_FEATURE_DIM,
    BELIEF_FEATURE_DIM,
    FrozenPriorSequentialPairedToolBeliefUpdater,
    IdentitySequentialPairedToolBeliefUpdater,
    LearnedToolBeliefUpdater,
    PairedToolBeliefUpdater,
    SequentialPairedToolBeliefUpdater,
    ToolBeliefResidualNetwork,
    ToolPairBeliefResidualNetwork,
    encode_belief,
)
from activemap.agent.tool_features import TOOL_RESULT_FEATURE_DIM
from activemap.geo_tools.records import GeoToolName, GeoToolResult


def _belief() -> AgentBelief:
    return AgentBelief(
        edit_probabilities=[0.7, 0.1, 0.1, 0.1],
        confidence=0.6,
        uncertainty=0.65,
        geometry_delta=[0.0] * 8,
    )


def test_tool_belief_network_contract() -> None:
    model = ToolBeliefResidualNetwork(hidden_dim=16, dropout=0.0)
    features = torch.zeros((3, BELIEF_FEATURE_DIM + TOOL_RESULT_FEATURE_DIM))

    edit_delta, confidence_delta, geometry_delta = model(features)

    assert edit_delta.shape == (3, 4)
    assert confidence_delta.shape == (3,)
    assert geometry_delta.shape == (3, 8)
    assert len(encode_belief(_belief())) == BELIEF_FEATURE_DIM


def test_tool_pair_belief_network_contract() -> None:
    model = ToolPairBeliefResidualNetwork(hidden_dim=16, dropout=0.0)
    features = torch.zeros((3, BELIEF_FEATURE_DIM + 2 * TOOL_RESULT_FEATURE_DIM))

    edit_delta, confidence_delta, geometry_delta = model(features)

    assert edit_delta.shape == (3, 4)
    assert confidence_delta.shape == (3,)
    assert geometry_delta.shape == (3, 8)

    anchored = ToolPairBeliefResidualNetwork(
        hidden_dim=16,
        dropout=0.0,
        belief_feature_dim=ANCHORED_BELIEF_FEATURE_DIM,
    )
    anchored_features = torch.zeros(
        (3, ANCHORED_BELIEF_FEATURE_DIM + 2 * TOOL_RESULT_FEATURE_DIM)
    )
    assert anchored(anchored_features)[0].shape == (3, 4)


def test_reliability_gate_conservatively_scales_all_residuals() -> None:
    model = ToolPairBeliefResidualNetwork(
        hidden_dim=16, dropout=0.0, reliability_gate=True, gate_bias=-2.0
    )
    features = torch.zeros((3, BELIEF_FEATURE_DIM + 2 * TOOL_RESULT_FEATURE_DIM))

    residuals = model(features)

    assert len(residuals) == 4
    assert residuals[3].shape == (3,)
    assert torch.allclose(residuals[3], torch.full((3,), torch.sigmoid(torch.tensor(-2.0))))


def test_zero_residual_preserves_probabilities_and_geometry() -> None:
    model = ToolBeliefResidualNetwork(hidden_dim=16, dropout=0.0)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    updater = LearnedToolBeliefUpdater(model)
    belief = _belief()
    result = GeoToolResult(
        call_id="quality-1",
        tool=GeoToolName.IMAGE_QUALITY,
        success=True,
        outputs={"valid_fraction": 0.9},
        cost=0.1,
    )

    updated = updater.update(belief, result)

    assert updated.edit_probabilities == pytest.approx(belief.edit_probabilities)
    assert updated.confidence == pytest.approx(belief.confidence)
    assert updated.geometry_delta == pytest.approx(belief.geometry_delta)
    assert sum(updated.edit_probabilities) == pytest.approx(1.0)
    assert updated.recommended_edit is None


def test_learned_update_is_bounded_and_returns_valid_belief() -> None:
    torch.manual_seed(7)
    updater = LearnedToolBeliefUpdater(
        ToolBeliefResidualNetwork(hidden_dim=16, dropout=0.0),
        max_logit_delta=1.0,
        geometry_scale=0.1,
    )
    result = GeoToolResult(
        call_id="change-1",
        tool=GeoToolName.TEMPORAL_CHANGE,
        success=True,
        outputs={"changed_fraction": 0.4},
        cost=0.1,
    )

    updated = updater.update(_belief(), result)

    assert sum(updated.edit_probabilities) == pytest.approx(1.0)
    assert 0.0 <= updated.confidence <= 1.0
    assert 0.0 <= updated.uncertainty <= 1.0
    assert all(abs(value) <= 0.1 for value in updated.geometry_delta)


def test_checkpoint_loader_restores_architecture_and_bounds(tmp_path) -> None:
    model = ToolBeliefResidualNetwork(hidden_dim=12, dropout=0.0)
    checkpoint = tmp_path / "tool-belief.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": 12,
            "dropout": 0.0,
            "max_logit_delta": 1.5,
            "geometry_scale": 0.2,
            "belief_feature_dim": BELIEF_FEATURE_DIM,
            "tool_result_feature_dim": TOOL_RESULT_FEATURE_DIM,
        },
        checkpoint,
    )

    updater = LearnedToolBeliefUpdater.from_checkpoint(checkpoint)

    assert updater.model.backbone[0].out_features == 12
    assert updater.max_logit_delta == pytest.approx(1.5)
    assert updater.geometry_scale == pytest.approx(0.2)


def test_paired_checkpoint_requires_explicit_model_kind(tmp_path) -> None:
    model = ToolPairBeliefResidualNetwork(hidden_dim=12, dropout=0.0)
    checkpoint = tmp_path / "paired-tool-belief.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kind": "paired_quality_temporal",
            "hidden_dim": 12,
            "dropout": 0.0,
            "max_logit_delta": 1.5,
            "geometry_scale": 0.2,
            "belief_feature_dim": BELIEF_FEATURE_DIM,
            "tool_result_feature_dim": TOOL_RESULT_FEATURE_DIM,
        },
        checkpoint,
    )

    updater = PairedToolBeliefUpdater.from_checkpoint(checkpoint)

    assert updater.model.backbone[0].out_features == 12
    assert updater.max_logit_delta == pytest.approx(1.5)
    assert updater.geometry_scale == pytest.approx(0.2)


def test_paired_checkpoint_restores_reliability_gate(tmp_path) -> None:
    model = ToolPairBeliefResidualNetwork(
        hidden_dim=12, dropout=0.0, reliability_gate=True, gate_bias=-2.0
    )
    checkpoint = tmp_path / "gated-paired-tool-belief.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kind": "post_acquisition_paired_quality_temporal",
            "hidden_dim": 12,
            "dropout": 0.0,
            "max_logit_delta": 1.5,
            "geometry_scale": 0.2,
            "belief_feature_dim": BELIEF_FEATURE_DIM,
            "tool_result_feature_dim": TOOL_RESULT_FEATURE_DIM,
            "reliability_gate": True,
            "gate_bias": -2.0,
        },
        checkpoint,
    )

    updater = PairedToolBeliefUpdater.from_checkpoint(checkpoint)

    assert updater.model.reliability_gate is True
    assert hasattr(updater.model, "reliability_head")


def test_sequential_paired_updater_waits_for_matching_quality_context() -> None:
    class RecordingPairUpdater:
        def __init__(self) -> None:
            self.calls = []

        def update_pair(self, belief, quality, temporal, *, use_quality=True):
            self.calls.append((quality, temporal, use_quality))
            return belief.model_copy(update={"confidence": 0.9})

    pair = RecordingPairUpdater()
    updater = SequentialPairedToolBeliefUpdater(pair)  # type: ignore[arg-type]
    belief = _belief()
    quality = GeoToolResult(
        call_id="quality-e1",
        tool=GeoToolName.IMAGE_QUALITY,
        success=True,
        outputs={"evidence_id": "e1", "valid_fraction": 0.9},
        cost=0.03,
    )
    wrong_temporal = GeoToolResult(
        call_id="temporal-e2",
        tool=GeoToolName.TEMPORAL_CHANGE,
        success=True,
        outputs={"evidence_id": "e2", "changed_fraction": 0.4},
        cost=0.15,
    )
    temporal = wrong_temporal.model_copy(
        update={
            "call_id": "temporal-e1",
            "outputs": {"evidence_id": "e1", "changed_fraction": 0.4},
        }
    )

    after_quality = updater.update(belief, quality)
    after_wrong_temporal = updater.update(after_quality, wrong_temporal)
    updated = updater.update(after_wrong_temporal, temporal)

    assert after_quality == belief
    assert after_wrong_temporal == belief
    assert updated.confidence == pytest.approx(0.9)
    assert len(pair.calls) == 1
    assert updater.matched_pairs == 1
    assert updater.unmatched_temporal == 1


def test_frozen_prior_ablation_does_not_accumulate_previous_pair() -> None:
    class RecordingPairUpdater:
        def __init__(self) -> None:
            self.beliefs: list[AgentBelief] = []

        def update_pair(self, belief, quality, temporal, *, use_quality=True):
            self.beliefs.append(belief)
            return belief.model_copy(
                update={"confidence": min(1.0, belief.confidence + 0.1)}
            )

    pair = RecordingPairUpdater()
    updater = FrozenPriorSequentialPairedToolBeliefUpdater(
        pair  # type: ignore[arg-type]
    )
    belief = _belief()
    for evidence_id in ("e1", "e2"):
        quality = GeoToolResult(
            call_id=f"quality-{evidence_id}",
            tool=GeoToolName.IMAGE_QUALITY,
            success=True,
            outputs={"evidence_id": evidence_id, "valid_fraction": 0.9},
            cost=0.03,
        )
        temporal = GeoToolResult(
            call_id=f"temporal-{evidence_id}",
            tool=GeoToolName.TEMPORAL_CHANGE,
            success=True,
            outputs={"evidence_id": evidence_id, "changed_fraction": 0.4},
            cost=0.15,
        )
        belief = updater.update(belief, quality)
        belief = updater.update(belief, temporal)

    assert len(pair.beliefs) == 2
    assert pair.beliefs[0].confidence == pair.beliefs[1].confidence == 0.6
    assert belief.confidence == 0.7


def test_identity_belief_ablation_pairs_tools_without_updating_belief() -> None:
    class FailingPairUpdater:
        def update_pair(self, *args, **kwargs):
            raise AssertionError("identity ablation must not invoke learned updater")

    updater = IdentitySequentialPairedToolBeliefUpdater(
        FailingPairUpdater()  # type: ignore[arg-type]
    )
    belief = _belief()
    quality = GeoToolResult(
        call_id="quality-e1",
        tool=GeoToolName.IMAGE_QUALITY,
        success=True,
        outputs={"evidence_id": "e1"},
        cost=0.03,
    )
    temporal = GeoToolResult(
        call_id="temporal-e1",
        tool=GeoToolName.TEMPORAL_CHANGE,
        success=True,
        outputs={"evidence_id": "e1"},
        cost=0.15,
    )

    assert updater.update(belief, quality) is belief
    assert updater.update(belief, temporal) is belief
    assert updater.matched_pairs == 1
