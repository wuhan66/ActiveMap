import pytest
import torch

from activemap.agent.evidence_value_head import (
    CANDIDATE_DIM,
    CONTEXT_DIM,
    MASK_FEATURE_SET,
    POLICY_RELATIVE_MASK_CANDIDATE_DIM,
    EvidenceValueHead,
    EvidenceValueHeadConfig,
    EvidenceValueNormalizer,
    EvidenceValuePredictor,
    StructuredMapActionPredictor,
    TopKSetEvidenceValueConfig,
    TopKSetEvidenceValueHead,
    TopKSetEvidenceValuePredictor,
    evidence_value_features,
    executable_value_example,
    load_evidence_value_predictor,
    risk_adjusted_scores,
    topk_set_reranker_scores,
)
from activemap.features import ONLINE_OBSERVABLE_STATE_CONTRACT
from activemap.selector_records import SelectorSample
from scripts.train_evidence_value_head import collate_examples, evidence_value_loss


def _sample(split: str = "train") -> SelectorSample:
    evidence_ids = ["good", "bad"]
    return SelectorSample(
        sample_id="example",
        split=split,
        edit_type="ADD",
        hypothesis_features=[0.0] * 16,
        state_features=[0.0] * 8,
        evidence_ids=evidence_ids,
        evidence_features=[[0.0] * 13, [1.0] * 13],
        evidence_costs=[1.0, 2.0],
        false_edit_risks=[0.0, 1.0],
        oracle_utilities=[0.4, -0.2],
        stop_utility=0.1,
        metadata={
            "utility_mode": "executable",
            "gt_edit": "ADD",
            "executable_outcomes": {
                "good": {
                    "quality_gain": 0.5,
                    "false_edit": False,
                    "wrong_edit": False,
                    "missed_edit": False,
                },
                "bad": {
                    "quality_gain": -0.1,
                    "false_edit": True,
                    "wrong_edit": False,
                    "missed_edit": True,
                },
            },
            "evidence_predictions": {
                evidence_id: {
                    "edit_probabilities": [0.1, 0.7, 0.1, 0.1],
                    "gated_edit": "ADD",
                    "confidence": 0.8,
                    "geometry_delta": [0.0] * 8,
                    "mask_features": [float(index) / 10.0 for index in range(10)],
                }
                for evidence_id in evidence_ids
            },
            "mask_feature_contract": {
                "schema_version": "target-free-mask-features-v2",
                "feature_names": [f"feature-{index}" for index in range(10)],
                "target_free": True,
                "scope": "candidate_local_grid",
            },
        },
    )


def test_executable_example_has_structured_targets():
    row = executable_value_example(_sample())
    assert row["context"].shape == (CONTEXT_DIM,)
    assert row["candidates"].shape == (2, CANDIDATE_DIM)
    assert row["utility_gains"].tolist() == pytest.approx([0.3, -0.3])
    assert row["beneficial"].tolist() == [1.0, 0.0]
    assert row["unsafe"].tolist() == [0.0, 1.0]
    assert row["missed"].tolist() == [0.0, 1.0]
    assert row["terminal_target"] == 1


def test_executable_example_refuses_test_data():
    with pytest.raises(ValueError, match="test samples"):
        executable_value_example(_sample("test"))


def test_deployable_features_do_not_require_outcome_labels():
    sample = _sample("test")
    metadata = dict(sample.metadata)
    metadata.pop("executable_outcomes")
    context, candidates = evidence_value_features(
        sample.model_copy(update={"metadata": metadata})
    )
    assert context.shape == (CONTEXT_DIM,)
    assert candidates.shape == (2, CANDIDATE_DIM)


def test_policy_relative_mask_features_match_audited_dimension():
    sample = _sample("test")
    metadata = dict(sample.metadata)
    metadata.pop("executable_outcomes")
    context, candidates = evidence_value_features(
        sample.model_copy(update={"metadata": metadata}),
        feature_set=MASK_FEATURE_SET,
    )
    assert context.shape == (CONTEXT_DIM,)
    assert candidates.shape == (2, POLICY_RELATIVE_MASK_CANDIDATE_DIM)
    assert candidates[0, 28:32].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert candidates[0, 46:56].tolist() == pytest.approx(
        [float(index) / 10.0 for index in range(10)]
    )


def test_policy_relative_mask_features_require_target_free_contract():
    sample = _sample()
    metadata = dict(sample.metadata)
    metadata["mask_feature_contract"] = {
        **metadata["mask_feature_contract"],
        "target_free": False,
    }
    with pytest.raises(ValueError, match="target-free provenance"):
        evidence_value_features(
            sample.model_copy(update={"metadata": metadata}),
            feature_set=MASK_FEATURE_SET,
        )


def test_deployable_features_support_terminal_state_without_candidates():
    sample = _sample().model_copy(
        update={
            "evidence_ids": [],
            "evidence_features": [],
            "evidence_costs": [],
            "false_edit_risks": [],
            "oracle_utilities": [],
        }
    )
    context, candidates = evidence_value_features(sample)
    assert context.shape == (CONTEXT_DIM,)
    assert candidates.shape == (0, CANDIDATE_DIM)


def test_value_head_supports_variable_candidate_axis_and_backward():
    rows = [executable_value_example(_sample()), executable_value_example(_sample())]
    rows[1]["candidates"] = rows[1]["candidates"][:1]
    for name in ("utility_gains", "quality_gains", "beneficial", "unsafe", "missed"):
        rows[1][name] = rows[1][name][:1]
    batch = collate_examples(rows)
    model = EvidenceValueHead(EvidenceValueHeadConfig(hidden_dim=16))
    outputs = model(batch["context"], batch["candidates"])
    assert outputs["utility"].shape == (2, 2)
    loss, components = evidence_value_loss(outputs, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())


def test_value_head_positive_weights_increase_rare_positive_losses():
    batch = collate_examples([executable_value_example(_sample())])
    model = EvidenceValueHead(EvidenceValueHeadConfig(hidden_dim=16))
    outputs = model(batch["context"], batch["candidates"])
    _, baseline = evidence_value_loss(outputs, batch)
    _, weighted = evidence_value_loss(
        outputs,
        batch,
        positive_weight=8.0,
        beneficial_positive_weight=8.0,
    )
    assert weighted["utility"] != baseline["utility"]
    assert weighted["beneficial"] > baseline["beneficial"]


def test_pairwise_loss_prefers_positive_candidate_above_nonpositive():
    batch = collate_examples([executable_value_example(_sample())])
    model = EvidenceValueHead(EvidenceValueHeadConfig(hidden_dim=16))
    outputs = model(batch["context"], batch["candidates"])
    corrected = {name: value.clone() for name, value in outputs.items()}
    reversed_order = {name: value.clone() for name, value in outputs.items()}
    corrected["utility"] = torch.tensor([[1.0, -1.0]])
    reversed_order["utility"] = torch.tensor([[-1.0, 1.0]])
    _, corrected_components = evidence_value_loss(
        corrected, batch, pairwise_weight=1.0
    )
    _, reversed_components = evidence_value_loss(
        reversed_order, batch, pairwise_weight=1.0
    )
    assert corrected_components["pairwise"] < reversed_components["pairwise"]


def test_terminal_head_can_train_without_candidate_rows():
    model = EvidenceValueHead(
        EvidenceValueHeadConfig(hidden_dim=16, terminal_classes=4)
    )
    logits = model.terminal_logits(torch.zeros(3, CONTEXT_DIM))
    assert logits.shape == (3, 4)
    F = torch.nn.functional
    loss = F.cross_entropy(logits, torch.tensor([0, 1, 2]))
    loss.backward()
    assert torch.isfinite(loss)


def test_risk_adjustment_penalizes_unsafe_candidate():
    outputs = {
        "utility": torch.tensor([[0.2, 0.2]]),
        "unsafe_logit": torch.tensor([[-5.0, 5.0]]),
        "missed_logit": torch.zeros(1, 2),
    }
    scores = risk_adjusted_scores(
        outputs, unsafe_weight=0.2, missed_weight=0.0
    )
    assert scores[0, 0] > scores[0, 1]


def test_beneficial_head_contributes_to_deployed_score_when_enabled():
    outputs = {
        "utility": torch.zeros(1, 2),
        "beneficial_logit": torch.tensor([[-5.0, 5.0]]),
        "unsafe_logit": torch.zeros(1, 2),
        "missed_logit": torch.zeros(1, 2),
    }
    scores = risk_adjusted_scores(
        outputs,
        beneficial_weight=0.1,
        unsafe_weight=0.0,
        missed_weight=0.0,
    )
    assert scores[0, 1] > scores[0, 0]


def test_topk_set_reranker_supports_variable_candidates_and_backward():
    rows = [executable_value_example(_sample()), executable_value_example(_sample())]
    rows[1]["candidates"] = rows[1]["candidates"][:1]
    for name in ("utility_gains", "quality_gains", "beneficial", "unsafe", "missed"):
        rows[1][name] = rows[1][name][:1]
    batch = collate_examples(rows)
    model = TopKSetEvidenceValueHead(
        TopKSetEvidenceValueConfig(hidden_dim=16, top_k=2)
    )
    outputs = model(batch["context"], batch["candidates"], batch["mask"])
    scores = topk_set_reranker_scores(
        outputs,
        beneficial_weight=0.1,
        beneficial_probability_threshold=0.0,
        unsafe_weight=0.1,
        missed_weight=0.05,
    )
    assert scores.shape == (2, 2)
    assert outputs["shortlist_indices"].shape == (2, 2)
    outputs["reranker"]["utility"].sum().backward()
    assert any(parameter.grad is not None for parameter in model.reranker.parameters())


def test_topk_set_reranker_filters_low_benefit_candidate():
    outputs = {
        "proposer_scores": torch.tensor([[0.3, 0.2, 0.1]]),
        "shortlist_indices": torch.tensor([[0, 2]]),
        "shortlist_mask": torch.tensor([[True, True]]),
        "reranker": {
            "utility": torch.tensor([[0.8, 0.4]]),
            "beneficial_logit": torch.tensor([[-2.0, 2.0]]),
            "unsafe_logit": torch.zeros(1, 2),
            "missed_logit": torch.zeros(1, 2),
        },
    }
    scores = topk_set_reranker_scores(
        outputs,
        beneficial_weight=0.0,
        beneficial_probability_threshold=0.5,
        unsafe_weight=0.0,
        missed_weight=0.0,
    )
    assert scores[0, 0] < -1e3
    assert scores[0, 1] < -1e3
    assert scores[0, 2] == pytest.approx(0.4)


def test_predictor_appends_calibrated_stop_score(tmp_path):
    model = EvidenceValueHead(EvidenceValueHeadConfig(hidden_dim=16))
    checkpoint = tmp_path / "value.pt"
    normalizer = EvidenceValueNormalizer(
        context_mean=torch.zeros(CONTEXT_DIM).numpy(),
        context_std=torch.ones(CONTEXT_DIM).numpy(),
        candidate_mean=torch.zeros(CANDIDATE_DIM).numpy(),
        candidate_std=torch.ones(CANDIDATE_DIM).numpy(),
    )
    torch.save(
        {
            "protocol": "evidence_value_head_v1",
            "model_config": model.config.as_dict(),
            "state_dict": model.state_dict(),
            "normalizer": normalizer.as_dict(),
            "safety_margin": 0.125,
            "unsafe_penalty": 0.1,
            "missed_penalty": 0.05,
        },
        checkpoint,
    )
    scores = EvidenceValuePredictor(str(checkpoint))(_sample())
    assert scores.shape == (3,)
    assert scores[-1] == pytest.approx(0.125)


def test_predictor_restores_policy_relative_mask_feature_protocol(tmp_path):
    config = EvidenceValueHeadConfig(
        candidate_dim=POLICY_RELATIVE_MASK_CANDIDATE_DIM,
        hidden_dim=16,
    )
    model = EvidenceValueHead(config)
    checkpoint = tmp_path / "mask-value.pt"
    normalizer = EvidenceValueNormalizer(
        context_mean=torch.zeros(CONTEXT_DIM).numpy(),
        context_std=torch.ones(CONTEXT_DIM).numpy(),
        candidate_mean=torch.zeros(POLICY_RELATIVE_MASK_CANDIDATE_DIM).numpy(),
        candidate_std=torch.ones(POLICY_RELATIVE_MASK_CANDIDATE_DIM).numpy(),
    )
    torch.save(
        {
            "protocol": "evidence_value_head_v1",
            "feature_set": MASK_FEATURE_SET,
            "model_config": config.as_dict(),
            "state_dict": model.state_dict(),
            "normalizer": normalizer.as_dict(),
            "safety_margin": 0.125,
        },
        checkpoint,
    )
    predictor = EvidenceValuePredictor(str(checkpoint))
    assert predictor.feature_set == MASK_FEATURE_SET
    assert predictor.score_sample(_sample()).shape == (2,)


def test_topk_predictor_maps_shortlist_back_to_candidate_axis(tmp_path):
    config = TopKSetEvidenceValueConfig(hidden_dim=16, top_k=1)
    model = TopKSetEvidenceValueHead(config)
    checkpoint = tmp_path / "topk.pt"
    normalizer = EvidenceValueNormalizer(
        context_mean=torch.zeros(CONTEXT_DIM).numpy(),
        context_std=torch.ones(CONTEXT_DIM).numpy(),
        candidate_mean=torch.zeros(CANDIDATE_DIM).numpy(),
        candidate_std=torch.ones(CANDIDATE_DIM).numpy(),
    )
    torch.save(
        {
            "protocol": "topk_set_evidence_reranker_v1",
            "model_config": config.as_dict(),
            "state_dict": model.state_dict(),
            "normalizer": normalizer.as_dict(),
            "safety_margin": 0.25,
            "stop_margin": 0.2,
            "data_contract": ONLINE_OBSERVABLE_STATE_CONTRACT,
            "beneficial_score_weight": 0.1,
            "beneficial_probability_threshold": 0.0,
            "unsafe_penalty": 0.1,
            "missed_penalty": 0.05,
        },
        checkpoint,
    )
    predictor = TopKSetEvidenceValuePredictor(str(checkpoint))
    scores = predictor.score_sample(_sample())
    assert scores.shape == (2,)
    assert (scores > -1e3).sum() == 1
    assert predictor.action_scores(_sample()).shape == (3,)
    assert predictor.action_scores(_sample())[-1] == pytest.approx(0.2)
    assert predictor.data_contract == ONLINE_OBSERVABLE_STATE_CONTRACT
    loaded = load_evidence_value_predictor(str(checkpoint))
    assert isinstance(loaded, TopKSetEvidenceValuePredictor)
    overridden = load_evidence_value_predictor(str(checkpoint), stop_margin_override=0.4)
    assert overridden.action_scores(_sample())[-1] == pytest.approx(0.4)


def test_structured_predictor_returns_terminal_edit(tmp_path):
    config = EvidenceValueHeadConfig(hidden_dim=16, terminal_classes=4)
    model = EvidenceValueHead(config)
    with torch.no_grad():
        model.terminal_head[-1].weight.zero_()
        model.terminal_head[-1].bias.copy_(torch.tensor([0.0, 3.0, 0.0, 0.0]))
    checkpoint = tmp_path / "structured.pt"
    normalizer = EvidenceValueNormalizer(
        context_mean=torch.zeros(CONTEXT_DIM).numpy(),
        context_std=torch.ones(CONTEXT_DIM).numpy(),
        candidate_mean=torch.zeros(CANDIDATE_DIM).numpy(),
        candidate_std=torch.ones(CANDIDATE_DIM).numpy(),
    )
    torch.save(
        {
            "protocol": "structured_map_action_policy_v1",
            "model_config": config.as_dict(),
            "state_dict": model.state_dict(),
            "normalizer": normalizer.as_dict(),
            "safety_margin": 0.25,
        },
        checkpoint,
    )
    predictor = StructuredMapActionPredictor(str(checkpoint))
    scores = predictor(_sample())
    assert scores.shape == (3,)
    assert scores[-1] == pytest.approx(0.25)
    assert predictor.last_terminal_edit.value == "ADD"
