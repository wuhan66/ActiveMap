import numpy as np
import pytest
import torch

from activemap.agent.active_catalog_candidate_ranker import (
    RANKER_FEATURE_NAMES,
    CandidateUtilityRanker,
    CandidateUtilityRankerConfig,
    observable_ranker_features,
    CandidateUtilityRankerPredictor,
)
from scripts.build_active_catalog_gate_sft import GATE_SYSTEM_PROMPT, convert_row
from scripts.diagnose_active_catalog_decisions import decompose
from scripts.evaluate_active_catalog_gate_vlm import parse_gate_output
from scripts.evaluate_active_catalog_ranker_guard import _decision
from scripts.train_active_catalog_candidate_ranker import (
    calibrate_margin,
    attach_online_event_utility,
    load_online_examples,
    ranker_loss,
)


def _state():
    return {
        "direct_draft": {"edit": "ADD", "confidence": 0.8},
        "belief": {"edit_probabilities": [0.1, 0.7, 0.1, 0.1], "uncertainty": 0.3},
        "budget": {"remaining": 3.0},
        "selected_evidence_ids": ["old"],
        "candidate_evidence": [],
    }


def test_observable_ranker_features_are_finite_and_fixed_width():
    candidate = {
        "evidence_id": "e1",
        "scale": 2,
        "clear_fraction": 0.75,
        "temporal_offset_normalized": -0.5,
        "cost": 1.5,
    }
    values = observable_ranker_features(_state(), candidate)
    assert values.shape == (len(RANKER_FEATURE_NAMES),)
    assert np.isfinite(values).all()


def test_candidate_ranker_supports_variable_candidate_axis():
    model = CandidateUtilityRanker(CandidateUtilityRankerConfig(hidden_dim=16))
    output = model(torch.zeros(2, 5, len(RANKER_FEATURE_NAMES)))
    assert output.shape == (2, 5)


def test_visual_candidate_ranker_requires_aligned_state_embedding(tmp_path):
    config = CandidateUtilityRankerConfig(
        input_dim=len(RANKER_FEATURE_NAMES) + 3, hidden_dim=8, dropout=0.0
    )
    model = CandidateUtilityRanker(config)
    checkpoint = tmp_path / "ranker.pt"
    torch.save(
        {
            "model_config": config.as_dict(),
            "state_dict": model.state_dict(),
            "feature_mean": np.zeros(config.input_dim),
            "feature_std": np.ones(config.input_dim),
        },
        checkpoint,
    )
    predictor = CandidateUtilityRankerPredictor(str(checkpoint))
    state = _state()
    state["candidate_evidence"] = [
        {
            "evidence_id": "e1",
            "scale": 2,
            "clear_fraction": 0.75,
            "temporal_offset_normalized": -0.5,
            "cost": 1.5,
        }
    ]
    with pytest.raises(ValueError):
        predictor.score_state(state)
    assert set(predictor.score_state(state, np.zeros(3))) == {"e1"}


def test_low_rank_visual_fusion_supports_candidate_axis():
    config = CandidateUtilityRankerConfig(
        input_dim=len(RANKER_FEATURE_NAMES) + 8,
        hidden_dim=16,
        dropout=0.0,
        fusion_type="low_rank",
        fusion_dim=4,
    )
    model = CandidateUtilityRanker(config)
    output = model(torch.zeros(2, 5, config.input_dim))
    assert output.shape == (2, 5)
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_decision_decomposition_attributes_candidate_regret():
    traces = [
        {
            "example_id": "a",
            "predicted_selection": "ACQUIRE",
            "predicted_evidence_id": "bad",
        },
        {"example_id": "b", "predicted_selection": "STOP", "predicted_evidence_id": None},
    ]
    evaluation = [
        {
            "example_id": "a",
            "stop_utility": 0.0,
            "candidates": [
                {"evidence_id": "good", "utility": 0.4},
                {"evidence_id": "bad", "utility": -0.1},
            ],
        },
        {
            "example_id": "b",
            "stop_utility": 0.0,
            "candidates": [{"evidence_id": "good", "utility": 0.2}],
        },
    ]
    report = decompose(traces, evaluation)
    assert report["regret"]["candidate_choice_regret"] == pytest.approx(0.5)
    assert report["regret"]["false_stop_regret"] == pytest.approx(0.2)
    assert report["counterfactual_mean_utility"]["vlm_gate_oracle_candidate"] == pytest.approx(
        0.2
    )


def test_ranker_loss_is_finite_with_padded_candidates():
    predicted = torch.tensor([[0.2, -0.1], [0.3, 0.0]], requires_grad=True)
    gains = torch.tensor([[0.4, -0.2], [-0.1, 0.0]])
    mask = torch.tensor([[True, True], [True, False]])
    loss = ranker_loss(
        predicted,
        gains,
        mask,
        temperature=0.1,
        regression_weight=1.0,
        listwise_weight=1.0,
        pairwise_weight=1.0,
        gate_weight=1.0,
        positive_weight=4.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(predicted.grad).all()


def test_margin_calibration_respects_false_call_constraint():
    examples = [
        {
            "gains": np.asarray([0.2]),
            "utilities": np.asarray([0.2]),
            "stop_utility": 0.0,
        },
        {
            "gains": np.asarray([-0.1]),
            "utilities": np.asarray([-0.1]),
            "stop_utility": 0.0,
        },
    ]
    margin, metrics = calibrate_margin(
        examples,
        [np.asarray([0.3]), np.asarray([0.05])],
        [0.0, 0.1],
        0.0,
    )
    assert 0.29 < margin < 0.3
    assert metrics["false_call_rate"] == 0.0


def test_hybrid_uses_vlm_gate_and_ranker_candidate():
    selection, evidence_id = _decision(
        "vlm_gate_ranker_candidate",
        scores={"weak": -0.1, "strong": 0.4},
        margin=0.1,
        vlm={"predicted_selection": "ACQUIRE", "predicted_evidence_id": "weak"},
    )
    assert selection == "ACQUIRE"
    assert evidence_id == "strong"


def test_gate_sft_removes_candidate_identity_from_target():
    row = {
        "example_id": "example",
        "split": "train",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "old"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/tmp/example.png"},
                    {"type": "text", "text": "{}"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": '{"stage":"SELECT","selection":"ACQUIRE",'
                        '"evidence_id":"secret-choice"}',
                    }
                ],
            },
        ],
    }
    converted = convert_row(row)
    assert converted["messages"][0]["content"][0]["text"] == GATE_SYSTEM_PROMPT
    assert converted["messages"][2]["content"][0]["text"] == '{"selection":"ACQUIRE"}'
    assert "secret-choice" not in converted["messages"][2]["content"][0]["text"]


def test_gate_output_parser_is_strict():
    assert parse_gate_output('{"selection":"STOP"}') == "STOP"
    with pytest.raises(ValueError):
        parse_gate_output('{"selection":"MAYBE"}')


def test_online_examples_align_trace_to_hidden_counterfactual_index(tmp_path):
    state = _state()
    state["candidate_evidence"] = [
        {
            "evidence_id": "e1",
            "scale": 2,
            "clear_fraction": 0.75,
            "temporal_offset_normalized": -0.5,
            "cost": 1.5,
        }
    ]
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(
        __import__("json").dumps(
            {
                "sample_id": "episode__s2",
                "source_episode": "episode",
                "aoi_id": "aoi",
                "budget": 3.0,
                "split": "train",
                "test_assets_read": False,
                "events": [{"observable_state": state}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "index.jsonl"
    index_path.write_text(
        __import__("json").dumps(
            {
                "source_episode": "episode",
                "aoi_id": "aoi",
                "budget": 3.0,
                "oracle_step": 2,
                "split": "train",
                "model_visible": False,
                "test_assets_read": False,
                "stop_utility": 0.1,
                "candidates": [{"evidence_id": "e1", "utility": 0.3, "cost": 1.5}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    examples = load_online_examples(trace_path, index_path, "train")
    assert len(examples) == 1
    assert examples[0]["gains"].tolist() == pytest.approx([0.2])
    assert examples[0]["features"].shape == (1, len(RANKER_FEATURE_NAMES))


def test_online_utility_is_appended_as_candidate_context(tmp_path):
    traces = tmp_path / "traces.jsonl"
    traces.write_text(
        __import__("json").dumps(
            {
                "sample_id": "sample",
                "events": [{"predicted_acquire_utility": -0.25}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = [{"example_id": "sample", "features": np.zeros((2, 3), dtype=np.float32)}]
    augmented = attach_online_event_utility(rows, traces)
    assert augmented[0]["features"].shape == (2, 4)
    assert augmented[0]["features"][:, -1].tolist() == [-0.25, -0.25]
