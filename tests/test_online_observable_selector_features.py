import math

import pytest

from activemap.agent.environment import MapMaintenanceEnv
from activemap.agent.records import AgentAction, AgentActionType
from activemap.agent.tools import CounterfactualBeliefUpdater
from activemap.features import ONLINE_OBSERVABLE_STATE_CONTRACT
from activemap.models import EditOperation
from activemap.selector_records import SelectorSample
from activemap.training.contracts import validate_selector_data_contract
from scripts.sanitize_online_selector_features import sanitize_sample


def sample_fixture() -> SelectorSample:
    return SelectorSample(
        sample_id="online-observable-fixture",
        split="val",
        edit_type=EditOperation.ADD,
        hypothesis_features=[0.1, 0.8, 0.05, 0.05] + [0.0] * 12,
        state_features=[1.0, 0.0, 0.1, 0.8, 0.05, 0.05, 1.0 / 3.0, -0.4],
        evidence_ids=["candidate-a", "candidate-b"],
        evidence_features=[[0.0] * 13, [0.0] * 13],
        evidence_costs=[0.2, 0.3],
        false_edit_risks=[0.1, 0.1],
        oracle_utilities=[0.2, 0.1],
        metadata={
            "selected_evidence_ids": ["direct"],
            "evidence_predictions": {
                "direct": {
                    "edit_probabilities": [0.1, 0.8, 0.05, 0.05],
                    "confidence": 0.6,
                    "geometry_delta": [0.0] * 8,
                },
                "candidate-a": {
                    "edit_probabilities": [0.1, 0.8, 0.05, 0.05],
                    "confidence": 0.8,
                    "geometry_delta": [0.0] * 8,
                },
                "candidate-b": {
                    "edit_probabilities": [0.8, 0.1, 0.05, 0.05],
                    "confidence": 0.4,
                    "geometry_delta": [0.0] * 8,
                },
            },
            "gt_edit": EditOperation.KEEP.value,
        },
    )


def test_sanitized_state7_is_observable_fused_confidence_and_preserves_labels():
    source = sample_fixture()
    sanitized = sanitize_sample(source)

    assert math.isclose(sanitized.state_features[7], 0.8)
    assert sanitized.oracle_utilities == source.oracle_utilities
    assert sanitized.stop_utility == source.stop_utility
    assert sanitized.metadata["online_state_contract"]["state7"] == "fused_belief_confidence"


def test_online_environment_keeps_state7_observable_after_acquisition():
    source = sanitize_sample(sample_fixture())
    environment = MapMaintenanceEnv(
        source,
        budget=1.0,
        score_fn=lambda current: [0.0] * (len(current.evidence_ids) + 1),
        belief_updater=CounterfactualBeliefUpdater(source),
    )
    environment.step(
        AgentAction(action=AgentActionType.ACQUIRE, evidence_id="candidate-a")
    )

    current = environment.current_sample()
    assert math.isclose(current.state_features[7], environment.belief.confidence)
    assert not math.isclose(current.state_features[7], environment.current_gain)


def test_sanitizer_rejects_test_records():
    test_sample = sample_fixture().model_copy(update={"split": "test"})
    with pytest.raises(ValueError, match="train/validation"):
        sanitize_sample(test_sample)


def test_training_contract_accepts_only_sanitized_online_records():
    sanitized = sanitize_sample(sample_fixture())

    assert validate_selector_data_contract(
        [sanitized], ONLINE_OBSERVABLE_STATE_CONTRACT
    ) == ONLINE_OBSERVABLE_STATE_CONTRACT
    with pytest.raises(ValueError, match="violates"):
        validate_selector_data_contract([sample_fixture()], ONLINE_OBSERVABLE_STATE_CONTRACT)
