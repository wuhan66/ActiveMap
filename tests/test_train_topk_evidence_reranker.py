import numpy as np
import pytest

from activemap.features import ONLINE_OBSERVABLE_STATE_CONTRACT
from activemap.selector_records import SelectorSample
from scripts.train_topk_evidence_reranker import resolve_data_contract, scan_stop_frontier


def _row(gains: list[float]) -> dict:
    return {
        "utility_gains": np.asarray(gains, dtype=np.float32),
        "stop_utility": 0.0,
    }


def test_stop_frontier_finds_noncollapsed_safe_point():
    examples = [
        _row([0.4, -0.2]),
        _row([0.3, -0.1]),
        _row([-0.2, -0.3]),
        _row([-0.1, -0.4]),
    ]
    scores = [
        np.asarray([0.9, 0.1]),
        np.asarray([0.8, 0.2]),
        np.asarray([0.3, 0.1]),
        np.asarray([0.2, 0.0]),
    ]
    result = scan_stop_frontier(
        examples,
        scores,
        maximum_false_call_rate=0.0,
        maximum_harmful_call_fraction=0.0,
        minimum_acquire_recall=0.5,
    )
    assert result["constraints_satisfied"] is True
    assert result["selected"]["calls"] == 2
    assert result["selected"]["acquire_recall"] == 1.0
    assert result["selected"]["utility_gain_over_stop_mean"] > 0.0


def test_stop_frontier_respects_zero_false_call_constraint():
    examples = [_row([0.2]), _row([-0.2])]
    scores = [np.asarray([0.9]), np.asarray([0.8])]
    result = scan_stop_frontier(
        examples,
        scores,
        maximum_false_call_rate=0.0,
        maximum_harmful_call_fraction=0.0,
        minimum_acquire_recall=1.0,
    )
    assert result["constraints_satisfied"] is True
    assert result["selected"]["calls"] == 1


def _sample(contract):
    return SelectorSample(
        sample_id="state",
        split="train",
        edit_type="KEEP",
        hypothesis_features=[0.0] * 16,
        state_features=[0.0] * 8,
        evidence_ids=["candidate"],
        evidence_features=[[0.0] * 13],
        evidence_costs=[1.0],
        false_edit_risks=[0.0],
        oracle_utilities=[0.0],
        metadata={"online_state_contract": contract},
    )


def test_resolve_data_contract_only_propagates_sanitized_online_states():
    assert resolve_data_contract([_sample(ONLINE_OBSERVABLE_STATE_CONTRACT)]) == (
        ONLINE_OBSERVABLE_STATE_CONTRACT
    )
    assert resolve_data_contract([_sample(None)]) is None
    with pytest.raises(ValueError, match="mixed or unsupported"):
        resolve_data_contract(
            [_sample(ONLINE_OBSERVABLE_STATE_CONTRACT), _sample(None)]
        )
