import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_updater_conditioned_selector.py"
SPEC = importlib.util.spec_from_file_location("evaluate_updater_conditioned", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_global_stop_index_is_valid_for_short_catalog_in_padded_batch() -> None:
    assert MODULE._is_stop_prediction(
        sample_id="short-catalog",
        candidate_count=3,
        prediction=7,
        stop_index=7,
    )


def test_nonstop_index_beyond_local_catalog_is_rejected() -> None:
    with pytest.raises(ValueError, match="padded candidate index 5"):
        MODULE._is_stop_prediction(
            sample_id="short-catalog",
            candidate_count=3,
            prediction=5,
            stop_index=7,
        )


def test_direct_anchor_is_not_derived_from_accumulated_selected_evidence() -> None:
    sample = SimpleNamespace(
        sample_id="step-one",
        metadata={
            "initial_evidence_id": "anchor",
            "selected_evidence_ids": ["anchor", "acquired-later"],
        },
    )
    assert MODULE._direct_evidence_id(sample) == "anchor"


def test_risk_guard_keeps_low_risk_candidate_and_suppresses_high_risk_choice() -> None:
    selected, triggered = MODULE._risk_guarded_choice(
        candidate_logits=MODULE.np.asarray([3.0, 2.0]),
        stop_logit=1.0,
        false_edit_risks=[0.9, 0.1],
        max_candidate_risk=0.2,
    )
    assert selected == 1
    assert triggered


def test_risk_guard_keeps_stop_available_when_every_candidate_is_risky() -> None:
    selected, triggered = MODULE._risk_guarded_choice(
        candidate_logits=MODULE.np.asarray([3.0, 2.0]),
        stop_logit=1.0,
        false_edit_risks=[0.9, 0.8],
        max_candidate_risk=0.2,
    )
    assert selected is None
    assert triggered
