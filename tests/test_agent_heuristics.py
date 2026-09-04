import numpy as np

from activemap.agent.heuristics import (
    AcquireUntilBudgetPolicy,
    acquire_all_scores,
    cheapest_scores,
    greedy_utility_scores,
    mapex_scores,
    quality_first_scores,
    stable_random_scores,
    uncertainty_scores,
)
from activemap.agent.records import AgentBelief, AgentCandidate, AgentObservation
from activemap.models import EditOperation
from activemap.selector_records import SelectorSample


def _sample() -> SelectorSample:
    evidence = np.zeros((3, 13), dtype=np.float64)
    evidence[:, :2] = [[0.2, 0.4], [0.9, 0.7], [0.5, 0.6]]
    evidence[:, 11] = [0.9, 0.2, 0.6]
    evidence[:, 12] = [0.8, 0.2, 0.5]
    hypothesis = [0.7, 0.1, 0.1, 0.1] + [0.0] * 9 + [0.8, 0.0, 0.0]
    return SelectorSample(
        sample_id="episode-a",
        split="val",
        edit_type=EditOperation.KEEP,
        hypothesis_features=hypothesis,
        state_features=[1.0, 0.0, 0.7, 0.1, 0.1, 0.1, 0.0, 0.2],
        evidence_ids=["a", "b", "c"],
        evidence_features=evidence.tolist(),
        evidence_costs=[3.0, 1.0, 2.0],
        false_edit_risks=[0.5, 0.1, 0.2],
        oracle_utilities=[-100.0, 100.0, 0.0],
        metadata={
            "evidence_predictions": {
                "a": {"edit_probabilities": [0.25, 0.25, 0.25, 0.25]},
                "b": {"edit_probabilities": [0.97, 0.01, 0.01, 0.01]},
                "c": {"edit_probabilities": [0.1, 0.7, 0.1, 0.1]},
            }
        },
    )


def test_heuristic_rankings_match_their_observable_definitions() -> None:
    sample = _sample()
    assert int(np.argmax(cheapest_scores(sample))) == 1
    np.testing.assert_array_equal(acquire_all_scores(sample), [0.0, -1.0, -2.0])
    assert int(np.argmax(quality_first_scores(sample))) == 1
    assert int(np.argmax(uncertainty_scores(sample))) == 0
    assert int(np.argmax(mapex_scores(sample))) == 2
    assert greedy_utility_scores(sample).shape == (4,)
    assert stable_random_scores(sample).shape == (3,)


def test_heuristics_do_not_read_oracle_utilities_or_ground_truth() -> None:
    sample = _sample()
    counterfactual = sample.model_copy(
        update={
            "edit_type": EditOperation.RESHAPE,
            "oracle_utilities": [999.0, -999.0, 42.0],
            "stop_utility": 999.0,
            "metadata": {**sample.metadata, "gt_edit": "RESHAPE"},
        }
    )
    functions = (
        stable_random_scores,
        acquire_all_scores,
        cheapest_scores,
        quality_first_scores,
        uncertainty_scores,
        mapex_scores,
        greedy_utility_scores,
    )
    for function in functions:
        np.testing.assert_allclose(function(sample), function(counterfactual))


def test_candidate_prediction_scores_are_non_degenerate() -> None:
    sample = _sample()
    uncertainty = uncertainty_scores(sample)
    mapex = mapex_scores(sample)

    assert len(np.unique(np.round(uncertainty, 8))) == 3
    assert len(np.unique(np.round(mapex, 8))) == 3
    assert not np.array_equal(np.argsort(uncertainty), np.argsort(mapex))


def test_budget_filling_policy_continues_after_multiple_acquisitions() -> None:
    belief = AgentBelief(
        edit_probabilities=[0.8, 0.1, 0.05, 0.05],
        confidence=0.8,
        uncertainty=0.3,
    )
    observation = AgentObservation(
        task_id="task",
        split="val",
        step=4,
        initial_budget=4.5,
        remaining_budget=1.0,
        spent_cost=3.5,
        selected_evidence_ids=["anchor", "one", "two"],
        belief=belief,
        candidates=[
            AgentCandidate(
                evidence_id="remaining",
                cost=1.0,
                selector_score=0.2,
                features=[0.0] * 13,
            )
        ],
    )
    policy = AcquireUntilBudgetPolicy()

    assert policy.act(observation).key == "ACQUIRE:remaining"
    terminal = observation.model_copy(update={"candidates": [], "remaining_budget": 0.0})
    assert policy.act(terminal).key == "REJECT"
