import numpy as np

from activemap.evaluation.selector import evaluate_score_policy, initial_states_for_budget
from activemap.oracle.counterfactual import (
    counterfactual_action_utilities,
    greedy_oracle_rollout,
)
from activemap.policy.baselines import BaselineName, baseline_scores
from activemap.synthetic import generate_selector_smoke_samples


def test_counterfactual_oracle_respects_budget_and_stop() -> None:
    values = [0.4, 0.9, 0.2]

    def quality(selected: tuple[int, ...]) -> float:
        return max((values[index] for index in selected), default=0.0)

    utilities = counterfactual_action_utilities(
        selected=(),
        candidate_count=3,
        costs=[1.0, 2.0, 1.0],
        quality_fn=quality,
        cost_weight=0.1,
    )
    assert max(utilities, key=lambda item: item.utility).action_index == 1

    rollout = greedy_oracle_rollout(
        candidate_count=3,
        costs=[1.0, 2.0, 1.0],
        budget=1.0,
        quality_fn=quality,
        cost_weight=0.1,
    )
    assert rollout[0].action_index == 0
    assert rollout[-1].stopped


def test_baselines_and_budgeted_metrics() -> None:
    samples = generate_selector_smoke_samples(sample_count=40, candidate_count=5, seed=2)
    test_samples = [sample for sample in samples if sample.split == "test"]
    metric = evaluate_score_policy(
        test_samples,
        method="quality",
        budget=2.5,
        score_fn=lambda sample: baseline_scores(sample, BaselineName.QUALITY),
    )
    assert metric.sample_count == len(test_samples)
    assert metric.mean_cost <= 2.5 + 1e-6
    assert metric.mean_regret >= -1e-6
    assert np.isfinite(metric.mean_utility)

    stop_metric = evaluate_score_policy(
        test_samples,
        method="always_stop",
        budget=2.5,
        score_fn=lambda sample: baseline_scores(sample, BaselineName.ALWAYS_STOP),
    )
    assert np.isclose(
        stop_metric.mean_utility,
        np.mean([sample.stop_utility for sample in test_samples]),
    )
    assert np.isclose(stop_metric.mean_cost, 0.0)


def test_one_step_evaluation_does_not_credit_lower_ranked_evidence() -> None:
    sample = generate_selector_smoke_samples(sample_count=40, candidate_count=2, seed=4)[0]
    sample = sample.model_copy(
        update={
            "evidence_costs": [1.0, 1.0],
            "oracle_utilities": [0.1, 0.9],
            "stop_utility": 0.0,
        }
    )
    metric = evaluate_score_policy(
        [sample],
        method="wrong_first",
        budget=2.0,
        score_fn=lambda _: np.asarray([2.0, 1.0]),
    )
    assert np.isclose(metric.mean_utility, 0.1)
    assert np.isclose(metric.mean_regret, 0.8)


def test_one_step_evaluation_keeps_negative_acquisition_utility() -> None:
    sample = generate_selector_smoke_samples(sample_count=40, candidate_count=2, seed=8)[0]
    sample = sample.model_copy(
        update={
            "evidence_costs": [1.0, 1.0],
            "oracle_utilities": [-0.3, 0.2],
            "stop_utility": 0.0,
        }
    )
    metric = evaluate_score_policy(
        [sample],
        method="harmful_acquire",
        budget=1.0,
        score_fn=lambda _: np.asarray([2.0, 1.0, -1.0]),
    )
    assert np.isclose(metric.mean_utility, -0.3)
    assert np.isclose(metric.mean_regret, 0.5)


def test_expanded_states_are_filtered_by_budget_and_initial_step() -> None:
    sample = generate_selector_smoke_samples(sample_count=40, candidate_count=2, seed=5)[0]
    states = [
        sample.model_copy(
            update={
                "sample_id": f"state-{budget}-{step}",
                "metadata": {
                    "budget": budget,
                    "oracle_step": step,
                    "source_episode": "episode-1",
                },
            }
        )
        for budget in (1.5, 3.0)
        for step in (0, 1)
    ]
    selected = initial_states_for_budget(states, 3.0)
    assert [state.sample_id for state in selected] == ["state-3.0-0"]
