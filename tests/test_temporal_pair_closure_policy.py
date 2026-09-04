from activemap.agent.records import AgentBelief, AgentCandidate, AgentObservation
from activemap.agent.tools import GreedyAgentPolicy, TemporalPairClosurePolicy
from activemap.models import EditOperation


def _observation() -> AgentObservation:
    return AgentObservation(
        task_id="task-1",
        split="val",
        step=1,
        initial_budget=3.0,
        remaining_budget=2.0,
        spent_cost=1.0,
        selected_evidence_ids=["scene-p00-city-add__y2019"],
        candidates=[
            AgentCandidate(
                evidence_id="scene-p00-city-add__y2012",
                cost=0.4,
                selector_score=0.75,
                features=[],
            )
        ],
        terminal_score=0.8,
        belief=AgentBelief(
            edit_probabilities=[0.1, 0.7, 0.1, 0.1],
            confidence=0.7,
            uncertainty=0.2,
            recommended_edit=EditOperation.ADD,
        ),
    )


def test_temporal_pair_closure_acquires_near_tied_counterpart() -> None:
    assert TemporalPairClosurePolicy(closure_margin=0.1).act(_observation()).key == (
        "ACQUIRE:scene-p00-city-add__y2012"
    )


def test_zero_margin_recovers_greedy_terminal_action() -> None:
    assert TemporalPairClosurePolicy().act(_observation()).key == GreedyAgentPolicy().act(
        _observation()
    ).key
