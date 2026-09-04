from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from activemap.agent.records import AgentBelief, AgentObservation
from activemap.geo_tools.records import GeoToolName
from scripts.evaluate_agent_rollouts import (
    StructuredLLMPolicy,
    _grounded_action_completions,
    _select_samples,
)


@dataclass(frozen=True)
class Sample:
    sample_id: str


def test_seeded_hash_sampling_is_reproducible_and_not_source_prefix() -> None:
    samples = [Sample(f"sample-{index:03d}") for index in range(100)]
    first = _select_samples(samples, limit=20, order="seeded-hash", seed=7)
    again = _select_samples(samples, limit=20, order="seeded-hash", seed=7)
    other = _select_samples(samples, limit=20, order="seeded-hash", seed=8)
    assert first == again
    assert first != samples[:20]
    assert first != other


def test_source_sampling_remains_backward_compatible() -> None:
    samples = [Sample(str(index)) for index in range(5)]
    assert _select_samples(samples, limit=2, order="source", seed=1) == samples[:2]
    with pytest.raises(ValueError, match="positive"):
        _select_samples(samples, limit=0, order="source", seed=1)


def test_tool_policy_uses_explicit_schema_at_initial_state() -> None:
    policy = StructuredLLMPolicy(
        None,
        None,
        device="cpu",
        max_length=128,
        system_prompt="generic",
        tool_system_prompt="tool-schema",
    )
    observation = SimpleNamespace(selected_evidence_ids=["evidence-1"], tool_history=[])
    assert policy.prompt_for(observation) == "tool-schema"


def test_grounded_decoder_enumerates_only_currently_executable_actions() -> None:
    observation = AgentObservation(
        task_id="task",
        split="train",
        step=1,
        initial_budget=1.0,
        remaining_budget=0.8,
        spent_cost=0.2,
        selected_evidence_ids=["evidence-a", "evidence-b"],
        belief=AgentBelief(
            edit_probabilities=[0.5, 0.3, 0.1, 0.1], confidence=0.5, uncertainty=0.6
        ),
        candidates=[],
        available_tools=[GeoToolName.IMAGE_QUALITY],
        tool_history=[],
    )

    actions = dict(_grounded_action_completions(observation))

    assert set(actions) == {
        "REJECT",
        "COMMIT:ADD",
        "COMMIT:DELETE",
        "COMMIT:RESHAPE",
        "USE_TOOL:IMAGE_QUALITY:0",
        "USE_TOOL:IMAGE_QUALITY:1",
    }
    assert '"evidence_index":0' in actions["USE_TOOL:IMAGE_QUALITY:0"]
    assert '"evidence_index":1' in actions["USE_TOOL:IMAGE_QUALITY:1"]
