import json
from types import SimpleNamespace

import pytest

from scripts.evaluate_agent_rollouts import (
    StructuredLLMPolicy,
    _normalize_structured_action_payload,
    _ensure_task_coverage,
    _load_tool_positive_task_ids,
    _parse_methods,
    _sample_task_id,
)


def test_parse_methods_supports_baseline_only_evaluation():
    assert _parse_methods("generic_selector,edit_conditioned_selector") == {
        "generic_selector",
        "edit_conditioned_selector",
    }


def test_parse_methods_supports_registered_heuristic_suite():
    assert _parse_methods(
        "random,acquire_all,cheapest,quality_first,uncertainty,mapex,greedy_utility"
    ) == {
        "random",
        "acquire_all",
        "cheapest",
        "quality_first",
        "uncertainty",
        "mapex",
        "greedy_utility",
    }


def test_parse_methods_supports_closed_loop_tool_ablations():
    assert _parse_methods(
        "forced_tools,qwen3_4b_sft_tools_no_belief,qwen3_4b_sft_tool_to_belief"
    ) == {
        "forced_tools",
        "qwen3_4b_sft_tools_no_belief",
        "qwen3_4b_sft_tool_to_belief",
    }


def test_parse_methods_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown rollout methods"):
        _parse_methods("generic_selector,imaginary")


def test_tool_policy_uses_explicit_schema_at_initial_state():
    policy = StructuredLLMPolicy(
        None,
        None,
        device="cpu",
        max_length=128,
        system_prompt="select",
        tool_system_prompt="tool",
    )

    assert policy.prompt_for(
        SimpleNamespace(selected_evidence_ids=["anchor"], tool_history=[])
    ) == "tool"
    assert policy.prompt_for(
        SimpleNamespace(selected_evidence_ids=["anchor", "candidate"], tool_history=[])
    ) == "tool"
    assert policy.prompt_for(
        SimpleNamespace(selected_evidence_ids=["anchor"], tool_history=[object()])
    ) == "tool"
    assert policy.do_sample is False


def test_selected_evidence_index_resolves_to_executable_tool_payload():
    observation = SimpleNamespace(
        task_id="task/with spaces",
        step=1,
        selected_evidence_ids=["evidence-anchor", "evidence-selected"],
    )
    payload, resolution = _normalize_structured_action_payload(
        {
            "action": "USE_TOOL",
            "tool_call": {
                "tool": "IMAGE_QUALITY",
                "inputs": {"evidence_index": 1},
                "parameters": {},
            },
        },
        observation,
    )

    assert resolution == {"mode": "selected_evidence_index_v1", "evidence_index": 1}
    assert payload["tool_call"]["inputs"] == {"evidence_id": "evidence-selected"}
    assert payload["tool_call"]["call_id"].startswith("policy-task-with-spaces-step1")


def test_string_pointer_and_incidental_call_id_are_normalized():
    observation = SimpleNamespace(
        task_id="task-a",
        step=0,
        selected_evidence_ids=["evidence-a"],
    )
    payload, resolution = _normalize_structured_action_payload(
        {
            "action": "USE_TOOL",
            "tool_call": {
                "call_id": "not a valid call id",
                "tool": "TEMPORAL_CHANGE",
                "inputs": {"evidence_index": "0"},
                "parameters": {},
            },
        },
        observation,
    )

    assert resolution["mode"] == "selected_evidence_index_v1"
    assert payload["tool_call"]["inputs"] == {"evidence_id": "evidence-a"}
    assert payload["tool_call"]["call_id"] == "policy-task-a-step0-temporal_change"


def test_tool_positive_tasks_are_loaded_from_natural_supervision(tmp_path):
    path = tmp_path / "sft.jsonl"
    rows = [
        {
            "oracle_tool_stage": stage,
            "messages": [
                {"role": "system", "content": "x"},
                {
                    "role": "user",
                    "content": f'{{"task_id":"task-{stage}","split":"val"}}',
                },
                {"role": "assistant", "content": "{}"},
            ],
        }
        for stage in (0, 1, 3)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert _load_tool_positive_task_ids(path) == {"task-1", "task-3"}


def test_sample_task_id_uses_public_identifier_for_source_episode():
    sample = SimpleNamespace(
        sample_id="sample",
        metadata={"source_episode": "episode-raw"},
    )

    assert _sample_task_id(sample).startswith("task-")


def test_tool_positive_coverage_uses_only_needed_replacement_slots():
    samples = [
        SimpleNamespace(sample_id=f"sample-{index}", metadata={"task_id": f"task-{index}"})
        for index in range(4)
    ]

    selected, injected = _ensure_task_coverage(
        samples,
        samples[:2],
        {"task-2"},
        limit=2,
    )

    assert injected == 1
    assert {sample.metadata["task_id"] for sample in selected} == {"task-0", "task-2"}


def test_tool_positive_coverage_can_record_missing_stage_tasks():
    samples = [
        SimpleNamespace(sample_id="sample-0", metadata={"task_id": "task-0"}),
        SimpleNamespace(sample_id="sample-1", metadata={"task_id": "task-1"}),
    ]

    selected, injected = _ensure_task_coverage(
        samples,
        samples[:1],
        {"task-1", "task-missing"},
        limit=1,
        allow_missing_required_tasks=True,
    )

    assert injected == 1
    assert {sample.metadata["task_id"] for sample in selected} == {"task-1"}
