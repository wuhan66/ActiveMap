import pytest

from scripts.evaluate_sequential_selector import (
    canonicalize_select_action,
    selector_metrics,
    task_bootstrap,
)


def _row(task: str, target: str, prediction: str, advantage: float) -> dict:
    return {
        "task_id": task,
        "target_selection": target,
        "predicted_selection": prediction,
        "policy_relative_advantage": advantage,
    }


def test_selector_metrics_score_realized_advantage_only_when_called():
    rows = [
        _row("a", "ACQUIRE", "ACQUIRE", 1.0),
        _row("b", "STOP", "ACQUIRE", -0.75),
        _row("c", "ACQUIRE", "STOP", 1.0),
        _row("d", "STOP", "STOP", -1.0),
    ]

    metrics = selector_metrics(rows)

    assert metrics["acquire_precision"] == 0.5
    assert metrics["acquire_recall"] == 0.5
    assert metrics["realized_utility_sum"] == 0.25
    assert metrics["realized_risk_sum"] == 0.75


def test_selector_task_bootstrap_is_task_grouped_and_deterministic():
    rows = [
        _row("a", "ACQUIRE", "ACQUIRE", 1.0),
        _row("a", "STOP", "STOP", -1.0),
        _row("b", "STOP", "STOP", -1.0),
        _row("b", "ACQUIRE", "ACQUIRE", 0.75),
    ]

    result = task_bootstrap(rows, repetitions=100, seed=7)

    assert result["task_count"] == 2
    assert result["intervals"]["realized_utility_mean"]["observed"] == pytest.approx(
        0.4375
    )


def _select_prompt(*evidence_ids: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "controller"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            '{"controller_stage":"SELECT","evidence_id":"'
                            + evidence_id
                            + '"}'
                        ),
                    }
                    for evidence_id in evidence_ids
                ],
            },
        ]
    }


def test_canonicalizer_hydrates_only_the_visible_missing_evidence_id():
    result = canonicalize_select_action(
        '{"stage":"SELECT","selection":"ACQUIRE"}',
        _select_prompt("evidence-visible"),
    )

    assert result.raw_json_valid
    assert not result.raw_schema_valid
    assert result.evidence_id_hydrated
    assert result.action is not None
    assert result.action.evidence_id == "evidence-visible"
    assert result.executable_action_error is None


@pytest.mark.parametrize(
    ("raw", "prompt"),
    [
        ('{"stage":"DRAFT","selection":"ACQUIRE"}', _select_prompt("evidence-a")),
        (
            '{"stage":"SELECT","selection":"ACQUIRE","unknown":true}',
            _select_prompt("evidence-a"),
        ),
        ('{"stage":"SELECT","selection":"ACQUIRE"}', _select_prompt("a", "b")),
    ],
)
def test_canonicalizer_rejects_non_select_unknown_or_ambiguous_actions(
    raw: str, prompt: dict
):
    result = canonicalize_select_action(raw, prompt)

    assert result.raw_json_valid
    assert not result.raw_schema_valid
    assert not result.evidence_id_hydrated
    assert result.action is None
    assert result.executable_action_error is not None
