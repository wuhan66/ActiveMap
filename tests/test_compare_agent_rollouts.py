import json
from pathlib import Path

import pytest

from scripts.compare_agent_rollouts import compare


def _write(path: Path, *, candidate: bool = False) -> Path:
    rows = []
    for task_index in range(3):
        for budget in (1.5, 3.0, 4.5):
            rows.append(
                {
                    "sample_id": f"task-{task_index}__b{budget}",
                    "task_id": f"task-{task_index}",
                    "budget": budget,
                    "target": "REJECT" if task_index == 0 else "COMMIT:ADD",
                    "prediction": "REJECT" if task_index == 0 else "COMMIT:ADD",
                    "terminal_correct": True,
                    "false_edit": False,
                    "missed_edit": False,
                    "spent_cost": 0.0 if candidate else 0.1,
                    "acquisitions": 0 if candidate else 1,
                    "joint_utility": 0.8 if candidate else 0.6,
                    "final_evidence_quality": 0.9 if candidate else 0.7,
                    "evidence_quality_gain": 0.3 if candidate else 0.1,
                    "quality_cost_utility": 0.25 if candidate else 0.05,
                    "tool_calls": 1 if candidate else 2,
                    "tool_cost": 0.15 if candidate else 0.30,
                    "mean_tool_belief_l1_delta": 0.1 if candidate else 0.0,
                    "tool_action_flips": 1 if candidate else 0,
                    "terminal_edit_changed_after_tools": candidate,
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_rollout_comparison_resamples_complete_tasks(tmp_path: Path) -> None:
    result = compare(
        _write(tmp_path / "base.jsonl"),
        _write(tmp_path / "candidate.jsonl", candidate=True),
        bootstrap=100,
        seed=7,
    )
    assert result["task_count"] == 3
    assert result["sample_count"] == 9
    assert result["budgets"] == [1.5, 3.0, 4.5]
    utility = result["paired_delta"]["joint_utility_auc"]
    assert utility["delta"] == pytest.approx(0.2)
    assert utility["ci95_low"] == pytest.approx(0.2)
    assert result["paired_delta"]["mean_cost"]["delta"] == pytest.approx(-0.1)
    quality_utility = result["paired_delta"]["quality_cost_utility_auc"]
    assert quality_utility["delta"] == pytest.approx(0.2)
    assert quality_utility["ci95_low"] == pytest.approx(0.2)
    assert result["paired_delta"]["mean_tool_calls"]["delta"] == pytest.approx(-1.0)
    assert result["paired_delta"]["mean_tool_cost"]["delta"] == pytest.approx(-0.15)
    assert result["paired_delta"]["mean_tool_belief_l1_delta"]["delta"] == pytest.approx(
        0.1
    )


def test_rollout_comparison_rejects_missing_pair(tmp_path: Path) -> None:
    baseline = _write(tmp_path / "base.jsonl")
    candidate = _write(tmp_path / "candidate.jsonl", candidate=True)
    lines = candidate.read_text(encoding="utf-8").splitlines()
    candidate.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identical task-budget keys"):
        compare(baseline, candidate, bootstrap=10, seed=7)


def test_rollout_comparison_prefers_versioned_writeback_utility_when_present(
    tmp_path: Path,
) -> None:
    baseline = _write(tmp_path / "base.jsonl")
    candidate = _write(tmp_path / "candidate.jsonl", candidate=True)
    for path, value in ((baseline, 0.1), (candidate, 0.4)):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            row["episode_utility_v2_balanced"] = value
            row["episode_utility_v2_safety"] = value - 0.05
            row["episode_utility_v2_cost_aware"] = value - 0.10
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    result = compare(baseline, candidate, bootstrap=100, seed=7)

    assert result["paired_delta"]["episode_utility_v2_balanced_auc"][
        "delta"
    ] == pytest.approx(0.3)
