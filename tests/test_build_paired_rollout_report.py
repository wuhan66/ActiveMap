import json
from pathlib import Path

import pytest
import yaml

from scripts.build_paired_rollout_report import build_report


def _registry(path: Path) -> Path:
    payload = {
        "protocol": {
            "bootstrap_replicates": 200,
            "bootstrap_seed": 7,
            "confidence_level": 0.95,
            "primary_budget": {"muno21": 3.0},
            "safety_gates": {"max_false_edit_rate": 0.60},
            "paired_comparison_margins": {
                "min_primary_utility_delta": 0.0,
                "min_quality_cost_auc_delta": 0.0,
                "max_false_edit_rate_delta": 0.01,
                "max_mean_cost_delta_for_matched_cost": 0.0,
            },
        },
        "experiments": [
            {"id": "baseline", "seeds": [1, 2]},
            {"id": "candidate", "seeds": [1, 2]},
        ],
        "required_paired_comparisons": [
            {
                "id": "candidate_vs_baseline",
                "claim": "candidate is better",
                "baseline_experiment": "baseline",
                "candidate_experiment": "candidate",
                "test_policy": "frozen_once",
                "required_gates": [
                    "quality_cost_safety_non_dominated",
                    "matched_cost",
                ],
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _rollouts(path: Path, *, candidate: bool) -> Path:
    rows = []
    for task in ("a", "b", "c"):
        for budget in (1.5, 3.0, 4.5):
            rows.append(
                {
                    "task_id": task,
                    "split": "test",
                    "budget": budget,
                    "target": "REJECT",
                    "quality_cost_utility": 0.6 if candidate else 0.4,
                    "final_evidence_quality": 0.8 if candidate else 0.7,
                    "terminal_correct": True,
                    "false_edit": False,
                    "missed_edit": False,
                    "spent_cost": 0.5 if candidate else 0.7,
                    "acquisitions": 1,
                    "tool_calls": 1 if candidate else 2,
                    "tool_cost": 0.1 if candidate else 0.2,
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_builds_cross_seed_paired_quality_cost_safety_report(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.yaml")
    baseline = {
        str(seed): _rollouts(tmp_path / f"baseline-{seed}.jsonl", candidate=False)
        for seed in (1, 2)
    }
    candidate = {
        str(seed): _rollouts(tmp_path / f"candidate-{seed}.jsonl", candidate=True)
        for seed in (1, 2)
    }
    ledger = tmp_path / "ledger.json"
    ledger.write_text('{"status":"complete","returncode":0}', encoding="utf-8")

    report = build_report(
        registry,
        baseline,
        candidate,
        comparison_id="candidate_vs_baseline",
        frozen_test_ledger=ledger,
    )

    assert report["task_count"] == 3
    assert report["test_assets_read"] is True
    assert report["by_budget"]["3"]["quality_cost_utility"][
        "delta_candidate_minus_baseline"
    ]["mean"] == pytest.approx(0.2)
    assert report["quality_cost_auc"]["delta_candidate_minus_baseline"][
        "ci95"
    ][0] == pytest.approx(0.2)
    assert report["gates"]["matched_cost"] is True
    assert report["all_required_gates_passed"] is True
    assert report["by_budget"]["3"]["false_edit_rate"]["candidate"]["ci95"][
        1
    ] > 0.0


def test_test_report_requires_complete_ledger(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.yaml")
    paths = {
        str(seed): _rollouts(tmp_path / f"rollout-{seed}.jsonl", candidate=False)
        for seed in (1, 2)
    }
    with pytest.raises(ValueError, match="require a frozen test ledger"):
        build_report(
            registry,
            paths,
            paths,
            comparison_id="candidate_vs_baseline",
        )


def test_report_rejects_seed_relabeling(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.yaml")
    path = _rollouts(tmp_path / "rollout.jsonl", candidate=False)
    ledger = tmp_path / "ledger.json"
    ledger.write_text('{"status":"complete","returncode":0}', encoding="utf-8")
    with pytest.raises(ValueError, match="baseline seed files"):
        build_report(
            registry,
            {"deterministic": path},
            {"1": path, "2": path},
            comparison_id="candidate_vs_baseline",
            frozen_test_ledger=ledger,
        )
