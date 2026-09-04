import json
from pathlib import Path

import pytest
import yaml

from scripts.audit_paper_result_bundles import audit_result_bundles
from scripts.build_paper_rollout_bundle import build_rollout_bundle


def _registry(path: Path, *, family: str) -> Path:
    point_metrics = (
        [
            "mean_quality_cost_utility",
            "regret",
            "mean_cost",
            "terminal_accuracy",
            "false_edit_rate",
        ]
        if family == "selector"
        else [
            "mean_quality_cost_utility",
            "false_edit_rate",
            "schema_valid_rate",
            "executable_valid_rate",
            "grounded_tool_recall",
            "belief_action_flip_rate",
        ]
    )
    payload = {
        "protocol": {
            "muno21_budgets": [1.0, 2.0],
            "bootstrap_unit": "task_or_aoi",
            "bootstrap_replicates": 100,
            "bootstrap_seed": 9,
            "confidence_level": 0.95,
        },
        "required_metrics": {
            "updater": ["score"],
            "selector": point_metrics if family == "selector" else ["score"],
            "agent": point_metrics if family == "agent" else ["score"],
            "writeback": ["score"],
        },
        "primary_metrics": {
            "updater": ["score"],
            "selector": ["mean_quality_cost_utility"] if family == "selector" else ["score"],
            "agent": ["mean_quality_cost_utility"] if family == "agent" else ["score"],
            "rl": ["score"],
            "writeback": ["score"],
        },
        "curve_metrics": {family: ["quality_cost_auc"]},
        "curve_primary_metrics": {family: ["quality_cost_auc"]},
        "experiments": [
            {
                "id": "policy",
                "family": family,
                "test_policy": "validation_only",
                "seeds": [7, 8],
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _write_rollouts(path: Path, *, seed: int, oracle: bool = False) -> Path:
    rows = []
    for task_index, task in enumerate(("a", "b")):
        for budget in (1.0, 2.0):
            utility = 0.1 * seed + 0.2 * budget + 0.1 * task_index
            if oracle:
                utility += 0.25
            rows.append(
                {
                    "sample_id": f"{seed}-{task}-{budget}",
                    "task_id": task,
                    "split": "val",
                    "budget": budget,
                    "target": "REJECT",
                    "prediction": "REJECT",
                    "terminal_correct": True,
                    "false_edit": False,
                    "missed_edit": False,
                    "spent_cost": 0.5 * budget,
                    "quality_cost_utility": utility,
                    "tool_positive_episode": task == "a",
                    "tool_calls": 1 if task == "a" else 0,
                    "tool_action_flips": 1 if task == "a" else 0,
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _write_validity(path: Path, *, seed: int) -> Path:
    rows = []
    for task in ("a", "b"):
        for budget in (1.0, 2.0):
            rows.append(
                {
                    "task_id": task,
                    "split": "val",
                    "budget": budget,
                    "step": 0,
                    "schema_valid": True,
                    "executable": not (seed == 8 and task == "b"),
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_selector_point_and_curve_bundles_cover_all_registry_cells(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.yaml", family="selector")
    rollout_paths = {
        str(seed): _write_rollouts(tmp_path / f"rollout-{seed}.jsonl", seed=seed)
        for seed in (7, 8)
    }
    oracle_paths = {
        str(seed): _write_rollouts(
            tmp_path / f"oracle-{seed}.jsonl", seed=seed, oracle=True
        )
        for seed in (7, 8)
    }
    bundles = tmp_path / "bundles"
    bundles.mkdir()
    for budget in (1.0, 2.0):
        bundle = build_rollout_bundle(
            registry,
            rollout_paths,
            experiment_id="policy",
            variant=None,
            budget=budget,
            seed_oracle_paths=oracle_paths,
        )
        assert bundle["metrics"]["regret"]["mean"] == pytest.approx(0.25)
        (bundles / f"point-{budget}.json").write_text(
            json.dumps(bundle), encoding="utf-8"
        )
    curve = build_rollout_bundle(
        registry,
        rollout_paths,
        experiment_id="policy",
        variant=None,
        budget=None,
    )
    assert curve["cell_kind"] == "curve_summary"
    assert curve["metrics"]["quality_cost_auc"]["mean"] == pytest.approx(1.1)
    (bundles / "curve.json").write_text(json.dumps(curve), encoding="utf-8")

    report = audit_result_bundles(registry, bundles)
    assert report["expected_cell_count"] == 3
    assert report["paper_tables_complete"] is True


def test_agent_point_uses_task_budget_validity_and_grounded_tool_metrics(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path / "registry.yaml", family="agent")
    rollout_paths = {
        str(seed): _write_rollouts(tmp_path / f"rollout-{seed}.jsonl", seed=seed)
        for seed in (7, 8)
    }
    validity_paths = {
        str(seed): _write_validity(tmp_path / f"validity-{seed}.jsonl", seed=seed)
        for seed in (7, 8)
    }
    bundle = build_rollout_bundle(
        registry,
        rollout_paths,
        experiment_id="policy",
        variant=None,
        budget=1.0,
        seed_validity_paths=validity_paths,
    )
    assert bundle["metrics"]["schema_valid_rate"]["mean"] == 1.0
    assert bundle["metrics"]["executable_valid_rate"]["mean"] == pytest.approx(0.75)
    assert bundle["metrics"]["grounded_tool_recall"]["mean"] == 1.0
    assert bundle["metrics"]["belief_action_flip_rate"]["mean"] == 1.0


def test_agent_point_refuses_missing_validity_without_explicit_mode(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.yaml", family="agent")
    rollout_paths = {
        str(seed): _write_rollouts(tmp_path / f"rollout-{seed}.jsonl", seed=seed)
        for seed in (7, 8)
    }
    with pytest.raises(ValueError, match="validity metrics"):
        build_rollout_bundle(
            registry,
            rollout_paths,
            experiment_id="policy",
            variant=None,
            budget=1.0,
        )
