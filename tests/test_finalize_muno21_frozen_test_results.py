import json
from pathlib import Path

import pytest

from scripts.finalize_muno21_frozen_test_results import SEEDS, finalize


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _rollouts() -> list[dict[str, object]]:
    return [
        {
            "task_id": task,
            "split": "test",
            "budget": budget,
            "target": "COMMIT:ADD",
            "terminal_correct": True,
            "false_edit": False,
            "missed_edit": False,
            "spent_cost": budget / 2,
            "acquisitions": 1,
            "final_evidence_quality": 0.8,
            "quality_cost_utility": 0.7 + 0.01 * budget,
            "tool_positive_episode": task == "a",
            "tool_calls": int(task == "a"),
            "tool_cost": 0.1 if task == "a" else 0.0,
            "tool_action_flips": int(task == "a"),
        }
        for task in ("a", "b", "c")
        for budget in (1.5, 3.0, 4.5)
    ]


def test_finalize_emits_rollout_and_writeback_bundles(tmp_path: Path) -> None:
    registry = Path("configs/experiments/paper_registry.yaml")
    ledger = tmp_path / "ledger.json"
    ledger.write_text('{"status":"complete","returncode":0}', encoding="utf-8")
    rollouts = tmp_path / "rollouts"
    writebacks = tmp_path / "writebacks"
    official = tmp_path / "official"
    rollout_names = (
        "oracle",
        "generic_selector",
        "edit_conditioned_selector",
        "qwen3_4b_sft",
        "forced_tools",
        "qwen3_4b_sft_tools_no_belief",
        "qwen3_4b_sft_tool_to_belief",
        "random",
        "cheapest",
        "quality_first",
        "uncertainty",
        "mapex",
        "greedy_utility",
    )
    agent_names = rollout_names[3:]
    writeback_names = (
        "generic_selector",
        "edit_conditioned_selector",
        "agent_tool_to_belief",
    )
    for seed in SEEDS:
        seed_root = rollouts / f"seed{seed}"
        for name in rollout_names:
            _write_jsonl(seed_root / f"{name}.jsonl", _rollouts())
        for name in agent_names:
            _write_jsonl(
                seed_root / f"llm_calls_{name}.jsonl",
                [
                    {
                        "task_id": task,
                        "split": "test",
                        "budget": budget,
                        "schema_valid": True,
                        "executable": True,
                    }
                    for task in ("a", "b", "c")
                    for budget in (1.5, 3.0, 4.5)
                ],
            )
        for name in writeback_names:
            _write_jsonl(
                writebacks / f"seed{seed}" / name / "writeback.jsonl",
                [
                    {
                        "task_id": task,
                        "budget": budget,
                        "target": "ADD",
                        "raster_iou": 0.8,
                        "added_polygon_iou": 0.7,
                        "removed_polygon_iou": 0.0,
                        "vector_delta_topology_valid": True,
                        "vector_replay_iou": 1.0,
                    }
                    for task in ("a", "b", "c")
                    for budget in (1.5, 3.0, 4.5)
                ],
            )
            _write_jsonl(
                official / f"seed{seed}" / name / "official_metrics.jsonl",
                [
                    {
                        "task_id": task,
                        "budget": budget,
                        "apls_improvement": 0.4,
                        "pixel_f1_improvement": 0.5,
                    }
                    for task in ("a", "b")
                    for budget in (1.5, 3.0, 4.5)
                ]
                + [
                    {
                        "task_id": "__aggregate__",
                        "budget": budget,
                        "no_change_error_rate": 0.1,
                    }
                    for budget in (1.5, 3.0, 4.5)
                ],
            )

    output = tmp_path / "paper"
    manifest = finalize(registry, ledger, rollouts, writebacks, official, output)

    assert manifest["bundle_count"] == 57
    assert manifest["paired_report_count"] == 11
    assert len(list(output.rglob("*.json"))) == 70
    assert (output / "paired_tables" / "paired_claim_matrix.csv").is_file()
    error_metric = json.loads(
        (
            output
            / "writeback"
            / "agent_tool_to_belief"
            / "budget-3.json"
        ).read_text(encoding="utf-8")
    )["metrics"]["no_change_error_rate"]
    assert error_metric["bootstrap_unit"] == "model_seed_official_aggregate"


def test_finalize_refuses_incomplete_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text('{"status":"started"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not complete"):
        finalize(
            Path("configs/experiments/paper_registry.yaml"),
            ledger,
            tmp_path / "rollouts",
            tmp_path / "writebacks",
            tmp_path / "official",
            tmp_path / "paper",
        )
