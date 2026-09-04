from __future__ import annotations

import json

from scripts.assess_muno21_balanced_tool_multiseed import main


def _controller_interval(delta: float) -> dict[str, float]:
    return {
        "delta": delta,
        "ci95_low": delta - 0.01,
        "ci95_high": delta + 0.01,
    }


def _writeback_interval(delta: float) -> dict[str, float]:
    return {
        "observed_delta": delta,
        "ci95_low": delta - 0.01,
        "ci95_high": delta + 0.01,
    }


def test_strict_balanced_tool_promotion_passes(tmp_path, monkeypatch) -> None:
    seeds = [11, 12]
    candidate = {
        "episode_utility_v2_balanced_auc": 0.2,
        "mean_tool_calls": 0.1,
        "mean_tool_belief_l1_delta": 0.02,
        "mean_tool_action_flips": 0.01,
    }
    comparisons = {}
    for baseline in ("edit_conditioned_selector", "qwen3_4b_sft_tools_no_belief"):
        comparisons[baseline] = {
            "per_seed": [
                {"seed": seed, "candidate": candidate, "baseline": {}, "delta": {}}
                for seed in seeds
            ],
            "paired_delta": {
                "episode_utility_v2_balanced_auc": _controller_interval(0.04),
                "false_edit_rate": _controller_interval(-0.03),
            },
        }
    controller = {
        "schema_version": "activemap-agent-three-seed-bootstrap-v2",
        "protocol": {"model_seeds": seeds, "test_assets_read": False},
        "comparisons": comparisons,
    }
    controller_path = tmp_path / "controller.json"
    controller_path.write_text(json.dumps(controller))

    writeback_paths = []
    for name in ("safe_sft", "safe_raw"):
        data = {
            "schema_version": "agent-writeback-seed-matched-aggregate-v1",
            "model_seeds": seeds,
            "candidate_minus_seed_matched_sft": {
                "raster_iou_gain_auc": _writeback_interval(0.04),
                "false_edit_auc": _writeback_interval(-0.03),
                "missed_edit_auc": _writeback_interval(-0.03),
            },
            "test_assets_read": False,
        }
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data))
        writeback_paths.append(path)

    run_dirs = []
    for seed in seeds:
        run = tmp_path / f"run_seed{seed}"
        selection = run / "evaluation/selection"
        selection.mkdir(parents=True)
        (selection / "static_checkpoint_decision.json").write_text(
            json.dumps(
                {
                    "protocol": {"test_assets_read": False},
                    "selection_passed": True,
                    "selected_checkpoint": "checkpoint-400",
                }
            )
        )
        run_dirs.append(run)

    output = tmp_path / "promotion.json"
    argv = [
        "assess_muno21_balanced_tool_multiseed.py",
        str(controller_path),
        *(str(path) for path in writeback_paths),
        str(output),
    ]
    for run in run_dirs:
        argv.extend(["--run-dir", str(run)])
    monkeypatch.setattr("sys.argv", argv)
    main()
    result = json.loads(output.read_text())
    assert result["status"] == "pass"
    assert all(result["groups"].values())
    assert result["test_assets_read"] is False
