from __future__ import annotations

import json

from scripts.plot_muno21_checkpoint_safety_trajectory import main


def test_plots_checkpoint_safety_trajectories(tmp_path, monkeypatch) -> None:
    runs = []
    for seed in (11, 12):
        run = tmp_path / f"balanced_seed{seed}"
        for step, false_call in ((100, 0.08), (400, 0.01)):
            output = run / "evaluation" / f"checkpoint-{step}" / "actions"
            output.mkdir(parents=True)
            summary = {
                "sample_count": 20,
                "macro_f1": 0.4 + step / 10000,
                "exact_action_accuracy": 0.6,
                "tool_metrics": {
                    "false_call_rate": false_call,
                    "grounded_call_accuracy": 0.6,
                    "tool_positive_exact_accuracy": 0.5,
                },
                "test_assets_read": False,
            }
            (output / "summary.json").write_text(json.dumps(summary))
        selection = run / "evaluation" / "selection"
        selection.mkdir(parents=True)
        (selection / "static_checkpoint_decision.json").write_text(
            json.dumps(
                {
                    "protocol": {"test_assets_read": False},
                    "selected_checkpoint": "checkpoint-400",
                }
            )
        )
        runs.append(run)

    output = tmp_path / "figures"
    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_muno21_checkpoint_safety_trajectory.py",
            str(output),
            *(str(run) for run in runs),
        ],
    )
    main()
    assert (output / "checkpoint_safety_trajectory.csv").is_file()
    assert (output / "checkpoint_safety_trajectory.json").is_file()
    assert (output / "checkpoint_safety_trajectory.png").stat().st_size > 1000
    assert (output / "checkpoint_safety_trajectory.pdf").stat().st_size > 1000
