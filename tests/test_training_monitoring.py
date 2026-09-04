import json
from pathlib import Path

from activemap.training.curves import render_run_comparison
from activemap.training.monitoring import RunMonitor


def test_monitor_writes_live_state_history_and_stop(tmp_path: Path) -> None:
    monitor = RunMonitor(tmp_path, {"tensorboard": False})
    monitor.write_state("running", epoch=0)
    monitor.record_epoch({"epoch": 1, "train": {"loss": 2.0}, "val": {"loss": 1.0}})
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    history = json.loads((tmp_path / "history.jsonl").read_text(encoding="utf-8"))
    assert state["status"] == "running"
    assert history["val"]["loss"] == 1.0
    assert (tmp_path / "history.csv").is_file()
    assert (tmp_path / "curves" / "training_curves.png").is_file()
    (tmp_path / "control" / "STOP").touch()
    assert monitor.should_stop()
    monitor.close()


def test_run_comparison_renders_shared_validation_panels(tmp_path: Path) -> None:
    records = [
        {
            "epoch": 1,
            "val": {
                "loss": 1.0,
                "iou": 0.5,
                "delete_recall": 0.2,
                "false_edit_rate": 0.05,
            },
        },
        {
            "epoch": 2,
            "val": {
                "loss": 0.8,
                "iou": 0.6,
                "delete_recall": 0.3,
                "false_edit_rate": 0.04,
            },
        },
    ]
    output = tmp_path / "comparison.png"
    render_run_comparison({"baseline": records, "candidate": records}, output)
    assert output.is_file()
    assert output.stat().st_size > 0
