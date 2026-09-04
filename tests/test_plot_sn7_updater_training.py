import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from scripts.plot_sn7_updater_training import plot_histories


def test_plot_histories_exports_four_clean_figures(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    history.write_text(
        "".join(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": 1.0 / epoch,
                    "val": {
                        "committed_map_iou": 0.5 + epoch * 0.05,
                        "map_iou_delta": -0.05 + epoch * 0.05,
                        "change_iou": 0.4 + epoch * 0.05,
                    },
                }
            )
            + "\n"
            for epoch in (1, 2)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "plots"
    result = plot_histories([("run", history)], output)

    assert len(result["outputs"]) == 4
    assert all((output / name).stat().st_size > 1_000 for name in result["outputs"])
