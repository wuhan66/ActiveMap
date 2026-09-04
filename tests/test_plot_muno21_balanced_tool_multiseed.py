from __future__ import annotations

import json

from scripts.plot_muno21_balanced_tool_multiseed import main


def test_renders_direction_corrected_forests(tmp_path, monkeypatch) -> None:
    def row(comparison: str, metric: str, direction: str, delta: float) -> dict:
        return {
            "comparison": comparison,
            "metric": metric,
            "direction": direction,
            "delta": delta,
            "ci95_low": delta - 0.01,
            "ci95_high": delta + 0.01,
            "strictly_better": abs(delta) > 0.01,
        }

    data = {
        "schema_version": "muno21-balanced-tool-multiseed-table-v1",
        "controller_rows": [
            row("agent - selector", "terminal_accuracy", "higher", 0.03),
            row("agent - selector", "false_edit_rate", "lower", -0.03),
        ],
        "writeback_rows": [
            row("safe - raw", "raster_iou_gain_auc", "higher", 0.02),
            row("safe - raw", "false_edit_auc", "lower", -0.02),
        ],
        "split": "val",
        "test_assets_read": False,
    }
    source = tmp_path / "table.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "figures"
    monkeypatch.setattr(
        "sys.argv",
        ["plot_muno21_balanced_tool_multiseed.py", str(source), str(output)],
    )
    main()
    for stem in ("controller_forest", "writeback_forest"):
        assert (output / f"{stem}.png").stat().st_size > 1000
        assert (output / f"{stem}.pdf").stat().st_size > 1000
