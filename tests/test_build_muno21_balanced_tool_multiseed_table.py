from __future__ import annotations

import json

from scripts.build_muno21_balanced_tool_multiseed_table import main


def _interval(delta: float) -> dict[str, float]:
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


def test_builds_direction_aware_multiseed_tables(tmp_path, monkeypatch) -> None:
    seeds = [1, 2, 3, 4]
    controller = {
        "schema_version": "activemap-agent-three-seed-bootstrap-v2",
        "protocol": {"model_seeds": seeds, "test_assets_read": False},
        "candidate": "agent",
        "comparisons": {
            "selector": {
                "paired_delta": {
                    "terminal_accuracy": _interval(0.03),
                    "false_edit_rate": _interval(-0.03),
                }
            }
        },
    }
    controller_path = tmp_path / "controller.json"
    controller_path.write_text(json.dumps(controller), encoding="utf-8")

    writeback_paths = []
    for index in range(3):
        data = {
            "schema_version": "agent-writeback-seed-matched-aggregate-v1",
            "model_seeds": seeds,
            "candidate_minus_seed_matched_sft": {
                "raster_iou_gain_auc": _writeback_interval(0.03 + index / 100),
                "false_edit_auc": _writeback_interval(-0.03),
            },
            "split": "val",
            "test_assets_read": False,
        }
        path = tmp_path / f"writeback_{index}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        writeback_paths.append(path)

    output = tmp_path / "tables"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_muno21_balanced_tool_multiseed_table.py",
            str(controller_path),
            *(str(path) for path in writeback_paths),
            str(output),
        ],
    )
    main()

    result = json.loads((output / "table.json").read_text())
    false_edit = next(
        row
        for row in result["controller_rows"]
        if row["metric"] == "false_edit_rate"
    )
    assert false_edit["direction"] == "lower"
    assert false_edit["strictly_better"] is True
    assert result["test_assets_read"] is False
    assert (output / "controller.csv").is_file()
    assert (output / "writeback.csv").is_file()
    assert "Executable Writeback" in (output / "table.md").read_text()
    assert r"\begin{table*}" in (output / "table.tex").read_text()
