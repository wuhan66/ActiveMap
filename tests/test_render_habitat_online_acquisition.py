from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def load_renderer():
    script = Path(__file__).resolve().parents[1] / "scripts" / "render_habitat_online_acquisition.py"
    spec = importlib.util.spec_from_file_location("habitat_online_renderer", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_panel_status_reports_policy_specific_calls_and_quality() -> None:
    renderer = load_renderer()

    assert renderer.panel_status(None) == "reference"
    assert (
        renderer.panel_status({"sensor_calls": 7, "final_reference_map_quality": 0.875})
        == "7 views | agreement 0.875"
    )


def test_render_supports_independent_summary_per_panel(tmp_path: Path, monkeypatch) -> None:
    renderer = load_renderer()
    reference = np.array([[0.5, 0.0], [1.0, 0.5]], dtype=np.float32)
    map_a = reference.copy()
    map_b = np.array([[0.5, 0.0], [0.5, 0.5]], dtype=np.float32)
    reference_path = tmp_path / "reference.npy"
    map_a_path = tmp_path / "a.npy"
    map_b_path = tmp_path / "b.npy"
    np.save(reference_path, reference)
    np.save(map_a_path, map_a)
    np.save(map_b_path, map_b)

    def summary(calls: int, acquired: bool) -> dict[str, object]:
        return {
            "sensor_calls": calls,
            "final_reference_map_quality": 1.0 if acquired else 0.75,
            "grid": {"x_min": 0.0, "z_max": 2.0, "resolution_m": 1.0},
            "start_position": [0.5, 0.0, 1.5],
            "goal_position": [1.5, 0.0, 0.5],
            "trace": [{"position": [1.5, 0.0, 0.5], "acquired": acquired}],
        }

    summary_a = tmp_path / "summary_a.json"
    summary_b = tmp_path / "summary_b.json"
    summary_a.write_text(json.dumps(summary(4, True)), encoding="utf-8")
    summary_b.write_text(json.dumps(summary(2, False)), encoding="utf-8")
    output = tmp_path / "comparison.png"
    monkeypatch.setattr(
        "sys.argv",
        [
            "render_habitat_online_acquisition.py",
            str(output),
            "--reference",
            str(reference_path),
            "--panel",
            f"Acquire={map_a_path}",
            "--panel",
            f"Gate={map_b_path}",
            "--panel-summary",
            f"Acquire={summary_a}",
            "--panel-summary",
            f"Gate={summary_b}",
            "--panel-width",
            "100",
        ],
    )

    renderer.main()

    assert output.is_file()
    assert output.stat().st_size > 0
