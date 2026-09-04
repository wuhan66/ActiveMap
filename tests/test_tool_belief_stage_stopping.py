import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_tool_belief_stage_stopping import _costs, analyze_stage_choices


def test_stage_costs_are_cumulative_and_monotonic(tmp_path: Path) -> None:
    path = tmp_path / "details.jsonl"
    rows = [
        {"sequence_id": "one", "split": "train", "step": step, "spent_cost": 0.18 * step}
        for step in range(1, 4)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert _costs(path, "train") == {"one": [0.0, 0.18, 0.36, 0.54]}


def test_stage_oracle_charges_cost_and_prefers_earlier_tie() -> None:
    result = analyze_stage_choices(
        ["one", "two"],
        np.asarray([1, 1, 1, 1, 0, 0, 0, 0]),
        np.asarray([0, 1, 1, 1, 0, 1, 0, 0]),
        np.full(8, 0.8),
        np.asarray([0.0, 0.18, 0.36, 0.54, 0.0, 0.18, 0.36, 0.54]),
    )

    assert result["oracle_selected_stage_counts"] == {"0": 1, "1": 1, "2": 0, "3": 0}
    assert result["oracle"]["mean_tool_cost"] == pytest.approx(0.09)
    assert result["oracle"]["mean_joint_utility"] == pytest.approx(0.91)
    assert result["fixed_stages"]["stage_3"]["mean_joint_utility"] == pytest.approx(0.46)
