import json
from pathlib import Path

from scripts.build_sn7_controller_intervention_assets import build


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _comparison(metric_values: dict[str, tuple[float, float, float]]) -> dict:
    return {
        "paired_delta": {
            key: {"delta": value[0], "ci95_low": value[1], "ci95_high": value[2]}
            for key, value in metric_values.items()
        },
        "test_assets_read": False,
    }


def _condition(root: Path, *, passes: bool) -> None:
    quality = (0.2, 0.1, 0.3)
    false_edit = (-0.1, -0.2, -0.05)
    cost = (-0.02, -0.03, -0.01) if passes else (-0.01, -0.02, 0.01)
    _write(
        root / "benefit_vs_notool.json",
        _comparison(
            {
                "raster_iou_auc": quality,
                "false_edit_auc": false_edit,
                "missed_edit_auc": (0.0, -0.01, 0.01),
            }
        ),
    )
    _write(
        root / "benefit_vs_forced.json",
        _comparison({"spent_cost_auc": cost}),
    )
    _write(
        root / "controller_summary.json",
        {
            "per_seed": [
                {
                    "variant": "benefit",
                    "tool_call_episode_rate": 0.2,
                    "mean_tool_calls": 0.4,
                },
                {
                    "variant": "benefit",
                    "tool_call_episode_rate": 0.3,
                    "mean_tool_calls": 0.6,
                },
            ],
            "test_assets_read": False,
        },
    )


def test_build_intervention_assets_distinguishes_strict_gate(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _condition(first, passes=True)
    _condition(second, passes=False)

    payload = build([("first", first), ("second", second)])

    assert payload["rows"][0]["checks"]["strict_gate_passed"] is True
    assert payload["rows"][1]["checks"]["strict_gate_passed"] is False
    assert payload["all_conditions_strict"] is False
    assert payload["rows"][0]["tool_call_episode_rate"] == 0.25
    assert len(payload["rows"][0]["inputs"]["controller"]["sha256"]) == 64
