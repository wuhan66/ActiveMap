import json
from pathlib import Path

import pytest

from scripts.export_sn7_step0_paper_tables import export


METRICS = {
    "raster_iou_auc": 0.5,
    "raster_iou_gain_auc": 0.01,
    "false_edit_auc": 0.02,
    "missed_edit_auc": 0.03,
    "spent_cost_auc": 0.04,
    "episode_utility_v2_balanced_auc": 0.05,
    "episode_utility_v2_safety_auc": 0.06,
    "episode_utility_v2_cost_aware_auc": 0.07,
    "vector_replay_iou_auc": 1.0,
    "vector_delta_topology_valid_auc": 1.0,
}


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _summary() -> dict:
    rows = []
    for seed in (1, 2, 3):
        for index, variant in enumerate(("notool", "forced", "benefit")):
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "terminal_accuracy": 0.5 + 0.01 * index,
                    "false_edit_rate": 0.2 - 0.01 * index,
                    "missed_edit_rate": 0.1,
                    "mean_quality_gain": 0.01,
                    "mean_quality_cost_utility": 0.02,
                    "mean_tool_calls": float(index),
                    "tool_call_episode_rate": 0.1 * index,
                    "mean_tool_belief_l1_delta": 0.01 * index,
                }
            )
    return {
        "seeds": [1, 2, 3],
        "per_seed": rows,
        "comparisons": {"benefit_vs_notool": {}, "benefit_vs_forced": {}},
        "protocol": {"split": "val", "test_assets_read": False},
    }


def _comparison(baseline_delta: float) -> dict:
    baseline = {key: value + baseline_delta for key, value in METRICS.items()}
    return {
        "model_seeds": [1, 2, 3],
        "baseline": baseline,
        "candidate": dict(METRICS),
        "paired_delta": {
            key: {"delta": METRICS[key] - baseline[key], "ci95_low": -0.01, "ci95_high": 0.01}
            for key in METRICS
        },
        "split": "val",
        "test_assets_read": False,
    }


def test_export_writes_submission_tables(tmp_path: Path) -> None:
    manifest = export(
        _write(tmp_path / "summary.json", _summary()),
        _write(tmp_path / "notool.json", _comparison(-0.01)),
        _write(tmp_path / "forced.json", _comparison(0.01)),
        _write(
            tmp_path / "promotion.json",
            {
                "promote": True,
                "claim_boundary": "Validation-only evidence.",
                "test_assets_read": False,
            },
        ),
        tmp_path / "tables",
    )
    assert manifest["promotion_passed"] is True
    assert manifest["seeds"] == [1, 2, 3]
    assert (tmp_path / "tables" / "controller_table.tex").exists()
    assert "ActiveMap (benefit-aware)" in (
        tmp_path / "tables" / "writeback_table.md"
    ).read_text(encoding="utf-8")
    assert len(manifest["inputs"]["controller_summary"]["sha256"]) == 64


def test_export_rejects_test_input(tmp_path: Path) -> None:
    summary = _summary()
    summary["protocol"]["split"] = "test"
    with pytest.raises(ValueError, match="validation-only"):
        export(
            _write(tmp_path / "summary.json", summary),
            _write(tmp_path / "notool.json", _comparison(-0.01)),
            _write(tmp_path / "forced.json", _comparison(0.01)),
            _write(
                tmp_path / "promotion.json",
                {"promote": True, "split": "val", "test_assets_read": False},
            ),
            tmp_path / "tables",
        )
