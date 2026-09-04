import json
from pathlib import Path

import pytest

from scripts.export_sn7_step0_chronological_paper_artifacts import METRICS, load


def _summary() -> dict:
    return {
        "schema_version": "activemap-chronological-maintenance-three-seed-v1",
        "seed_count": 3,
        "test_assets_read": False,
        "aggregate": {
            metric: {
                "mean": 0.1,
                "hierarchical_ci95_low": 0.05,
                "hierarchical_ci95_high": 0.15,
            }
            for metric in METRICS
        },
    }


def test_load_requires_all_frozen_rows(tmp_path: Path) -> None:
    for label in ("0p5", "0p7", "0p9"):
        path = tmp_path / f"threshold_{label}" / "benefit" / "three_seed_v2.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_summary()), encoding="utf-8")
    for variant in ("notool", "forced"):
        path = tmp_path / "threshold_0p7" / variant / "three_seed_v2.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_summary()), encoding="utf-8")
    sensitivity, policies, sources = load(tmp_path)
    assert len(sensitivity) == 3
    assert len(policies) == 3
    assert len(sources) == 6


def test_load_rejects_test_access(tmp_path: Path) -> None:
    path = tmp_path / "threshold_0p5" / "benefit" / "three_seed_v2.json"
    path.parent.mkdir(parents=True)
    payload = _summary()
    payload["test_assets_read"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="validation"):
        load(tmp_path)
