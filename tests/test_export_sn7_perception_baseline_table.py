from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.export_sn7_perception_baseline_table import build_table


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _concat_payload() -> dict:
    def metric(value: float) -> dict:
        return {"mean": value, "sample_std": 0.01, "per_seed": {}}

    return {
        "schema_version": "sn7-concat-unet-three-seed-validation-v1",
        "split": "val",
        "test_assets_read": False,
        "seed_count": 3,
        "sample_count_per_seed": 100,
        "aoi_count": 4,
        "aggregate": {
            "mean_raster_iou": metric(0.70),
            "edit_accuracy": metric(0.80),
            "false_edit_rate": metric(0.10),
        },
    }


def _comparison_payload() -> dict:
    def observed(value: float) -> dict:
        return {"observed": value, "ci_low": value - 0.01, "ci_high": value + 0.01}

    return {
        "schema_version": "sn7-updater-paired-aoi-bootstrap-v1",
        "sample_count": 100,
        "aoi_count": 4,
        "test_assets_read": False,
        "results": {
            "changemamba": {
                "committed_map_iou": observed(0.80),
                "map_iou_delta": observed(0.20),
                "operation_accuracy": observed(0.90),
                "false_edit_rate": observed(0.05),
            },
            "ban": {
                "committed_map_iou": observed(0.55),
                "map_iou_delta": observed(-0.05),
                "operation_accuracy": observed(0.60),
                "false_edit_rate": observed(0.40),
            },
        },
    }


def test_build_table_aligns_shared_validation_support(tmp_path: Path) -> None:
    concat = _write(tmp_path / "concat.json", _concat_payload())
    external = _write(tmp_path / "external.json", _comparison_payload())
    table = build_table(concat, external)
    assert table["test_assets_read"] is False
    assert [row["method"] for row in table["rows"]] == [
        "Concat U-Net",
        "ChangeMamba",
        "BAN (Open-CD)",
    ]
    assert table["prior_map_iou"] == pytest.approx(0.60)
    assert table["rows"][0]["map_iou_gain"] == pytest.approx(0.10)


def test_build_table_rejects_test_or_mismatched_support(tmp_path: Path) -> None:
    concat_payload = _concat_payload()
    concat_payload["test_assets_read"] = True
    concat = _write(tmp_path / "concat.json", concat_payload)
    external = _write(tmp_path / "external.json", _comparison_payload())
    with pytest.raises(ValueError, match="validation-only"):
        build_table(concat, external)

    concat_payload["test_assets_read"] = False
    concat_payload["sample_count_per_seed"] = 99
    _write(concat, concat_payload)
    with pytest.raises(ValueError, match="validation support"):
        build_table(concat, external)
