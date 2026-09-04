from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_sn7_concat_unet_capacity import summarize


def _summary(path: Path, *, iou: float, accuracy: float, false_edit: float) -> Path:
    payload = {
        "split": "val",
        "sample_count": 100,
        "aoi_count": 4,
        "checkpoint": str(path.with_suffix(".pt")),
        "edit_accuracy": accuracy,
        "macro_f1": 0.7,
        "update_f1": 0.7,
        "false_edit_rate": false_edit,
        "missed_update_rate": 0.1,
        "mean_raster_iou": iou,
        "mean_polygon_iou": 0.5,
        "topology_valid_rate": 0.99,
        "ece": 0.1,
        "per_edit": {
            name: {"f1": 0.7} for name in ("KEEP", "ADD", "DELETE", "RESHAPE")
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_capacity_gate_promotes_only_safe_meaningful_gain(tmp_path: Path) -> None:
    specs = [
        (32, 2_930_448, _summary(tmp_path / "w32.json", iou=0.70, accuracy=0.90, false_edit=0.10), tmp_path),
        (48, 6_586_000, _summary(tmp_path / "w48.json", iou=0.704, accuracy=0.896, false_edit=0.104), tmp_path),
        (64, 11_701_776, _summary(tmp_path / "w64.json", iou=0.71, accuracy=0.89, false_edit=0.10), tmp_path),
    ]
    payload = summarize(specs)
    assert payload["selected_width"] == 48
    assert payload["three_seed_confirmation_required"] is True
    rows = {row["base_channels"]: row for row in payload["runs"]}
    assert rows[48]["pilot_promotion_passed"] is True
    assert rows[64]["pilot_promotion_passed"] is False


def test_capacity_gate_keeps_width32_when_wider_models_fail(tmp_path: Path) -> None:
    specs = [
        (32, 2_930_448, _summary(tmp_path / "w32.json", iou=0.70, accuracy=0.90, false_edit=0.10), tmp_path),
        (48, 6_586_000, _summary(tmp_path / "w48.json", iou=0.702, accuracy=0.90, false_edit=0.10), tmp_path),
        (64, 11_701_776, _summary(tmp_path / "w64.json", iou=0.71, accuracy=0.89, false_edit=0.12), tmp_path),
    ]
    payload = summarize(specs)
    assert payload["selected_width"] == 32
    assert payload["three_seed_confirmation_required"] is False
