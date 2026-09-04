import csv
import json
from pathlib import Path

import pytest

from scripts.plot_sn7_step0_quality_cost_safety import METHOD_ORDER, load_inputs


def _write_csv(path: Path, fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "variant", *fields])
        writer.writeheader()
        for index, method in enumerate(METHOD_ORDER):
            writer.writerow(
                {
                    "method": method,
                    "variant": str(index),
                    **{field: 0.1 + index for field in fields},
                }
            )


def test_load_inputs_requires_promoted_validation_tables(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "split": "val",
                "test_assets_read": False,
                "promotion_passed": True,
            }
        ),
        encoding="utf-8",
    )
    _write_csv(tmp_path / "controller_table.csv", ["terminal_accuracy_mean"])
    _write_csv(tmp_path / "writeback_table.csv", ["raster_iou_auc"])
    loaded = load_inputs(tmp_path)
    assert list(loaded["controller"]) == list(METHOD_ORDER)


def test_load_inputs_rejects_test_manifest(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "split": "test",
                "test_assets_read": True,
                "promotion_passed": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="validation-only"):
        load_inputs(tmp_path)
