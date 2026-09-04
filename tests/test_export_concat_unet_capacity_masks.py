from __future__ import annotations

import json
from pathlib import Path

from scripts.export_concat_unet_capacity_masks import select_samples


def _write_predictions(path: Path, method_offset: float) -> Path:
    rows = []
    for edit_index, edit in enumerate(("KEEP", "ADD", "DELETE", "RESHAPE")):
        for index in range(3):
            rows.append(
                {
                    "sample_id": f"{edit.lower()}-{index}",
                    "target_edit": edit,
                    "predicted_edit": edit,
                    "raster_iou": 0.9 - index * method_offset - edit_index * 0.01,
                }
            )
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def test_select_samples_is_balanced_and_disagreement_ranked(tmp_path: Path) -> None:
    first = _write_predictions(tmp_path / "first.jsonl", 0.01)
    second = _write_predictions(tmp_path / "second.jsonl", 0.05)
    rows = select_samples(
        [("width32", first), ("width64", second)],
        per_edit=1,
    )
    assert [row["target_edit"] for row in rows] == [
        "KEEP",
        "ADD",
        "DELETE",
        "RESHAPE",
    ]
    assert all(row["sample_id"].endswith("-2") for row in rows)
    assert all(row["disagreement"] > 0 for row in rows)
