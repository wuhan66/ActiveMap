import json
from pathlib import Path

from scripts.build_active_catalog_balanced_sentinel import (
    build_sentinel,
    materialize_images,
)


def _row(example_id: str, selection: str) -> dict:
    return {
        "example_id": example_id,
        "split": "val",
        "messages": [
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps({"selection": selection})}]}
        ],
    }


def test_sentinel_keeps_all_acquire_and_matches_stop_per_aoi() -> None:
    rows = [_row("a1", "ACQUIRE"), _row("a2", "STOP"), _row("a3", "STOP"), _row("b1", "ACQUIRE"), _row("b2", "STOP")]
    index = [
        {"example_id": row["example_id"], "aoi_id": "A" if row["example_id"].startswith("a") else "B"}
        for row in rows
    ]

    selected, selected_index, summary = build_sentinel(rows, index, seed=7)

    assert {row["example_id"] for row in selected} >= {"a1", "b1"}
    assert len(selected) == len(selected_index) == 4
    assert summary["action_counts"] == {"ACQUIRE": 2, "STOP": 2}
    assert summary["diagnostic_only"] is True
    assert summary["test_assets_read"] is False


def test_sentinel_can_cap_acquire_records_per_aoi() -> None:
    rows = [
        _row("a1", "ACQUIRE"),
        _row("a2", "ACQUIRE"),
        _row("a3", "STOP"),
        _row("a4", "STOP"),
    ]
    index = [{"example_id": row["example_id"], "aoi_id": "A"} for row in rows]

    selected, _, summary = build_sentinel(
        rows, index, seed=7, max_acquire_per_aoi=1
    )

    assert len(selected) == 2
    assert summary["action_counts"] == {"ACQUIRE": 1, "STOP": 1}
    assert summary["max_acquire_per_aoi"] == 1


def test_sentinel_materializes_relative_images_with_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "images").mkdir(parents=True)
    image = source / "images" / "task.jpg"
    image.write_bytes(b"image-bytes")
    rows = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "images/task.jpg"}],
                }
            ]
        }
    ]

    manifest = materialize_images(rows, source, output)

    assert manifest["count"] == 1
    assert len(manifest["manifest_sha256"]) == 64
    assert (output / "images" / "task.jpg").read_bytes() == b"image-bytes"
