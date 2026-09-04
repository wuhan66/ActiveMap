import json
from pathlib import Path

import pytest

from scripts.filter_episodes_by_assets import filter_episodes


def _row(episode_id: str, image_path: Path, split: str = "train") -> str:
    return json.dumps(
        {
            "episode_id": episode_id,
            "aoi_id": episode_id,
            "split": split,
            "evidence_catalog": [{"image_path": str(image_path), "udm_path": None}],
        }
    )


def test_filter_episodes_keeps_only_complete_remapped_assets(tmp_path: Path) -> None:
    old_root = Path("/old")
    new_root = tmp_path / "new"
    new_root.mkdir()
    (new_root / "present.tif").write_bytes(b"image")
    source = tmp_path / "episodes.jsonl"
    source.write_text(
        _row("keep", old_root / "present.tif")
        + "\n"
        + _row("drop", old_root / "missing.tif")
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "available.jsonl"
    summary = filter_episodes(
        source,
        output,
        splits={"train"},
        mappings=((old_root, new_root),),
    )
    assert summary["kept_episodes"] == 1
    assert summary["rejected_missing_assets"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["episode_id"] == "keep"


def test_filter_episodes_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "available.jsonl"
    output.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        filter_episodes(
            tmp_path / "missing.jsonl", output, splits={"train"}, mappings=()
        )
