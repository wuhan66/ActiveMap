import json
from pathlib import Path

from scripts.audit_episode_assets import audit, parse_asset_root_maps


def test_audit_episode_assets_remaps_only_requested_split(tmp_path: Path) -> None:
    source = tmp_path / "source" / "sn7"
    target = tmp_path / "sn7"
    image = target / "aoi" / "image.tif"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    episodes = tmp_path / "episodes.jsonl"
    rows = [
        {
            "split": "val",
            "evidence_catalog": [{"image_path": str(source / "aoi/image.tif")}],
        },
        {
            "split": "test",
            "evidence_catalog": [{"image_path": "/must/not/read.tif"}],
        },
    ]
    episodes.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = audit(
        episodes,
        split="val",
        mappings=parse_asset_root_maps([f"{source}={target}"]),
    )

    assert report["passed"] is True
    assert report["episodes"] == 1
    assert report["unique_assets"] == 1
    assert report["test_assets_read"] is False


def test_audit_episode_assets_fails_on_missing_file(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        json.dumps({"split": "val", "map_before": "/missing/map.geojson"}) + "\n",
        encoding="utf-8",
    )

    report = audit(episodes, split="val", mappings=())

    assert report["passed"] is False
    assert report["missing_assets"] == 1
