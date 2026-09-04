from pathlib import Path

import pandas as pd
from rasterio.transform import from_origin
from shapely.geometry import box

from activemap.data.edits import EditEvent
from activemap.data.episodes import build_episode, write_episodes
from activemap.data.evidence import build_evidence_catalog, evidence_cost
from activemap.models import EditOperation


def test_evidence_catalog_and_episode_round_trip(tmp_path: Path) -> None:
    transform = list(from_origin(0, 100, 1, 1))
    manifest = pd.DataFrame(
        {
            "aoi_id": ["aoi-1", "aoi-1"],
            "timestamp": ["2019_01", "2019_02"],
            "image_path": ["jan.tif", "feb.tif"],
            "udm_path": [None, "feb_udm.tif"],
            "width": [100, 100],
            "height": [100, 100],
            "transform": [transform, transform],
            "clear_fraction": [0.7, 1.0],
        }
    )
    geometry = box(10, 10, 20, 20)
    catalog = build_evidence_catalog(
        manifest,
        aoi_id="aoi-1",
        anchor_timestamp="2019_02",
        geometry=geometry,
    )
    assert len(catalog) == 6
    assert len({item.evidence_id for item in catalog}) == 6
    assert min(item.cost for item in catalog) >= 1.0
    assert evidence_cost(scale=1, temporal_distance=0, clear_fraction=1.0) == 1.0

    event = EditEvent(
        op=EditOperation.RESHAPE,
        object_id="obj-1",
        old_geometry=box(9, 10, 19, 20),
        new_geometry=geometry,
        iou=0.8,
        match_source="persistent_id",
    )
    episode = build_episode(
        episode_id="episode-1",
        split="train",
        source_dataset="spacenet7",
        map_before="before.geojson",
        target_map="target.geojson",
        event=event,
        evidence_catalog=catalog,
        hypothesis_source="test",
        hypothesis_confidence=0.8,
        is_synthetic=False,
        derivation_version="test-v1",
    )
    output = tmp_path / "episodes.jsonl"
    write_episodes([episode], output)
    assert output.read_text(encoding="utf-8").count("\n") == 1
