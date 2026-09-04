from activemap.models import EpisodeRecord, EvidenceItem
from scripts.evaluate_agent_rollouts import (
    _parse_asset_root_maps,
    _remap_episode_assets,
)


def test_episode_asset_paths_are_remapped_without_mutating_source() -> None:
    episode = EpisodeRecord.model_construct(
        episode_id="episode-1",
        split="val",
        evidence_catalog=[
            EvidenceItem(
                evidence_id="evidence-1",
                image_path="/mnt/mydisk/wh/ActiveMap/datasets/example.jpg",
                timestamp="2020-01-01T00:00:00Z",
                region=[0, 0, 16, 16],
                scale=1,
                cost=1.0,
                clear_fraction=1.0,
            )
        ],
    )
    mappings = _parse_asset_root_maps(
        ["/mnt/mydisk/wh/ActiveMap=/home/wh/ActiveMap"]
    )

    remapped = _remap_episode_assets(episode, mappings)

    assert (
        remapped.evidence_catalog[0].image_path
        == "/home/wh/ActiveMap/datasets/example.jpg"
    )
    assert episode.evidence_catalog[0].image_path.startswith("/mnt/mydisk")
