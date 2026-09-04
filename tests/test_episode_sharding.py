from __future__ import annotations

from test_episode_support_tools import _episode

from activemap.models import EditOperation
from scripts.merge_episode_shards import merge_episode_shards
from scripts.shard_episodes import shard_episodes, shard_index


def test_shard_index_is_deterministic_and_bounded() -> None:
    first = [shard_index(f"episode-{i}", seed=17, num_shards=8) for i in range(100)]
    second = [shard_index(f"episode-{i}", seed=17, num_shards=8) for i in range(100)]
    assert first == second
    assert min(first) >= 0
    assert max(first) < 8
    assert len(set(first)) == 8


def test_merge_episode_shards_preserves_train_val_boundary(tmp_path) -> None:
    rows = [
        _episode(split, EditOperation.KEEP, f"aoi-{index}", index)
        for index, split in enumerate(["train", "val"] * 10)
    ]
    rows.append(_episode("test", EditOperation.ADD, "held-out", 99))
    source = tmp_path / "episodes.jsonl"
    source.write_text("".join(row.model_dump_json() + "\n" for row in rows))
    shard_root = tmp_path / "shards"
    shard_episodes(source, shard_root, num_shards=2, seed=17)
    output = tmp_path / "train_val.jsonl"
    report = merge_episode_shards(shard_root, output)
    assert report["episode_count"] == 20
    assert report["test_assets_read"] is False
    assert '"split":"test"' not in output.read_text()
