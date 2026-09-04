from pathlib import Path

from activemap.agent.identifiers import public_task_id
from activemap.models import (
    CandidateHypothesis,
    EditOperation,
    EditRecord,
    EpisodeRecord,
)
from scripts.export_muno21_writeback_graphs import _episode_index_lookup


def _episode(index: int, split: str) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=f"muno21-{index}-p0-keep",
        split=split,
        source_dataset="MUNO21",
        map_before="before.graph",
        target_map="target.graph",
        hypothesis=CandidateHypothesis(
            op=EditOperation.KEEP,
            object_id=f"road-{index}",
            source="editable_prior",
        ),
        evidence_catalog=[],
        gt_edit=EditRecord(op=EditOperation.KEEP, object_id=f"road-{index}"),
        is_synthetic=False,
        derivation_version="test",
    )


def test_episode_lookup_filters_combined_train_val_manifest(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        _episode(1, "train").model_dump_json()
        + "\n"
        + _episode(2, "val").model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    lookup = _episode_index_lookup(episodes, expected_split="val")

    assert lookup == {public_task_id("muno21-2-p0-keep"): (2, 0)}
