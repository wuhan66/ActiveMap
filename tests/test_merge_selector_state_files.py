from pathlib import Path

import pytest

from activemap.models import EditOperation
from activemap.selector_records import SelectorSample
from scripts.merge_selector_state_files import merge_state_files


def _sample(sample_id: str, split: str) -> SelectorSample:
    return SelectorSample(
        sample_id=sample_id,
        split=split,
        edit_type=EditOperation.KEEP,
        hypothesis_features=[0.0] * 16,
        state_features=[0.0] * 8,
        evidence_ids=["evidence"],
        evidence_features=[[0.0] * 13],
        evidence_costs=[1.0],
        false_edit_risks=[0.0],
        oracle_utilities=[-0.1],
        metadata={"source_episode": f"episode-{sample_id}"},
    )


def _write(path: Path, sample: SelectorSample) -> None:
    path.write_text(sample.model_dump_json() + "\n", encoding="utf-8")


def test_merge_state_files_preserves_train_val_and_summary(tmp_path: Path) -> None:
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    _write(train, _sample("train", "train"))
    _write(val, _sample("val", "val"))
    output = tmp_path / "combined.jsonl"
    summary = merge_state_files([train, val], output)
    assert summary["state_count"] == 2
    assert summary["counts"] == {
        "split:train": 1,
        "split:val": 1,
        "target:STOP": 2,
    }
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_merge_state_files_rejects_duplicate_ids(tmp_path: Path) -> None:
    left, right = tmp_path / "left.jsonl", tmp_path / "right.jsonl"
    _write(left, _sample("same", "train"))
    _write(right, _sample("same", "val"))
    with pytest.raises(ValueError, match="duplicate selector state"):
        merge_state_files([left, right], tmp_path / "combined.jsonl")
