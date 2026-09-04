from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _episode(episode_id: str, split: str) -> str:
    return json.dumps({"episode_id": episode_id, "split": split}) + "\n"


def _state(sample_id: str, source_episode: str, split: str = "val") -> str:
    return json.dumps(
        {"sample_id": sample_id, "split": split, "metadata": {"source_episode": source_episode}}
    ) + "\n"


def _run(script: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_validation_shards_are_test_free_and_aggregate_cleanly(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        _episode("train-0", "train")
        + _episode("val-0", "val")
        + _episode("val-1", "val")
        + _episode("val-2", "val"),
        encoding="utf-8",
    )
    shard_dir = tmp_path / "shards"
    _run("shard_validation_selector_episodes.py", str(episodes), str(shard_dir), "--shards", "2")
    receipt = json.loads((shard_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["source_rows"] == {"total": 4, "train": 1, "val": 3, "test": 0}
    assert receipt["test_assets_read"] is False

    for index in range(2):
        source = shard_dir / f"episodes_val_shard_{index:02d}.jsonl"
        states = shard_dir / f"selector_states_val_shard_{index:02d}.jsonl"
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        states.write_text(
            "".join(_state(f"{row['episode_id']}__b1", row["episode_id"]) for row in records),
            encoding="utf-8",
        )

    output = tmp_path / "states.jsonl"
    _run("combine_validation_selector_states.py", str(shard_dir), str(output), "--shards", "2")
    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["validation_episodes"] == 3
    assert summary["state_source_episodes"] == 3
    assert summary["state_rows"] == 3
    assert summary["test_assets_read"] is False


def test_sharding_rejects_test_records(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(_episode("test-0", "test"), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "shard_validation_selector_episodes.py"),
            str(episodes),
            str(tmp_path / "shards"),
            "--shards",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "refuses a source manifest with test rows" in completed.stderr
