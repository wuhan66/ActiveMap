import json
from pathlib import Path
from types import SimpleNamespace

from scripts import build_active_catalog_validation_visual_index as module


def test_builds_label_free_validation_index(tmp_path, monkeypatch):
    states_path = tmp_path / "states.jsonl"
    episodes_path = tmp_path / "episodes.jsonl"
    states_path.write_text("{}\n", encoding="utf-8")
    episodes_path.write_text("{}\n", encoding="utf-8")
    states = [
        SimpleNamespace(
            split="val",
            metadata={
                "oracle_step": 0,
                "source_episode": episode,
                "initial_evidence_id": f"{episode}-initial",
            },
        )
        for episode in ("e1", "e2")
    ]
    episodes = [
        SimpleNamespace(episode_id=episode, split="val")
        for episode in ("e1", "e2")
    ]

    def fake_read(path, _model):
        return states if path == states_path else episodes

    def fake_render(_episode, _evidence, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"image")

    monkeypatch.setattr(module, "_read_jsonl", fake_read)
    monkeypatch.setattr(module, "_remap_episodes", lambda rows, _maps: rows)
    monkeypatch.setattr(module, "_render_initial_state", fake_render)
    output = tmp_path / "output"

    summary = module.build(states_path, episodes_path, output)

    prompts = [
        json.loads(line)
        for line in (output / "val_visual_prompts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    index = [
        json.loads(line)
        for line in (output / "val_visual_index.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert summary["assistant_targets_present"] is False
    assert summary["test_assets_read"] is False
    assert len(prompts) == len(index) == 2
    assert all(
        [item["role"] for item in row["messages"]] == ["system", "user"]
        for row in prompts
    )
    assert all(row["test_assets_read"] is False for row in [*prompts, *index])


def test_rejects_inconsistent_initial_evidence(tmp_path, monkeypatch):
    states_path = tmp_path / "states.jsonl"
    episodes_path = tmp_path / "episodes.jsonl"
    states_path.write_text("{}\n", encoding="utf-8")
    episodes_path.write_text("{}\n", encoding="utf-8")
    states = [
        SimpleNamespace(
            split="val",
            metadata={
                "oracle_step": 0,
                "source_episode": "e1",
                "initial_evidence_id": evidence,
            },
        )
        for evidence in ("a", "b")
    ]
    episodes = [SimpleNamespace(episode_id="e1", split="val")]
    monkeypatch.setattr(
        module,
        "_read_jsonl",
        lambda path, _model: states if path == states_path else episodes,
    )
    monkeypatch.setattr(module, "_remap_episodes", lambda rows, _maps: rows)

    try:
        module.build(states_path, episodes_path, tmp_path / "output")
    except ValueError as error:
        assert "inconsistent initial evidence" in str(error)
    else:
        raise AssertionError("expected inconsistent initial evidence to fail")
