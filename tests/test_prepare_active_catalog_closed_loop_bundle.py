import hashlib
import json

import pytest

from activemap.models import (
    CandidateHypothesis,
    EditOperation,
    EditRecord,
    EpisodeRecord,
    EvidenceItem,
    GeoJSONGeometry,
)
from activemap.selector_records import SelectorSample
from scripts.prepare_active_catalog_closed_loop_bundle import prepare


def _episode():
    geometry = GeoJSONGeometry(
        type="Polygon",
        coordinates=[[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    )
    return EpisodeRecord(
        episode_id="val-RESHAPE-a-0",
        aoi_id="a",
        split="val",
        source_dataset="fixture",
        map_before="/unused/before.geojson",
        target_map="/unused/after.geojson",
        hypothesis=CandidateHypothesis(
            op=EditOperation.RESHAPE,
            object_id="object",
            geometry=geometry,
            source="fixture",
        ),
        gt_edit=EditRecord(
            op=EditOperation.RESHAPE,
            object_id="object",
            geometry=geometry,
        ),
        evidence_catalog=[
            EvidenceItem(
                evidence_id="e-0",
                timestamp="2026_01",
                region=(0, 0, 16, 16),
                scale=1,
                image_path="/unused/image.tif",
                clear_fraction=1.0,
                cost=1.0,
            )
        ],
        is_synthetic=False,
        derivation_version="fixture-v1",
    )


def _sample():
    return SelectorSample(
        sample_id="initial",
        split="val",
        edit_type=EditOperation.RESHAPE,
        hypothesis_features=[0.0] * 16,
        state_features=[0.0] * 8,
        evidence_ids=["e-0"],
        evidence_features=[[0.0] * 13],
        evidence_costs=[1.0],
        false_edit_risks=[0.0],
        oracle_utilities=[0.2],
    )


def test_bundle_keeps_only_validation_initial_states(tmp_path):
    episode = _episode()
    state = _sample()
    state.metadata.update(
        {
            "source_episode": episode.episode_id,
            "aoi_id": "a",
            "oracle_step": 0,
            "budget": 3.0,
            "evidence_predictions": {
                "e-0": {
                    "edit_probabilities": [0.1, 0.2, 0.3, 0.4],
                    "confidence": 0.5,
                    "geometry_delta": [0.0] * 8,
                }
            },
        }
    )
    later = state.model_copy(
        update={
            "sample_id": "later",
            "metadata": {**state.metadata, "oracle_step": 1},
        }
    )
    states = tmp_path / "states.jsonl"
    states.write_text(
        state.model_dump_json() + "\n" + later.model_dump_json() + "\n",
        encoding="utf-8",
    )
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(episode.model_dump_json() + "\n", encoding="utf-8")
    output = tmp_path / "bundle"
    summary = prepare(states, episodes, output)
    assert summary["states"] == 1
    assert summary["episodes"] == 1
    assert summary["test_assets_read"] is False
    assert len((output / "states_val_step0.jsonl").read_text().splitlines()) == 1


def test_bundle_supports_train_without_test_access(tmp_path):
    episode = _episode().model_copy(update={"episode_id": "train-a", "split": "train"})
    state = _sample().model_copy(update={"sample_id": "train-state", "split": "train"})
    state.metadata.update(
        {
            "source_episode": episode.episode_id,
            "aoi_id": "a",
            "oracle_step": 0,
            "budget": 3.0,
            "evidence_predictions": {"e-0": {}},
        }
    )
    states = tmp_path / "states.jsonl"
    states.write_text(state.model_dump_json() + "\n", encoding="utf-8")
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(episode.model_dump_json() + "\n", encoding="utf-8")

    output = tmp_path / "bundle"
    summary = prepare(states, episodes, output, split="train")

    assert summary["split"] == "train"
    assert summary["test_assets_read"] is False
    assert (output / "states_train_step0.jsonl").is_file()
    assert (output / "episodes_train.jsonl").is_file()


def test_frozen_bundle_requires_and_records_one_shot_access(tmp_path, monkeypatch):
    episode = _episode().model_copy(update={"episode_id": "test-a", "split": "test"})
    state = _sample().model_copy(update={"sample_id": "test-state", "split": "test"})
    state.metadata.update(
        {
            "source_episode": episode.episode_id,
            "aoi_id": "a",
            "oracle_step": 0,
            "budget": 3.0,
            "evidence_predictions": {"e-0": {}},
        }
    )
    states = tmp_path / "states.jsonl"
    states.write_text(state.model_dump_json() + "\n", encoding="utf-8")
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(episode.model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="--frozen-test"):
        prepare(states, episodes, tmp_path / "blocked", split="test")

    token = "one-shot"
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": "activemap-frozen-test-access-v1",
                "status": "started",
                "authorization_sha256": hashlib.sha256(token.encode()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ACTIVEMAP_FROZEN_TEST", "1")
    monkeypatch.setenv("ACTIVEMAP_FROZEN_TEST_LEDGER", str(ledger))
    monkeypatch.setenv("ACTIVEMAP_FROZEN_TEST_TOKEN", token)
    output = tmp_path / "bundle"
    summary = prepare(states, episodes, output, split="test", frozen_test=True)
    assert summary["test_assets_read"] is True
    assert (output / "states_test_step0.jsonl").is_file()
    assert (output / "episodes_test.jsonl").is_file()
