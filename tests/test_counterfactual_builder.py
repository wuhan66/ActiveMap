from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from activemap.models import (  # noqa: E402
    CandidateHypothesis,
    EditOperation,
    EditRecord,
    EpisodeRecord,
    EvidenceItem,
    GeoJSONGeometry,
)
from activemap.nn.updater import PriorConditionedUNet, UpdaterConfig  # noqa: E402
from activemap.oracle.updater_counterfactual import (  # noqa: E402
    _expand_budget_states,
    build_selector_oracle_input_cache,
    build_selector_oracle_samples,
    remap_episode_assets,
)
from activemap.selector_records import SelectorSample  # noqa: E402
from activemap.training.data import load_selector_samples  # noqa: E402


def _box(minimum: float, maximum: float) -> GeoJSONGeometry:
    return GeoJSONGeometry(
        type="Polygon",
        coordinates=[
            [
                [minimum, minimum],
                [maximum, minimum],
                [maximum, maximum],
                [minimum, maximum],
                [minimum, minimum],
            ]
        ],
    )


def test_test_selector_oracle_requires_explicit_frozen_authorization(
    tmp_path: Path,
) -> None:
    with pytest.raises(PermissionError, match="requires --frozen-test"):
        build_selector_oracle_samples(
            tmp_path / "missing.pt",
            tmp_path / "missing.jsonl",
            tmp_path / "states.jsonl",
            splits=("test",),
        )


def test_updater_counterfactual_builds_budget_states(tmp_path: Path) -> None:
    image_path = tmp_path / "image.tif"
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 32, 1, 1),
    ) as dataset:
        dataset.write(np.full((3, 32, 32), 100, dtype=np.uint8))

    prior = _box(8, 16)
    target = _box(10, 18)
    episode = EpisodeRecord(
        episode_id="episode",
        aoi_id="aoi",
        anchor_timestamp="2019_02",
        split="train",
        source_dataset="test",
        map_before="old.geojson",
        target_map="new.geojson",
        prior_geometry=prior,
        target_geometry=target,
        hypothesis=CandidateHypothesis(
            op=EditOperation.RESHAPE,
            object_id="building",
            geometry=target,
            source="test",
            confidence=0.5,
        ),
        evidence_catalog=[
            EvidenceItem(
                evidence_id="jan",
                timestamp="2019_01",
                region=(0, 0, 32, 32),
                scale=1,
                image_path=str(image_path),
                clear_fraction=1.0,
                cost=1.0,
            ),
            EvidenceItem(
                evidence_id="feb",
                timestamp="2019_02",
                region=(0, 0, 32, 32),
                scale=2,
                image_path=str(image_path),
                clear_fraction=0.8,
                cost=1.5,
            ),
        ],
        gt_edit=EditRecord(
            op=EditOperation.RESHAPE,
            object_id="building",
            geometry=target,
        ),
        is_synthetic=False,
        derivation_version="test",
    )
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(episode.model_dump_json() + "\n", encoding="utf-8")
    model = PriorConditionedUNet(UpdaterConfig(base_channels=4, dropout=0.0))
    checkpoint_path = tmp_path / "updater.pt"
    torch.save(
        {"state_dict": model.state_dict(), "model_config": model.config.as_dict()},
        checkpoint_path,
    )
    output = tmp_path / "selector.jsonl"
    summary = build_selector_oracle_samples(
        checkpoint_path,
        episodes_path,
        output,
        device="cpu",
        image_size=16,
        budgets=(2.0, 3.0),
        max_steps=2,
    )
    samples = load_selector_samples(output)
    assert summary["samples"] == len(samples)
    assert len(samples) >= 2
    assert {sample.metadata["budget"] for sample in samples} == {2.0, 3.0}
    assert all(sample.evidence_features[0][-1] > 0 for sample in samples)


def test_executable_budget_utility_is_incremental_and_budget_normalized() -> None:
    sample = SelectorSample(
        sample_id="episode",
        split="train",
        edit_type=EditOperation.KEEP,
        hypothesis_features=[0.0] * 16,
        state_features=[0.0] * 8,
        evidence_ids=["initial", "helpful", "harmful"],
        evidence_features=[[0.0] * 13 for _ in range(3)],
        evidence_costs=[1.0, 1.0, 1.0],
        false_edit_risks=[0.0, 0.0, 1.0],
        oracle_utilities=[-0.2, 0.3, -0.5],
        metadata={
            "initial_evidence_id": "initial",
            "utility_mode": "executable",
            "utility_profile": "balanced",
            "evidence_predictions": {
                key: {
                    "edit_probabilities": [1.0, 0.0, 0.0, 0.0],
                    "confidence": 1.0,
                    "geometry_delta": [0.0] * 8,
                }
                for key in ("initial", "helpful", "harmful")
            },
        },
    )
    states = _expand_budget_states(
        sample, budgets=(2.0,), cost_weight=99.0, max_steps=1
    )
    assert len(states) == 1
    assert states[0].oracle_utilities == pytest.approx([0.45, -0.35])
    assert states[0].state_features[7] == pytest.approx(-0.2)
    assert states[0].target_index() == 0


def test_episode_asset_roots_are_remapped() -> None:
    item = EvidenceItem(
        evidence_id="jan",
        timestamp="2019_01",
        region=(0, 0, 32, 32),
        scale=1,
        image_path="/old/sn7/image.tif",
        udm_path="/old/sn7/udm.tif",
        prior_image_path="/old/sn7/prior.tif",
        prior_udm_path="/old/sn7/prior-udm.tif",
        prior_timestamp="2019_01",
        clear_fraction=1.0,
        cost=1.0,
    )
    episode = EpisodeRecord(
        episode_id="episode",
        aoi_id="aoi",
        anchor_timestamp="2019_01",
        split="train",
        source_dataset="test",
        map_before="old.geojson",
        target_map="new.geojson",
        hypothesis=CandidateHypothesis(
            op=EditOperation.KEEP,
            object_id="building",
            source="test",
            confidence=0.5,
        ),
        evidence_catalog=[item],
        gt_edit=EditRecord(op=EditOperation.KEEP, object_id="building"),
        is_synthetic=False,
        derivation_version="test",
    )
    remapped = remap_episode_assets(
        [episode], ((Path("/old"), Path("/new")),)
    )[0]
    assert remapped.evidence_catalog[0].image_path == str(Path("/new/sn7/image.tif"))
    assert remapped.evidence_catalog[0].udm_path == str(Path("/new/sn7/udm.tif"))
    assert remapped.evidence_catalog[0].prior_image_path == str(Path("/new/sn7/prior.tif"))
    assert remapped.evidence_catalog[0].prior_udm_path == str(Path("/new/sn7/prior-udm.tif"))


def test_hdf5_input_cache_matches_raw_counterfactual_inputs(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    image_path = tmp_path / "image.tif"
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 32, 1, 1),
    ) as dataset:
        image = np.stack(
            [np.full((32, 32), value, dtype=np.uint8) for value in (50, 100, 150)]
        )
        dataset.write(image)

    prior = _box(8, 16)
    target = _box(10, 18)
    episode = EpisodeRecord(
        episode_id="cache_episode",
        aoi_id="cache_aoi",
        anchor_timestamp="2019_02",
        split="train",
        source_dataset="test",
        map_before="old.geojson",
        target_map="new.geojson",
        prior_geometry=prior,
        target_geometry=target,
        hypothesis=CandidateHypothesis(
            op=EditOperation.RESHAPE,
            object_id="building",
            geometry=target,
            source="test",
            confidence=0.5,
        ),
        evidence_catalog=[
            EvidenceItem(
                evidence_id="jan",
                timestamp="2019_01",
                region=(0, 0, 32, 32),
                scale=1,
                image_path=str(image_path),
                clear_fraction=1.0,
                cost=1.0,
            ),
            EvidenceItem(
                evidence_id="feb",
                timestamp="2019_02",
                region=(0, 0, 32, 32),
                scale=2,
                image_path=str(image_path),
                clear_fraction=0.8,
                cost=1.5,
            ),
        ],
        gt_edit=EditRecord(
            op=EditOperation.RESHAPE,
            object_id="building",
            geometry=target,
        ),
        is_synthetic=False,
        derivation_version="test",
    )
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(episode.model_dump_json() + "\n", encoding="utf-8")
    model = PriorConditionedUNet(UpdaterConfig(base_channels=4, dropout=0.0))
    checkpoint_path = tmp_path / "updater.pt"
    torch.save(
        {"state_dict": model.state_dict(), "model_config": model.config.as_dict()},
        checkpoint_path,
    )

    cache_path = tmp_path / "oracle_inputs.h5"
    cache_summary = build_selector_oracle_input_cache(
        episodes_path,
        cache_path,
        image_size=16,
        image_channels=3,
        splits=("train",),
    )
    raw_output = tmp_path / "raw.jsonl"
    parallel_output = tmp_path / "parallel.jsonl"
    cached_output = tmp_path / "cached.jsonl"
    build_selector_oracle_samples(
        checkpoint_path,
        episodes_path,
        raw_output,
        device="cpu",
        image_size=16,
        budgets=(2.0,),
        splits=("train",),
        candidate_workers=1,
    )
    build_selector_oracle_samples(
        checkpoint_path,
        episodes_path,
        parallel_output,
        device="cpu",
        image_size=16,
        budgets=(2.0,),
        splits=("train",),
        candidate_workers=2,
    )
    build_selector_oracle_samples(
        checkpoint_path,
        episodes_path,
        cached_output,
        device="cpu",
        image_size=16,
        budgets=(2.0,),
        splits=("train",),
        input_cache=cache_path,
    )

    assert cache_summary["candidates"] == 2
    raw_samples = load_selector_samples(raw_output)
    parallel_samples = load_selector_samples(parallel_output)
    cached_samples = load_selector_samples(cached_output)
    assert [sample.model_dump() for sample in parallel_samples] == [
        sample.model_dump() for sample in raw_samples
    ]
    assert [sample.model_dump() for sample in cached_samples] == [
        sample.model_dump() for sample in raw_samples
    ]


def test_temporal_hdf5_cache_matches_six_channel_oracle_inputs(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    old_path = tmp_path / "old.tif"
    current_path = tmp_path / "current.tif"
    for path, values in ((old_path, (30, 60, 90)), (current_path, (120, 150, 180))):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=32,
            height=32,
            count=3,
            dtype="uint8",
            crs="EPSG:3857",
            transform=from_origin(0, 32, 1, 1),
        ) as dataset:
            dataset.write(
                np.stack([np.full((32, 32), value, dtype=np.uint8) for value in values])
            )

    prior = _box(8, 16)
    target = _box(10, 18)
    episode = EpisodeRecord(
        episode_id="temporal_cache_episode",
        aoi_id="cache_aoi",
        anchor_timestamp="2019_02",
        split="train",
        source_dataset="test",
        map_before="old.geojson",
        target_map="new.geojson",
        prior_geometry=prior,
        target_geometry=target,
        hypothesis=CandidateHypothesis(
            op=EditOperation.RESHAPE,
            object_id="building",
            geometry=target,
            source="test",
            confidence=0.5,
        ),
        evidence_catalog=[
            EvidenceItem(
                evidence_id="current",
                timestamp="2019_02",
                region=(0, 0, 32, 32),
                scale=1,
                image_path=str(current_path),
                prior_image_path=str(old_path),
                prior_timestamp="2019_01",
                clear_fraction=1.0,
                cost=1.0,
            ),
            EvidenceItem(
                evidence_id="current-context",
                timestamp="2019_02",
                region=(0, 0, 32, 32),
                scale=2,
                image_path=str(current_path),
                prior_image_path=str(old_path),
                prior_timestamp="2019_01",
                clear_fraction=1.0,
                cost=1.5,
            )
        ],
        gt_edit=EditRecord(
            op=EditOperation.RESHAPE,
            object_id="building",
            geometry=target,
        ),
        is_synthetic=False,
        derivation_version="test",
    )
    episodes_path = tmp_path / "temporal_episodes.jsonl"
    episodes_path.write_text(episode.model_dump_json() + "\n", encoding="utf-8")
    model = PriorConditionedUNet(
        UpdaterConfig(
            image_channels=6,
            temporal_pair_input=True,
            base_channels=4,
            dropout=0.0,
        )
    )
    checkpoint_path = tmp_path / "temporal_updater.pt"
    torch.save(
        {"state_dict": model.state_dict(), "model_config": model.config.as_dict()},
        checkpoint_path,
    )
    cache_path = tmp_path / "temporal_oracle_inputs.h5"
    cache_summary = build_selector_oracle_input_cache(
        episodes_path,
        cache_path,
        image_size=16,
        image_channels=6,
        temporal_pair_input=True,
        splits=("train",),
    )
    raw_output = tmp_path / "temporal_raw.jsonl"
    cached_output = tmp_path / "temporal_cached.jsonl"
    for output, cache in ((raw_output, None), (cached_output, cache_path)):
        build_selector_oracle_samples(
            checkpoint_path,
            episodes_path,
            output,
            device="cpu",
            image_size=16,
            budgets=(2.0,),
            splits=("train",),
            input_cache=cache,
        )

    assert cache_summary["temporal_pair_input"] is True
    assert cache_summary["candidates"] == 2
    assert [sample.model_dump() for sample in load_selector_samples(cached_output)] == [
        sample.model_dump() for sample in load_selector_samples(raw_output)
    ]
