import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from activemap.data.inria import build_inria_segmentation, build_inria_updater
from activemap.data.merge_updaters import merge_updater_manifests
from activemap.data.muno21 import (
    _scenario_crop_windows,
    build_muno21_evidence_episodes,
    build_muno21_updater,
    muno_tags_to_edit,
    read_muno_graph,
)
from activemap.models import EditOperation, EpisodeRecord
from activemap.updater_records import UpdaterSample, load_updater_samples

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402


def test_read_muno_graph_deduplicates_bidirectional_lines(tmp_path: Path) -> None:
    path = tmp_path / "sample.graph"
    path.write_text("0 0\n10 0\n10 10\n\n0 1\n1 0\n1 2\n2 1\n", encoding="utf-8")
    graph = read_muno_graph(path)
    assert graph.vertices.shape == (3, 2)
    assert len(graph.edges) == 4
    assert len(graph.lines()) == 2


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (["nochange"], EditOperation.KEEP),
        (["constructed"], EditOperation.ADD),
        (["was_missing"], EditOperation.ADD),
        (["bulldozed"], EditOperation.DELETE),
        (["was_incorrect"], EditOperation.DELETE),
        (["constructed", "bulldozed"], EditOperation.RESHAPE),
    ],
)
def test_muno_tags_map_to_typed_edits(tags: list[str], expected: EditOperation) -> None:
    assert muno_tags_to_edit(tags) == expected


def test_muno_large_scenario_crops_cover_disconnected_changes() -> None:
    annotation = {
        "Cluster": {
            "Window": [0, 0, 2200, 2200],
            "Changes": [
                {
                    "Segments": [
                        {
                            "Start": {"X": 100, "Y": 100},
                            "End": {"X": 200, "Y": 200},
                        },
                        {
                            "Start": {"X": 1900, "Y": 1900},
                            "End": {"X": 2100, "Y": 2100},
                        },
                    ]
                }
            ],
        }
    }
    crops = _scenario_crop_windows(annotation, padding=128, max_source_crop_size=1024)

    assert len(crops) == 2
    for x, y in ((150, 150), (2000, 2000)):
        assert any(left <= x <= right and top <= y <= bottom for left, top, right, bottom in crops)


def test_build_muno21_add_scenario(tmp_path: Path) -> None:
    root = tmp_path / "mapupdate"
    image_dir = root / "naip" / "jpg"
    graph_dir = root / "graphs" / "graphs"
    image_dir.mkdir(parents=True)
    graph_dir.mkdir(parents=True)
    Image.new("RGB", (96, 96), (120, 140, 100)).save(image_dir / "city_0_0_2019.jpg")
    annotation = {
        "Cluster": {
            "Region": "city",
            "Tile": [0, 0],
            "Window": [16, 16, 64, 64],
            "Changes": [
                {
                    "Deleted": False,
                    "Segments": [{"Start": {"X": 20, "Y": 20}, "End": {"X": 58, "Y": 52}}],
                }
            ],
        },
        "Tags": ["constructed"],
        "Years": [2016, 2017],
    }
    (root / "annotations.json").write_text(json.dumps([annotation]), encoding="utf-8")
    (root / "train.json").write_text(json.dumps(["city"]), encoding="utf-8")
    (root / "test.json").write_text(json.dumps([]), encoding="utf-8")
    (graph_dir / "city_0_0_2013-07-01.graph").write_text(
        "8 8\n12 8\n\n0 1\n1 0\n", encoding="utf-8"
    )
    (graph_dir / "city_0_0_2020-07-01.graph").write_text(
        "8 8\n12 8\n20 20\n58 52\n\n0 1\n1 0\n2 3\n3 2\n", encoding="utf-8"
    )
    output = tmp_path / "muno_output"
    summary = build_muno21_updater(root, output, image_size=64, padding=8)
    samples = load_updater_samples(output / "updater_samples.jsonl")
    assert summary["samples"] == 1
    assert samples[0].edit_type == EditOperation.ADD
    assert samples[0].geometry_family == "polyline"
    assert samples[0].supervision_type == "full_scene_temporal"
    assert np.load(samples[0].image_path).shape == (3, 64, 64)
    assert np.load(samples[0].prior_mask_path).sum() > 0
    assert np.load(samples[0].target_mask_path).sum() > np.load(samples[0].prior_mask_path).sum()

    pillow_limit = Image.MAX_IMAGE_PIXELS
    with pytest.raises(ValueError, match="trusted limit"):
        build_muno21_updater(
            root,
            tmp_path / "muno_too_large",
            image_size=64,
            padding=8,
            max_source_pixels=100,
        )
    assert Image.MAX_IMAGE_PIXELS == pillow_limit


def test_build_muno21_nochange_with_null_changes(tmp_path: Path) -> None:
    root = tmp_path / "mapupdate"
    image_dir = root / "naip" / "jpg"
    graph_dir = root / "graphs" / "graphs"
    image_dir.mkdir(parents=True)
    graph_dir.mkdir(parents=True)
    Image.new("RGB", (96, 96), (120, 140, 100)).save(image_dir / "city_0_0_2019.jpg")
    annotation = {
        "Cluster": {
            "Region": "city",
            "Tile": [0, 0],
            "Window": [16, 16, 64, 64],
            "Changes": None,
        },
        "Tags": ["nochange"],
        "Years": [2016, 2017],
    }
    (root / "annotations.json").write_text(json.dumps([annotation]), encoding="utf-8")
    (root / "train.json").write_text(json.dumps(["city"]), encoding="utf-8")
    (root / "test.json").write_text(json.dumps([]), encoding="utf-8")
    (graph_dir / "city_0_0_2013-07-01.graph").write_text(
        "20 20\n58 52\n\n0 1\n1 0\n", encoding="utf-8"
    )
    (graph_dir / "city_0_0_2020-07-01.graph").write_text(
        "20 20\n58 52\n70 70\n80 80\n\n0 1\n1 0\n2 3\n3 2\n", encoding="utf-8"
    )

    output = tmp_path / "muno_output"
    summary = build_muno21_updater(root, output, image_size=64, padding=8)
    samples = load_updater_samples(output / "updater_samples.jsonl")

    assert summary["operations"] == {"KEEP": 1}
    assert len(samples) == 1
    assert samples[0].edit_type == EditOperation.KEEP
    prior = np.load(samples[0].prior_mask_path)
    target = np.load(samples[0].target_mask_path)
    assert prior.sum() > 0
    assert np.array_equal(prior, target)
    assert samples[0].source_metadata["target_graph"].endswith("city_0_0_2013-07-01.graph")


def test_build_muno21_real_temporal_evidence_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_dir = tmp_path / "naip"
    image_dir.mkdir()
    for year in (2013, 2015, 2017, 2019):
        Image.new("RGB", (64, 64), (year % 255, 100, 120)).save(
            image_dir / f"city_0_0_{year}.jpg"
        )
    sample = UpdaterSample(
        sample_id="muno21-keep",
        aoi_id="city",
        split="train",
        image_path="unused.npy",
        prior_mask_path="unused.npy",
        target_mask_path="unused.npy",
        edit_type=EditOperation.KEEP,
        geometry_delta=[0.0] * 8,
        object_id="road-1",
        dataset_name="muno21",
        geometry_family="polyline",
        supervision_type="full_scene_temporal",
        source_metadata={
            "region": "city",
            "tile": [0, 0],
            "crop_bounds": [8, 8, 56, 56],
            "source_image": str(image_dir / "city_0_0_2019.jpg"),
            "prior_graph": "prior.graph",
            "target_graph": "target.graph",
        },
    )
    manifest = tmp_path / "updater_samples.jsonl"
    manifest.write_text(sample.model_dump_json() + "\n", encoding="utf-8")
    output = tmp_path / "episodes.jsonl"

    summary = build_muno21_evidence_episodes(manifest, output)
    episode = EpisodeRecord.model_validate_json(output.read_text(encoding="utf-8"))

    assert summary["episodes"] == 1
    assert summary["evidence_count_histogram"] == {4: 1}
    assert episode.anchor_timestamp == "2019-01"
    assert [item.timestamp for item in episode.evidence_catalog] == [
        "2013-01",
        "2015-01",
        "2017-01",
        "2019-01",
    ]
    assert episode.evidence_catalog[-1].cost == pytest.approx(0.25)
    assert episode.gt_edit.op == EditOperation.KEEP
    with pytest.raises(PermissionError, match="requires --frozen-test"):
        build_muno21_evidence_episodes(manifest, tmp_path / "test.jsonl", splits=("test",))

    token = "one-shot-test-token"
    ledger = tmp_path / "frozen_test_ledger.json"
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
    manifest.write_text(
        sample.model_copy(update={"split": "test"}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    test_output = tmp_path / "test.jsonl"
    test_summary = build_muno21_evidence_episodes(
        manifest,
        test_output,
        splits=("test",),
        frozen_test=True,
    )
    assert test_summary["splits"] == {"test": 1}
    assert test_summary["test_assets_read"] is True
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_muno21_evidence_episodes(
            manifest,
            test_output,
            splits=("test",),
            frozen_test=True,
        )


def test_build_inria_controlled_prior_samples(tmp_path: Path) -> None:
    root = tmp_path / "inria" / "train"
    image_dir = root / "images"
    gt_dir = root / "gt"
    image_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    transform = from_origin(0.0, 19.2, 0.3, 0.3)
    image = np.full((3, 64, 64), 160, dtype=np.uint8)
    image[:, :8, :] = 0
    label = np.zeros((64, 64), dtype=np.uint8)
    label[20:40, 20:40] = 255
    profile = {
        "driver": "GTiff",
        "width": 64,
        "height": 64,
        "count": 3,
        "dtype": "uint8",
        "transform": transform,
        "crs": "EPSG:32633",
    }
    with rasterio.open(image_dir / "austin1.tif", "w", **profile) as dataset:
        dataset.write(image)
    with rasterio.open(gt_dir / "austin1.tif", "w", **{**profile, "count": 1}) as dataset:
        dataset.write(label, 1)
    output = tmp_path / "inria_output"
    summary = build_inria_updater(
        root.parent,
        output,
        image_size=64,
        context_pixels=8,
        max_objects_per_tile=1,
        min_area_pixels=4,
    )
    samples = load_updater_samples(output / "updater_samples.jsonl")
    assert summary["samples"] == 3
    assert {sample.edit_type for sample in samples} == {
        EditOperation.KEEP,
        EditOperation.ADD,
        EditOperation.RESHAPE,
    }
    assert all(sample.geometry_family == "polygon" for sample in samples)
    assert {sample.supervision_type for sample in samples} == {
        "single_timestamp",
        "synthetic_prior",
    }

    segmentation_output = tmp_path / "inria_segmentation"
    segmentation_summary = build_inria_segmentation(
        root.parent,
        segmentation_output,
        image_size=64,
        window_size=64,
        stride=64,
    )
    segmentation_samples = load_updater_samples(segmentation_output / "updater_samples.jsonl")
    assert segmentation_summary["samples"] == 1
    assert segmentation_summary["foreground"] == {"positive": 1}
    assert len(segmentation_samples) == 1
    scene_sample = segmentation_samples[0]
    assert scene_sample.edit_type == EditOperation.KEEP
    assert np.load(scene_sample.target_mask_path).sum() == 400
    assert np.array_equal(
        np.load(scene_sample.prior_mask_path), np.load(scene_sample.target_mask_path)
    )
    scene_valid = np.load(scene_sample.valid_mask_path)
    assert scene_valid[:8, :].sum() == 0
    assert scene_valid[8:, :].all()


def test_build_inria_skips_low_valid_boundary_crops(tmp_path: Path) -> None:
    root = tmp_path / "inria" / "train"
    image_dir = root / "images"
    gt_dir = root / "gt"
    image_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    transform = from_origin(0.0, 19.2, 0.3, 0.3)
    image = np.full((3, 64, 64), 160, dtype=np.uint8)
    label = np.zeros((64, 64), dtype=np.uint8)
    label[0:4, 0:4] = 255
    profile = {
        "driver": "GTiff",
        "width": 64,
        "height": 64,
        "count": 3,
        "dtype": "uint8",
        "transform": transform,
        "crs": "EPSG:32633",
    }
    with rasterio.open(image_dir / "austin1.tif", "w", **profile) as dataset:
        dataset.write(image)
    with rasterio.open(gt_dir / "austin1.tif", "w", **{**profile, "count": 1}) as dataset:
        dataset.write(label, 1)

    output = tmp_path / "inria_output"
    summary = build_inria_updater(
        root.parent,
        output,
        image_size=64,
        context_pixels=48,
        max_objects_per_tile=1,
        min_area_pixels=4,
        min_valid_fraction=0.75,
    )

    assert summary["samples"] == 0
    assert summary["skipped_low_valid"] == 1
    assert (output / "updater_samples.jsonl").read_text(encoding="utf-8") == ""


def test_merge_updater_manifests_preserves_provenance(tmp_path: Path) -> None:
    source_dirs = [tmp_path / "a", tmp_path / "b"]
    manifests = []
    for index, source_dir in enumerate(source_dirs):
        source_dir.mkdir()
        for name in ("image", "prior", "target"):
            np.save(source_dir / f"{name}.npy", np.zeros((1, 8, 8), dtype=np.float32))
        sample = UpdaterSample(
            sample_id=f"sample-{index}",
            aoi_id=f"aoi-{index}",
            split="train",
            image_path="image.npy",
            prior_mask_path="prior.npy",
            target_mask_path="target.npy",
            edit_type=EditOperation.KEEP,
            geometry_delta=[0.0] * 8,
            dataset_name=f"dataset-{index}",
        )
        manifest = source_dir / "samples.jsonl"
        manifest.write_text(sample.model_dump_json() + "\n", encoding="utf-8")
        manifests.append(manifest)

    output = tmp_path / "merged" / "samples.jsonl"
    summary = merge_updater_manifests(manifests, output)
    merged = load_updater_samples(output)

    assert summary["samples"] == 2
    assert summary["datasets"] == {"dataset-0": 1, "dataset-1": 1}
    assert all(Path(sample.image_path).is_absolute() for sample in merged)
