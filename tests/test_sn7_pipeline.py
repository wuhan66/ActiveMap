import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from activemap.data.episode_audit import audit_episode_dataset
from activemap.data.episode_builder import build_sn7_episodes
from activemap.data.qc import render_updater_qc
from activemap.data.sn7_pairs import iter_snapshot_pairs
from activemap.data.updater_audit import audit_updater_dataset
from activemap.data.sn7_scan import scan_sn7_edits
from activemap.data.updater_crops import build_updater_crops
from activemap.models import EditOperation
from activemap.updater_records import load_updater_samples


def _write_raster(path: Path, bands: int, value: int) -> None:
    array = np.full((bands, 64, 64), value, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=64,
        height=64,
        count=bands,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 64, 1, 1),
    ) as dataset:
        dataset.write(array)


def _write_labels(path: Path, rows: list[tuple[str, object]]) -> None:
    frame = gpd.GeoDataFrame(
        {"object_id": [row[0] for row in rows]},
        geometry=[row[1] for row in rows],
        crs="EPSG:3857",
    )
    frame.to_file(path, driver="GeoJSON")


def _manifest(tmp_path: Path) -> pd.DataFrame:
    january_image = tmp_path / "january.tif"
    february_image = tmp_path / "february.tif"
    udm = tmp_path / "february-udm.tif"
    old_labels = tmp_path / "old.geojson"
    new_labels = tmp_path / "new.geojson"
    _write_raster(january_image, 3, 100)
    _write_raster(february_image, 3, 120)
    _write_raster(udm, 1, 0)
    _write_labels(
        old_labels,
        [
            ("keep", box(5, 40, 15, 50)),
            ("delete", box(20, 40, 30, 50)),
            ("reshape", box(35, 40, 45, 50)),
        ],
    )
    _write_labels(
        new_labels,
        [
            ("keep", box(5, 40, 15, 50)),
            ("add", box(20, 20, 30, 30)),
            ("reshape", box(38, 40, 48, 50)),
        ],
    )
    transform = list(from_origin(0, 64, 1, 1))
    return pd.DataFrame(
        {
            "aoi_id": ["aoi-1", "aoi-1"],
            "timestamp": ["2019_01", "2019_02"],
            "image_path": [str(january_image), str(february_image)],
            "label_path": [str(old_labels), str(new_labels)],
            "udm_path": [None, str(udm)],
            "split": ["train", "train"],
            "width": [64, 64],
            "height": [64, 64],
            "transform": [transform, transform],
            "clear_fraction": [1.0, 1.0],
        }
    )


def test_real_snapshot_pipeline_builds_all_edit_types(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output_dir = tmp_path / "updater"
    summary = build_updater_crops(
        manifest,
        output_dir,
        image_size=32,
        context_pixels=4,
        keep_iou_min=0.8,
        fallback_match_iou_min=0.2,
    )
    assert summary["samples"] == 4
    samples = load_updater_samples(output_dir / "updater_samples.jsonl")
    assert {sample.edit_type for sample in samples} == set(EditOperation)
    assert all(Path(sample.image_path).is_file() for sample in samples)
    assert all(len(sample.geometry_delta) == 8 for sample in samples)
    qc = render_updater_qc(
        output_dir / "updater_samples.jsonl", tmp_path / "qc", count=2, seed=3
    )
    assert qc["rendered"] == 2
    assert len(list((tmp_path / "qc").glob("*.png"))) == 2

    exact_qc = render_updater_qc(
        output_dir / "updater_samples.jsonl",
        tmp_path / "exact-qc",
        sample_ids={samples[0].sample_id},
    )
    assert exact_qc["rendered"] == 1
    assert exact_qc["requested_sample_ids"] == [samples[0].sample_id]

    episode_path = tmp_path / "episodes.jsonl"
    episode_summary = build_sn7_episodes(
        manifest,
        episode_path,
        context_units=(0.0,),
        scales=(1,),
    )
    assert episode_summary["episodes"] == 4
    payloads = [json.loads(line) for line in episode_path.read_text().splitlines()]
    assert {payload["gt_edit"]["op"] for payload in payloads} == {
        operation.value for operation in EditOperation
    }
    episode_audit = audit_episode_dataset(
        episode_path,
        expected_derivation_version="sn7-adjacent-v3-distance-gated",
    )
    assert episode_audit["passed"]
    assert episode_audit["episode_count"] == 4
    assert all(
        item["evidence_catalog"][0]["prior_image_path"] == str(tmp_path / "january.tif")
        for item in payloads
    )

    temporal_dir = tmp_path / "updater-temporal"
    temporal_summary = build_updater_crops(
        manifest,
        temporal_dir,
        image_size=32,
        context_pixels=4,
        keep_iou_min=0.8,
        fallback_match_iou_min=0.2,
        include_prior_image=True,
    )
    temporal_samples = load_updater_samples(temporal_dir / "updater_samples.jsonl")
    assert temporal_summary["temporal_pair_input"] is True
    assert all(
        sample.prior_image_path is not None and Path(sample.prior_image_path).is_file()
        for sample in temporal_samples
    )
    temporal_audit = audit_updater_dataset(temporal_dir / "updater_samples.jsonl")
    assert temporal_audit["passed"]
    assert temporal_audit["temporal_pair_input"] is True
    assert temporal_audit["temporal_pair_missing_count"] == 0
    assert temporal_audit["temporal_pair_shape_mismatch_count"] == 0


def test_snapshot_pair_skips_missing_months(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest.loc[1, "timestamp"] = "2019_03"
    assert not list(iter_snapshot_pairs(manifest, max_month_gap=1))


def test_scan_sn7_edits_writes_pair_counts(tmp_path: Path) -> None:
    output = tmp_path / "pair_counts.parquet"
    summary = scan_sn7_edits(_manifest(tmp_path), output)
    assert summary["events"] == 4
    assert summary["snapshot_pairs"] == 1
    row = pd.read_parquet(output).iloc[0]
    assert (row["keep"], row["add"], row["delete"], row["reshape"]) == (1, 1, 1, 1)
