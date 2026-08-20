from pathlib import Path

from activemap.data.sn7 import build_sn7_manifest, classify_asset, extract_timestamp


def _make_asset(root: Path, aoi: str, timestamp: str) -> None:
    image_dir = root / "train" / aoi / "images_masked"
    label_dir = root / "train" / aoi / "labels_match"
    udm_dir = root / "train" / aoi / "UDM_masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    udm_dir.mkdir(parents=True, exist_ok=True)
    stem = f"global_monthly_{timestamp}_mosaic_{aoi}"
    (image_dir / f"{stem}.tif").touch()
    (label_dir / f"{stem}_Buildings.geojson").write_text("{}", encoding="utf-8")
    (udm_dir / f"{stem}_UDM.tif").touch()


def test_asset_classification_and_timestamp() -> None:
    assert classify_asset(Path("UDM_masks/global_monthly_2019_01_UDM.tif")) == "udm"
    assert classify_asset(Path("labels_match/sample_Buildings.geojson")) == "label"
    assert extract_timestamp(Path("global_monthly_2019_01_mosaic_AOI.tif")) == "2019_01"


def test_build_manifest_from_official_style_layout(tmp_path: Path) -> None:
    for aoi_index in range(4):
        aoi = f"L15-000{aoi_index}E-000{aoi_index}N_1_2_3"
        _make_asset(tmp_path, aoi, "2019_01")
        _make_asset(tmp_path, aoi, "2019_02")

    frame = build_sn7_manifest(tmp_path, read_raster_metadata=False)
    assert len(frame) == 8
    assert frame["aoi_id"].nunique() == 4
    assert frame["has_label"].all()
    assert frame["has_udm"].all()
    assert not frame.duplicated(["aoi_id", "timestamp"]).any()


def test_manifest_prefers_masked_images_and_matched_world_labels(tmp_path: Path) -> None:
    aoi = "L15-0001E-0001N_1_2_3"
    _make_asset(tmp_path, aoi, "2019_01")
    aoi_root = tmp_path / "train" / aoi
    stem = f"global_monthly_2019_01_mosaic_{aoi}"
    raw_image_dir = aoi_root / "images"
    raw_label_dir = aoi_root / "labels"
    pixel_label_dir = aoi_root / "labels_match_pix"
    raw_image_dir.mkdir()
    raw_label_dir.mkdir()
    pixel_label_dir.mkdir()
    (raw_image_dir / f"{stem}.tif").touch()
    (raw_label_dir / f"{stem}_Buildings.geojson").write_text("{}", encoding="utf-8")
    (pixel_label_dir / f"{stem}_Buildings.geojson").write_text("{}", encoding="utf-8")

    frame = build_sn7_manifest(tmp_path, read_raster_metadata=False)
    assert len(frame) == 1
    assert Path(frame.iloc[0]["image_path"]).parent.name == "images_masked"
    assert Path(frame.iloc[0]["label_path"]).parent.name == "labels_match"
    assert frame.iloc[0]["image_variant"] == "images_masked"
    assert frame.iloc[0]["label_variant"] == "labels_match"
