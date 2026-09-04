import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from activemap.geo_tools import GeoToolCall, GeoToolName, build_default_registry


def _write_raster(path: Path, data: np.ndarray, *, dtype: str = "float32") -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        dtype=dtype,
        crs="EPSG:3857",
        transform=from_origin(0, data.shape[1], 1, 1),
    ) as dataset:
        dataset.write(data.astype(dtype))


def _multispectral() -> np.ndarray:
    y, x = np.mgrid[1:17, 1:17]
    red = x.astype(np.float32)
    green = y.astype(np.float32)
    blue = (x + y).astype(np.float32)
    nir = 3.0 * red
    return np.stack([red, green, blue, nir])


def test_default_registry_runs_crop_quality_index_and_segmentation(tmp_path: Path) -> None:
    image_path = tmp_path / "multispectral.tif"
    _write_raster(image_path, _multispectral())
    registry = build_default_registry(tmp_path / "artifacts")
    assert set(registry.names()) == set(GeoToolName)

    crop = registry.execute(
        GeoToolCall(
            call_id="crop-1",
            tool=GeoToolName.RASTER_CROP,
            inputs={"image_path": str(image_path)},
            parameters={"pixel_window": [2, 3, 6, 5], "out_size": [8, 8], "bands": [1, 4]},
        )
    )
    assert crop.success
    assert crop.outputs["shape"] == [2, 8, 8]
    assert Path(crop.artifacts[0]).is_file()

    quality = registry.execute(
        GeoToolCall(
            call_id="quality-1",
            tool=GeoToolName.IMAGE_QUALITY,
            inputs={"image_path": str(image_path)},
            parameters={"bands": [1, 2, 3]},
        )
    )
    assert quality.success
    assert quality.outputs["valid_fraction"] == 1.0

    ndvi = registry.execute(
        GeoToolCall(
            call_id="ndvi-1",
            tool=GeoToolName.SPECTRAL_INDEX,
            inputs={"image_path": str(image_path)},
            parameters={"index": "NDVI", "band_map": {"red": 1, "nir": 4}},
        )
    )
    assert ndvi.success
    assert abs(ndvi.outputs["stats"]["mean"] - 0.5) < 1e-6

    segment = registry.execute(
        GeoToolCall(
            call_id="segment-1",
            tool=GeoToolName.RASTER_SEGMENT,
            inputs={"image_path": str(image_path)},
            parameters={"band": 1, "threshold": 8.0, "minimum_area": 4},
        )
    )
    assert segment.success
    assert segment.outputs["component_count"] == 1


def test_missing_spectral_bands_fail_explicitly(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.tif"
    _write_raster(image_path, _multispectral()[:3])
    result = build_default_registry(tmp_path / "artifacts").execute(
        GeoToolCall(
            call_id="invalid-ndvi",
            tool=GeoToolName.SPECTRAL_INDEX,
            inputs={"image_path": str(image_path)},
            parameters={"index": "NDVI", "band_map": {"red": 1}},
        )
    )
    assert not result.success
    assert "requires band_map" in str(result.error)


def test_uint8_raster_supports_quality_and_temporal_change(tmp_path: Path) -> None:
    before_path = tmp_path / "before-uint8.tif"
    after_path = tmp_path / "after-uint8.tif"
    before = np.clip(_multispectral()[:3] * 4, 0, 255).astype(np.uint8)
    after = before.copy()
    after[:, :8, :8] = 255 - after[:, :8, :8]
    _write_raster(before_path, before, dtype="uint8")
    _write_raster(after_path, after, dtype="uint8")
    registry = build_default_registry(tmp_path / "artifacts")

    quality = registry.execute(
        GeoToolCall(
            call_id="quality-uint8",
            tool=GeoToolName.IMAGE_QUALITY,
            inputs={"image_path": str(before_path)},
        )
    )
    change = registry.execute(
        GeoToolCall(
            call_id="change-uint8",
            tool=GeoToolName.TEMPORAL_CHANGE,
            inputs={"before_path": str(before_path), "after_path": str(after_path)},
            parameters={"threshold": 0.1},
        )
    )

    assert quality.success, quality.error
    assert change.success, change.error
    assert quality.outputs["valid_fraction"] == 1.0
    assert change.outputs["changed_fraction"] > 0.0


def test_cached_reads_do_not_share_mutable_valid_masks(tmp_path: Path) -> None:
    image_path = tmp_path / "cache-isolation.tif"
    _write_raster(image_path, _multispectral()[:3], dtype="float32")
    registry = build_default_registry(tmp_path / "artifacts")
    call = GeoToolCall(
        call_id="quality-cache-1",
        tool=GeoToolName.IMAGE_QUALITY,
        inputs={"image_path": str(image_path)},
        parameters={"out_size": [8, 8]},
    )

    first = registry.execute(call)
    second = registry.execute(call.model_copy(update={"call_id": "quality-cache-2"}))

    assert first.success
    assert second.success
    assert first.outputs == second.outputs


def test_temporal_change_terrain_and_vector_inspection(tmp_path: Path) -> None:
    before_path = tmp_path / "before.tif"
    after_path = tmp_path / "after.tif"
    before = _multispectral()[:3]
    after = before.copy()
    after[0, :8, :8] = np.flip(after[0, :8, :8], axis=1)
    _write_raster(before_path, before)
    _write_raster(after_path, after)
    registry = build_default_registry(tmp_path / "artifacts")
    change = registry.execute(
        GeoToolCall(
            call_id="change-1",
            tool=GeoToolName.TEMPORAL_CHANGE,
            inputs={"before_path": str(before_path), "after_path": str(after_path)},
            parameters={"bands": [1, 2, 3], "threshold": 0.1},
        )
    )
    assert change.success
    assert change.outputs["changed_fraction"] > 0.0

    y, x = np.mgrid[0:16, 0:16]
    dem_path = tmp_path / "dem.tif"
    _write_raster(dem_path, (x + 2.0 * y)[None, ...].astype(np.float32))
    terrain = registry.execute(
        GeoToolCall(
            call_id="terrain-1",
            tool=GeoToolName.TERRAIN_ANALYSIS,
            inputs={"dem_path": str(dem_path)},
        )
    )
    assert terrain.success
    assert terrain.outputs["slope_degrees"]["mean"] > 0.0

    vector_path = tmp_path / "map.geojson"
    vector_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": 1},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    vector = registry.execute(
        GeoToolCall(
            call_id="vector-1",
            tool=GeoToolName.VECTOR_INSPECT,
            inputs={"vector_path": str(vector_path)},
        )
    )
    assert vector.success
    assert vector.outputs["feature_count"] == 1
    assert vector.outputs["invalid_count"] == 0
