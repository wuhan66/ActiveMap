from pathlib import Path

import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from activemap.data.raster_masks import rasterize_geometry_mask
from activemap.data.updater_mask_refresh import refresh_updater_masks
from activemap.models import EditOperation, GeoJSONGeometry
from activemap.updater_records import UpdaterSample


def test_refresh_updater_masks_supports_dry_run_and_write(tmp_path: Path) -> None:
    transform = from_origin(0, 10, 1, 1)
    prior_geometry = box(-10, 9.75, 0.25, 9.95)
    target_geometry = box(0.05, 9.75, 0.25, 9.95)
    image_path = tmp_path / "image.npy"
    prior_path = tmp_path / "prior.npy"
    target_path = tmp_path / "target.npy"
    np.save(image_path, np.zeros((3, 10, 10), dtype=np.float32))
    np.save(prior_path, np.zeros((10, 10), dtype=np.float32))
    np.save(
        target_path,
        rasterize_geometry_mask(target_geometry, transform, 10),
    )
    sample = UpdaterSample(
        sample_id="reshape-sliver",
        aoi_id="aoi-1",
        split="train",
        image_path=image_path.name,
        prior_mask_path=prior_path.name,
        target_mask_path=target_path.name,
        edit_type=EditOperation.RESHAPE,
        geometry_delta=[0.0] * 8,
        object_id="object-1",
        crop_transform=list(transform)[:6],
        prior_geometry=GeoJSONGeometry.model_validate(mapping(prior_geometry)),
        target_geometry=GeoJSONGeometry.model_validate(mapping(target_geometry)),
    )
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(sample.model_dump_json() + "\n", encoding="utf-8")

    dry_run = refresh_updater_masks(samples_path, tmp_path / "dry-run.json")
    assert dry_run["changed_samples"] == 1
    assert not np.any(np.load(prior_path))

    written = refresh_updater_masks(samples_path, tmp_path / "written.json", write=True)
    assert written["changed_prior_masks"] == 1
    assert int(np.load(prior_path).sum()) == 1
