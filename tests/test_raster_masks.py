import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box

from activemap.data.raster_masks import rasterize_geometry_mask


def test_tiny_geometry_gets_one_pixel_fallback() -> None:
    transform = from_origin(0, 10, 1, 1)
    geometry = box(0.05, 9.75, 0.25, 9.95)
    mask = rasterize_geometry_mask(geometry, transform, 10)
    assert mask.dtype == np.float32
    assert int(mask.sum()) == 1
    assert mask[0, 0] == 1.0


def test_empty_geometry_stays_empty() -> None:
    mask = rasterize_geometry_mask(None, from_origin(0, 10, 1, 1), 10)
    assert not np.any(mask)


def test_visible_sliver_uses_visible_representative_point() -> None:
    transform = from_origin(0, 10, 1, 1)
    geometry = box(-10, 9.75, 0.25, 9.95)
    mask = rasterize_geometry_mask(geometry, transform, 10)
    assert int(mask.sum()) == 1
    assert mask[0, 0] == 1.0
