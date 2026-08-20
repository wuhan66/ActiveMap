"""Deterministic synthetic-prior generation for balanced edit supervision."""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.affinity import scale, translate
from shapely.geometry.base import BaseGeometry

from activemap.models import EditOperation


@dataclass(frozen=True)
class SyntheticCase:
    prior_map: gpd.GeoDataFrame
    operation: EditOperation
    object_id: str
    target_geometry: BaseGeometry | None
    seed: int


def distort_geometry(
    geometry: BaseGeometry,
    *,
    shift_meters: float,
    area_scale: float,
    rng: np.random.Generator,
) -> BaseGeometry:
    angle = rng.uniform(0.0, 2.0 * np.pi)
    shifted = translate(
        geometry,
        xoff=float(np.cos(angle) * shift_meters),
        yoff=float(np.sin(angle) * shift_meters),
    )
    linear_scale = float(np.sqrt(area_scale))
    return scale(shifted, xfact=linear_scale, yfact=linear_scale, origin="centroid")


def generate_synthetic_case(
    target_map: gpd.GeoDataFrame,
    *,
    operation: EditOperation,
    object_id: str,
    id_column: str = "object_id",
    seed: int = 0,
    shift_meters: float = 5.0,
    area_scale: float = 1.2,
) -> SyntheticCase:
    if id_column not in target_map.columns:
        raise ValueError(f"target map is missing ID column: {id_column}")
    matches = target_map.index[target_map[id_column].astype(str) == str(object_id)].tolist()
    if len(matches) != 1:
        raise ValueError(f"expected exactly one object with ID {object_id}, found {len(matches)}")

    rng = np.random.default_rng(seed)
    row_index = matches[0]
    target_geometry = target_map.loc[row_index].geometry
    prior = target_map.copy(deep=True)

    if operation == EditOperation.ADD:
        prior = prior.drop(index=row_index).reset_index(drop=True)
    elif operation == EditOperation.DELETE:
        decoy = prior.loc[[row_index]].copy(deep=True)
        decoy_id = f"synthetic-delete-{object_id}-{seed}"
        decoy[id_column] = decoy_id
        decoy.geometry = decoy.geometry.apply(
            lambda geometry: distort_geometry(
                geometry,
                shift_meters=max(shift_meters * 3.0, 15.0),
                area_scale=area_scale,
                rng=rng,
            )
        )
        prior = gpd.GeoDataFrame(
            pd.concat([prior, decoy], ignore_index=True),
            geometry="geometry",
            crs=prior.crs,
        )
        object_id = decoy_id
        target_geometry = None
    elif operation == EditOperation.RESHAPE:
        prior.at[row_index, "geometry"] = distort_geometry(
            target_geometry,
            shift_meters=shift_meters,
            area_scale=area_scale,
            rng=rng,
        )
    elif operation != EditOperation.KEEP:
        raise ValueError(f"unsupported operation: {operation}")

    return SyntheticCase(
        prior_map=prior,
        operation=operation,
        object_id=str(object_id),
        target_geometry=target_geometry,
        seed=seed,
    )
