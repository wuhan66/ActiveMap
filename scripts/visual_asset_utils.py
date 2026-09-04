"""Small helpers for individual visual assets, without labels or contact sheets."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image


def filename(label: str) -> str:
    """Return a stable, human-readable filename for one semantic panel."""

    value = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return value or "unnamed_layer"


def save_panel_layers(
    case_dir: Path,
    panels: list[tuple[str, np.ndarray]],
    *,
    crop: tuple[slice, slice] | None = None,
) -> dict[str, dict[str, str]]:
    """Save each rendered panel as an unlabelled image, optionally with a crop.

    The returned relative paths are intended for the case manifest.  No contact
    sheet, label bar, or composition is generated here.
    """

    layer_dir = case_dir / "layers"
    layer_dir.mkdir()
    crop_dir = case_dir / "crops" if crop is not None else None
    if crop_dir is not None:
        crop_dir.mkdir()
    written: dict[str, dict[str, str]] = {}
    for label, panel in panels:
        stem = filename(label)
        image = Image.fromarray(np.asarray(panel, dtype=np.uint8), mode="RGB")
        layer_path = layer_dir / f"{stem}.png"
        image.save(layer_path, optimize=True)
        paths = {"layer": str(layer_path.relative_to(case_dir))}
        if crop_dir is not None and crop is not None:
            crop_path = crop_dir / f"{stem}.png"
            Image.fromarray(np.asarray(panel[crop], dtype=np.uint8), mode="RGB").save(
                crop_path, optimize=True
            )
            paths["crop"] = str(crop_path.relative_to(case_dir))
        written[label] = paths
    return written
