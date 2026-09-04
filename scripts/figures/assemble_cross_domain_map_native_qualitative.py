#!/usr/bin/env python3
"""Render a provenance-bound cross-domain map-native qualitative plate.

The figure is deliberately an interface audit, not a visual re-ranking of
experiments.  It uses three fixed validation cases: a frozen SN7 polygon
proposal, a paired MUNO21 road writeback, and one real SpaceNet8 multi-POST
selection/defer trace.  Every row is cropped to one common coordinate window
so that editable geometry, rather than low-resolution binary masks or text
cards, remains the visual subject.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "docs" / "figures" / "cross_domain_map_native_qualitative_20260819_v2"
SN7_ROOT = ROOT / "docs" / "figures" / "sn7_map_native_qualitative_20260809_full"
MUNO_ROOT = ROOT / "docs" / "figures" / "muno21_map_native_qualitative_20260812"
SPACENET_ROOT = ROOT / "docs" / "figures" / "spacenet8_map_native_qualitative_20260812_v3"

INK = "#1C232A"
MUTED = "#59636E"
RULE = "#D8DDE1"
PRIOR_RGB = np.array([235, 142, 52], dtype=np.uint8)
PROPOSAL_RGB = np.array([52, 116, 201], dtype=np.uint8)
REFERENCE_RGB = np.array([53, 160, 97], dtype=np.uint8)
FALSE_RGB = np.array([217, 86, 72], dtype=np.uint8)
MISSED_RGB = np.array([112, 184, 222], dtype=np.uint8)
MAP_BACKGROUND = np.array([250, 250, 248], dtype=np.uint8)
SN7_PALETTE = np.array(
    [
        [244, 162, 0],
        [0, 114, 178],
        [0, 158, 115],
        [213, 94, 0],
        [86, 180, 233],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class SourceCase:
    dataset: str
    case_id: str
    split: str
    test_assets_read: bool


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def ensure_validation(manifest_path: Path, *, dataset: str) -> SourceCase:
    manifest = read_json(manifest_path)
    source_protocol = str(manifest.get("source_protocol", "")).lower()
    is_validation = manifest.get("split") == "val" or "validation-only" in source_protocol
    if not is_validation or manifest.get("test_assets_read") is not False:
        raise ValueError(f"{dataset} qualitative source must be validation-only: {manifest_path}")
    case_id = str(manifest.get("case_id") or manifest.get("source_example_id") or manifest_path.parent.name)
    return SourceCase(dataset=dataset, case_id=case_id, split="val", test_assets_read=False)


def geometry_mask(image: np.ndarray, palette: np.ndarray, *, tolerance: int = 8) -> np.ndarray:
    image_i = image.astype(np.int16)[..., None, :]
    palette_i = palette.astype(np.int16)[None, None, :, :]
    return np.any(np.max(np.abs(image_i - palette_i), axis=-1) <= tolerance, axis=-1)


def non_background_mask(image: np.ndarray, *, tolerance: int = 8) -> np.ndarray:
    return np.max(np.abs(image.astype(np.int16) - MAP_BACKGROUND.astype(np.int16)), axis=-1) > tolerance


def overlay_vector(context: np.ndarray, vector: np.ndarray) -> np.ndarray:
    if context.shape != vector.shape:
        raise ValueError(f"context/vector shape mismatch: {context.shape} vs {vector.shape}")
    rendered = context.copy()
    mask = non_background_mask(vector)
    rendered[mask] = vector[mask]
    return rendered


def square_crop(mask: np.ndarray, *, image_shape: tuple[int, int], minimum: int, padding: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("cannot crop an image without registered geometry")
    height, width = image_shape
    half = max(minimum // 2, int(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) / 2) + padding)
    center_x = int(round((xs.min() + xs.max()) / 2))
    center_y = int(round((ys.min() + ys.max()) / 2))
    side = min(2 * half, width, height)
    x0 = min(max(center_x - side // 2, 0), width - side)
    y0 = min(max(center_y - side // 2, 0), height - side)
    return x0, y0, x0 + side, y0 + side


def crop(image: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    return image[y0:y1, x0:x1]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.55,
        }
    )


def panel(axis: plt.Axes, image: np.ndarray, title: str, *, label: str | None = None, note: str | None = None) -> None:
    axis.imshow(image, interpolation="antialiased")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=7.2, color=INK, fontweight="bold", loc="left", pad=3.2)
    for spine in axis.spines.values():
        spine.set_color(RULE)
        spine.set_linewidth(0.55)
    if label:
        axis.text(
            0.025,
            0.045,
            label,
            transform=axis.transAxes,
            fontsize=7.2,
            fontweight="bold",
            color="white",
            va="bottom",
            ha="left",
            bbox={"facecolor": INK, "edgecolor": "none", "pad": 1.15},
        )
    if note:
        axis.text(
            0.975,
            0.045,
            note,
            transform=axis.transAxes,
            fontsize=6.4,
            fontweight="bold",
            color=INK,
            va="bottom",
            ha="right",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.84},
        )


def row_tag(figure: plt.Figure, y: float, panel_id: str, text: str) -> None:
    figure.text(0.02, y, panel_id, fontsize=8.6, color="white", fontweight="bold", va="center", ha="center", bbox={"facecolor": INK, "edgecolor": "none", "pad": 1.35})
    figure.text(0.044, y, text, fontsize=8.2, color=INK, fontweight="bold", va="center", ha="left")


def draw_legend(figure: plt.Figure) -> None:
    legend = [
        ("prior", "#EB8E34"),
        ("proposal", "#3474C9"),
        ("reference", "#35A061"),
        ("false", "#D95648"),
        ("missed", "#70B8DE"),
    ]
    x = 0.31
    figure.text(0.055, 0.035, "Boundary code", fontsize=6.7, color=MUTED, va="center")
    for label, color in legend:
        figure.patches.append(Rectangle((x, 0.027), 0.011, 0.011, transform=figure.transFigure, facecolor=color, edgecolor="none"))
        figure.text(x + 0.014, 0.035, label, fontsize=6.5, color=MUTED, va="center")
        x += 0.105


def required(paths: Iterable[Path]) -> list[Path]:
    paths = list(paths)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing figure inputs:\n" + "\n".join(missing))
    return paths


def render(output_dir: Path) -> dict:
    sn7_case = SN7_ROOT / "008_add_success"
    muno_case = MUNO_ROOT / "01_improved_writeback_task-9773234aa3b1b826"
    spacenet_case = SPACENET_ROOT / "01_rank100_changer_validation_case"
    inputs = required(
        [
            sn7_case / "manifest.json",
            sn7_case / "01_context.png",
            sn7_case / "02_prior_vector.png",
            sn7_case / "03_reference_vector.png",
            sn7_case / "04_changemamba_writeback.png",
            sn7_case / "05_changemamba_residual.png",
            muno_case / "manifest.json",
            muno_case / "current_rgb.png",
            muno_case / "prior_map.png",
            muno_case / "direct_writeback.png",
            muno_case / "activemap_writeback.png",
            muno_case / "reference_map.png",
            spacenet_case / "manifest.json",
            spacenet_case / "first_post_rgb.png",
            spacenet_case / "selected_post_rgb.png",
            spacenet_case / "selected_change_draft.png",
            spacenet_case / "safe_commit_writeback.png",
            spacenet_case / "reference_change.png",
        ]
    )
    sources = [
        ensure_validation(sn7_case / "manifest.json", dataset="SN7"),
        ensure_validation(muno_case / "manifest.json", dataset="MUNO21"),
        ensure_validation(spacenet_case / "manifest.json", dataset="SpaceNet8"),
    ]
    spacenet_manifest = read_json(spacenet_case / "manifest.json")
    if spacenet_manifest.get("tool_call_count") != 1:
        raise ValueError("SpaceNet8 row must retain exactly one recorded evidence call")

    sn7_context = load_rgb(sn7_case / "01_context.png")
    sn7_prior = load_rgb(sn7_case / "02_prior_vector.png")
    sn7_direct = load_rgb(sn7_case / "04_changemamba_writeback.png")
    sn7_reference = load_rgb(sn7_case / "03_reference_vector.png")
    sn7_residual = load_rgb(sn7_case / "05_changemamba_residual.png")
    sn7_bounds = square_crop(geometry_mask(sn7_reference, SN7_PALETTE), image_shape=sn7_context.shape[:2], minimum=360, padding=70)

    muno_context = load_rgb(muno_case / "current_rgb.png")
    muno_prior = overlay_vector(muno_context, load_rgb(muno_case / "prior_map.png"))
    muno_direct = overlay_vector(muno_context, load_rgb(muno_case / "direct_writeback.png"))
    muno_active = overlay_vector(muno_context, load_rgb(muno_case / "activemap_writeback.png"))
    muno_reference = overlay_vector(muno_context, load_rgb(muno_case / "reference_map.png"))
    muno_bounds = (128, 162, 384, 418)

    spacenet_first = load_rgb(spacenet_case / "first_post_rgb.png")
    spacenet_selected = load_rgb(spacenet_case / "selected_post_rgb.png")
    spacenet_draft = overlay_vector(spacenet_selected, load_rgb(spacenet_case / "selected_change_draft.png"))
    spacenet_defer = overlay_vector(spacenet_selected, load_rgb(spacenet_case / "safe_commit_writeback.png"))
    spacenet_reference_map = load_rgb(spacenet_case / "reference_change.png")
    spacenet_reference = overlay_vector(spacenet_selected, spacenet_reference_map)
    spacenet_bounds = square_crop(
        non_background_mask(spacenet_reference_map),
        image_shape=spacenet_selected.shape[:2],
        minimum=192,
        padding=32,
    )

    setup_style()
    output_dir.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(3, 5, figsize=(7.16, 4.62), constrained_layout=False)
    figure.patch.set_facecolor("white")
    figure.subplots_adjust(left=0.055, right=0.989, top=0.927, bottom=0.085, wspace=0.105, hspace=0.55)

    row_tag(figure, 0.958, "a", "SN7 buildings: frozen editable-map draft")
    for axis, image, title in zip(
        axes[0],
        [sn7_context, sn7_prior, sn7_direct, sn7_reference, sn7_residual],
        ["Current imagery", "Prior map (empty)", "Frozen direct draft", "Reference polygon", "Draft residual"],
    ):
        panel(axis, crop(image, sn7_bounds), title)

    row_tag(figure, 0.650, "b", "MUNO21 roads: a paired controller writeback")
    for axis, image, title, note in zip(
        axes[1],
        [muno_context, muno_prior, muno_direct, muno_active, muno_reference],
        ["Current imagery", "Prior road graph", "Generic direct", "ActiveMap writeback", "Reference graph"],
        [None, None, "missed DELETE", "correct DELETE", None],
    ):
        panel(axis, crop(image, muno_bounds), title, note=note)

    row_tag(figure, 0.342, "c", "SpaceNet8 disaster imagery: selected evidence, then safe defer")
    for axis, image, title, note in zip(
        axes[2],
        [spacenet_first, spacenet_selected, spacenet_draft, spacenet_defer, spacenet_reference],
        ["First POST", "Selected POST", "Selected draft", "Safe Commit map", "Reference change"],
        [None, "one evidence call", "no durable support", "DEFER", None],
    ):
        panel(axis, crop(image, spacenet_bounds), title, note=note)

    draw_legend(figure)
    stem = output_dir / "cross_domain_map_native_qualitative"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "schema_version": "cross-domain-map-native-qualitative-v2",
        "figure_contract": {
            "core_conclusion": "The controller and writeback interface remain visually auditable across polygon, road-graph, and multi-observation map settings; direct perception, terminal writeback, and defer are shown as distinct states.",
            "archetype": "asymmetric mixed-modality image plate",
            "evidence_status": "validation-only qualitative examples",
            "backend": "python",
            "case_policy": "fixed registered validation cases; no visual re-ranking during assembly",
        },
        "sources": [source.__dict__ for source in sources],
        "case_roles": {
            "SN7": "frozen direct-perception editable polygon draft; not a controller performance claim",
            "MUNO21": "paired generic-direct versus ActiveMap road DELETE writeback",
            "SpaceNet8": "one real multi-POST evidence selection followed by a frozen Safe Commit defer",
        },
        "crop_bounds_xyxy": {
            "SN7": list(sn7_bounds),
            "MUNO21": list(muno_bounds),
            "SpaceNet8": list(spacenet_bounds),
        },
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "outputs": [f"{stem.name}{suffix}" for suffix in (".pdf", ".svg", ".png", ".tiff")],
        "image_integrity": "No contrast, color, or geometry edits are applied to source panels. The figure uses registered local crops and overlays original rendered boundary pixels onto the matching current imagery for white-background map panels.",
    }
    (output_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(render(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
