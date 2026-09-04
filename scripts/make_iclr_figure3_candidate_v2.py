#!/usr/bin/env python3
"""Render a provenance-locked, map-native Figure 3 candidate outside v40.

The renderer keeps the registered MUNO21 and SpaceNet8 held-out validation
examples, but gives the executable MUNO21 writeback visual priority. It never
opens test data, changes source pixels, or makes a policy-performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image


INK = "#1C232A"
MUTED = "#58636E"
RULE = "#D5DCE1"
FALSE = "#D55E00"
SAFE = "#009E73"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def vector_overlay(context: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Overlay sparse vector pixels while preserving the registered RGB image."""
    if context.shape != layer.shape:
        raise ValueError(f"image/layer shape mismatch: {context.shape} != {layer.shape}")
    alpha = np.any(layer < 245, axis=2)
    result = context.copy()
    result[alpha] = layer[alpha]
    return result


def vector_mask(image: np.ndarray) -> np.ndarray:
    return np.any(image < 245, axis=2)


def difference_crop(
    direct: np.ndarray,
    reference: np.ndarray,
    *,
    side: int = 180,
    padding: int = 28,
) -> tuple[int, int, int, int]:
    """Choose a deterministic crop around disagreement in a fixed source case."""
    changed = vector_mask(direct) ^ vector_mask(reference)
    ys, xs = np.where(changed)
    if len(xs) == 0:
        raise ValueError("registered MUNO21 direct/reference maps have no disagreement")
    height, width = changed.shape
    center_x = int(round((xs.min() + xs.max()) / 2))
    center_y = int(round((ys.min() + ys.max()) / 2))
    required = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) + 2 * padding
    side = min(max(side, required), width, height)
    x0 = min(max(center_x - side // 2, 0), width - side)
    y0 = min(max(center_y - side // 2, 0), height - side)
    return x0, y0, x0 + side, y0 + side


def crop(image: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    return image[y0:y1, x0:x1]


def add_panel(
    axis: plt.Axes,
    image: np.ndarray,
    title: str,
    *,
    tag: str | None = None,
    tag_color: str = INK,
) -> None:
    axis.imshow(image, interpolation="none")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=7.2, loc="left", color=INK, fontweight="bold", pad=3.5)
    for spine in axis.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color(RULE)
    if tag:
        axis.text(
            0.03,
            0.04,
            tag,
            transform=axis.transAxes,
            fontsize=6.3,
            color="white",
            fontweight="bold",
            ha="left",
            va="bottom",
            bbox={"facecolor": tag_color, "edgecolor": "none", "pad": 1.35},
        )


def add_zoom(
    axis: plt.Axes,
    image: np.ndarray,
    bounds: tuple[int, int, int, int],
    *,
    edge_color: str,
    label: str,
) -> None:
    inset = inset_axes(axis, width="39%", height="39%", loc="lower right", borderpad=0.35)
    inset.imshow(crop(image, bounds), interpolation="none")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(1.45)
        spine.set_color(edge_color)
    inset.text(
        0.04,
        0.05,
        label,
        transform=inset.transAxes,
        fontsize=5.7,
        color="white",
        fontweight="bold",
        ha="left",
        va="bottom",
        bbox={"facecolor": edge_color, "edgecolor": "none", "pad": 0.9},
    )


def load_registered_sources(paper_root: Path, project_root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    manifest_path = paper_root / "figures" / "results" / "cross_domain_map_native_qualitative.source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["figure_contract"]["case_policy"] != "fixed registered validation cases; no visual re-ranking during assembly":
        raise ValueError("source manifest does not declare the registered case policy")
    selected_sources = {source["dataset"]: source for source in manifest["sources"]}
    for dataset in ("MUNO21", "SpaceNet8"):
        source = selected_sources[dataset]
        if source["split"] != "val" or source["test_assets_read"] is not False:
            raise ValueError(f"{dataset} source is not an eligible held-out validation case")
    paths: dict[str, Path] = {}
    for raw_path, expected_hash in manifest["input_sha256"].items():
        path = project_root / Path(raw_path.replace("\\", "/"))
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"registered source hash mismatch: {path}")
        paths[path.name] = path
    return paths, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_root", type=Path)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--allow-existing-dir",
        action="store_true",
        help="write prefixed candidate assets into an existing directory without overwriting files",
    )
    args = parser.parse_args()
    if args.output_dir.exists() and not args.allow_existing_dir:
        raise FileExistsError(f"refusing to overwrite candidate output: {args.output_dir}")

    stem = args.output_dir / "iclr_figure3_map_native_candidate_v2"
    manifest_output = args.output_dir / "iclr_figure3_map_native_candidate_v2.source_manifest.json"
    outputs_to_write = [stem.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png", ".tiff")]
    collisions = [path for path in [*outputs_to_write, manifest_output] if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite candidate assets: {collisions}")

    paths, source_manifest = load_registered_sources(args.paper_root, args.project_root)
    args.output_dir.mkdir(parents=True, exist_ok=args.allow_existing_dir)

    muno_context = load_rgb(paths["current_rgb.png"])
    muno_prior = load_rgb(paths["prior_map.png"])
    muno_direct = load_rgb(paths["direct_writeback.png"])
    muno_active = load_rgb(paths["activemap_writeback.png"])
    muno_reference = load_rgb(paths["reference_map.png"])
    muno_prior_overlay = vector_overlay(muno_context, muno_prior)
    muno_direct_overlay = vector_overlay(muno_context, muno_direct)
    muno_active_overlay = vector_overlay(muno_context, muno_active)
    muno_reference_overlay = vector_overlay(muno_context, muno_reference)
    muno_zoom = difference_crop(muno_direct, muno_reference)

    sn8_first = load_rgb(paths["first_post_rgb.png"])
    sn8_selected = load_rgb(paths["selected_post_rgb.png"])
    sn8_draft = vector_overlay(sn8_selected, load_rgb(paths["selected_change_draft.png"]))
    sn8_defer = vector_overlay(sn8_selected, load_rgb(paths["safe_commit_writeback.png"]))
    sn8_reference = vector_overlay(sn8_selected, load_rgb(paths["reference_change.png"]))

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure = plt.figure(figsize=(7.16, 5.55), constrained_layout=False)
    figure.patch.set_facecolor("white")
    grid = figure.add_gridspec(2, 1, height_ratios=(1.25, 1.0), hspace=0.34)
    top = grid[0].subgridspec(1, 4, wspace=0.075)
    bottom = grid[1].subgridspec(1, 5, wspace=0.075)
    top_axes = [figure.add_subplot(top[0, index]) for index in range(4)]
    bottom_axes = [figure.add_subplot(bottom[0, index]) for index in range(5)]

    figure.text(0.012, 0.972, "a", fontsize=9.2, color="white", fontweight="bold", va="top", ha="left", bbox={"facecolor": INK, "edgecolor": "none", "pad": 1.4})
    figure.text(0.041, 0.972, "MUNO21: optional verification changes an executable road writeback", fontsize=8.7, color=INK, fontweight="bold", va="top", ha="left")
    for axis, image, title, tag, color in zip(
        top_axes,
        [muno_prior_overlay, muno_direct_overlay, muno_active_overlay, muno_reference_overlay],
        ["Editable prior", "Generic direct", "ActiveMap writeback", "Reference graph"],
        [None, "missed DELETE", "correct DELETE", None],
        [INK, FALSE, SAFE, INK],
    ):
        add_panel(axis, image, title, tag=tag, tag_color=color)
    add_zoom(top_axes[1], muno_direct_overlay, muno_zoom, edge_color=FALSE, label="missed")
    add_zoom(top_axes[2], muno_active_overlay, muno_zoom, edge_color=SAFE, label="correct")

    figure.text(0.012, 0.506, "b", fontsize=9.2, color="white", fontweight="bold", va="top", ha="left", bbox={"facecolor": INK, "edgecolor": "none", "pad": 1.4})
    figure.text(0.041, 0.506, "SpaceNet8: one recorded evidence call is separate from the final Safe Commit decision", fontsize=8.7, color=INK, fontweight="bold", va="top", ha="left")
    for axis, image, title, tag, color in zip(
        bottom_axes,
        [sn8_first, sn8_selected, sn8_draft, sn8_defer, sn8_reference],
        ["First POST", "Selected POST", "Selected draft", "Safe Commit map", "Reference change"],
        [None, "one evidence call", "proposal", "DEFER", None],
        [INK, "#0072B2", "#0072B2", INK, INK],
    ):
        add_panel(axis, image, title, tag=tag, tag_color=color)

    figure.text(0.012, 0.023, "Blue: proposal; green: reference or accepted writeback; orange: prior. All panels are fixed held-out validation cases.", fontsize=6.45, color=MUTED, ha="left", va="bottom")

    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)

    outputs = sorted(args.output_dir.glob("iclr_figure3_map_native_candidate_v2.*"))
    receipt = {
        "schema_version": "iclr-figure3-map-native-candidate-v2",
        "candidate_only": True,
        "source_manifest_sha256": sha256(
            args.paper_root / "figures" / "results" / "cross_domain_map_native_qualitative.source_manifest.json"
        ),
        "source_images": {str(path.relative_to(args.project_root)): sha256(path) for path in sorted(paths.values())},
        "case_policy": source_manifest["figure_contract"]["case_policy"],
        "renderer_reads_test_assets": False,
        "deterministic_muno_difference_crop_xyxy": list(muno_zoom),
        "outputs": {path.name: sha256(path) for path in outputs},
    }
    manifest_output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
