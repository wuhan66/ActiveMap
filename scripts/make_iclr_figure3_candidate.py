#!/usr/bin/env python3
"""Render a provenance-locked, map-native candidate ICLR qualitative plate.

The fixed SN7, MUNO21, and SpaceNet8 validation cases are already registered
in the v40 qualitative manifest. This renderer preserves those cases and their
registered crops, composites sparse map layers over their real image context,
and writes a candidate-only asset outside the frozen submission package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_image(path: Path, crop: tuple[int, int, int, int]) -> Image.Image:
    return Image.open(path).convert("RGB").crop(crop)


def overlay_on_context(context: Image.Image, layer: Image.Image) -> Image.Image:
    """Treat near-white pixels as transparent for sparse vector/mask layers."""
    foreground = np.asarray(layer.convert("RGB"))
    background = np.asarray(context.convert("RGB"))
    alpha = np.any(foreground < 245, axis=2)[..., None]
    return Image.fromarray(np.where(alpha, foreground, background).astype(np.uint8), mode="RGB")


def resolve_inputs(
    manifest: dict[str, Any], project_root: Path
) -> dict[str, Path]:
    if (
        manifest.get("figure_contract", {}).get("case_policy")
        != "fixed registered validation cases; no visual re-ranking during assembly"
    ):
        raise ValueError("qualitative manifest does not declare fixed registered cases")
    if any(case.get("split") != "val" or case.get("test_assets_read") is not False for case in manifest["sources"]):
        raise ValueError("candidate plate must only use held-out validation cases")
    resolved: dict[str, Path] = {}
    for raw_path, expected_hash in manifest["input_sha256"].items():
        path = project_root / Path(raw_path.replace("\\", "/"))
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"source hash mismatch: {path}")
        resolved[path.name] = path
    return resolved


def add_row_header(figure: plt.Figure, axis: plt.Axes, text: str) -> None:
    bounds = axis.get_position()
    figure.text(
        0.012,
        bounds.y1 + 0.022,
        text,
        fontsize=8.1,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def render_row(
    axes: list[plt.Axes],
    images: list[Image.Image],
    labels: list[str],
) -> None:
    for axis, image, label in zip(axes, images, labels):
        axis.imshow(image, interpolation="none")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#D4D9DE")
            spine.set_linewidth(0.6)
        axis.set_xlabel(label, fontsize=6.7, labelpad=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_root", type=Path)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    source_manifest_path = (
        args.paper_root / "figures" / "results" / "cross_domain_map_native_qualitative.source_manifest.json"
    )
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    sources = resolve_inputs(manifest, args.project_root)
    crops = {name: tuple(bounds) for name, bounds in manifest["crop_bounds_xyxy"].items()}
    args.output_dir.mkdir(parents=True, exist_ok=False)

    sn7_context = load_image(sources["01_context.png"], crops["SN7"])
    sn7_prior = load_image(sources["02_prior_vector.png"], crops["SN7"])
    sn7_reference = load_image(sources["03_reference_vector.png"], crops["SN7"])
    sn7_direct = load_image(sources["04_changemamba_writeback.png"], crops["SN7"])
    sn7_residual = load_image(sources["05_changemamba_residual.png"], crops["SN7"])

    muno_context = load_image(sources["current_rgb.png"], crops["MUNO21"])
    muno_prior = overlay_on_context(muno_context, load_image(sources["prior_map.png"], crops["MUNO21"]))
    muno_direct = overlay_on_context(muno_context, load_image(sources["direct_writeback.png"], crops["MUNO21"]))
    muno_active = overlay_on_context(muno_context, load_image(sources["activemap_writeback.png"], crops["MUNO21"]))
    muno_reference = overlay_on_context(muno_context, load_image(sources["reference_map.png"], crops["MUNO21"]))

    sn8_first = load_image(sources["first_post_rgb.png"], crops["SpaceNet8"])
    sn8_selected = load_image(sources["selected_post_rgb.png"], crops["SpaceNet8"])
    sn8_draft = overlay_on_context(sn8_selected, load_image(sources["selected_change_draft.png"], crops["SpaceNet8"]))
    sn8_safe = overlay_on_context(sn8_selected, load_image(sources["safe_commit_writeback.png"], crops["SpaceNet8"]))
    sn8_reference = overlay_on_context(sn8_selected, load_image(sources["reference_change.png"], crops["SpaceNet8"]))

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(3, 5, figsize=(7.16, 6.1), constrained_layout=False)
    figure.subplots_adjust(left=0.055, right=0.995, top=0.85, bottom=0.075, wspace=0.075, hspace=0.70)
    figure.text(
        0.012,
        0.988,
        "Map-native validation cases distinguish direct proposals, executed writebacks, and safe defer",
        fontsize=10.0,
        fontweight="bold",
        ha="left",
        va="top",
    )

    render_row(
        list(axes[0]),
        [sn7_context, sn7_prior, sn7_direct, sn7_reference, sn7_residual],
        ["Current imagery", "Prior map", "Frozen direct draft", "Reference map", "Residual"],
    )
    render_row(
        list(axes[1]),
        [muno_context, muno_prior, muno_direct, muno_active, muno_reference],
        ["Current imagery", "Prior road graph", "Generic direct", "ActiveMap writeback", "Reference graph"],
    )
    render_row(
        list(axes[2]),
        [sn8_first, sn8_selected, sn8_draft, sn8_safe, sn8_reference],
        ["First POST", "Selected POST", "Selected draft", "Safe Commit: DEFER", "Reference change"],
    )

    add_row_header(figure, axes[0, 0], "(a) SN7 buildings: frozen direct-perception interface audit")
    add_row_header(figure, axes[1, 0], "(b) MUNO21 roads: paired controller writeback")
    add_row_header(figure, axes[2, 0], "(c) SpaceNet8 disasters: selected evidence followed by Safe Commit defer")
    figure.text(
        0.012,
        0.018,
        "Overlay colors: orange = prior, blue = direct proposal, green = reference or accepted writeback, "
        "red-orange = false write, cyan = missed update. All cases are fixed held-out validation examples.",
        fontsize=6.55,
        color="#4E5963",
        ha="left",
        va="bottom",
    )

    prefix = args.output_dir / "iclr_figure3_map_native_candidate"
    figure.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    figure.savefig(prefix.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    figure.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)

    output_manifest = {
        "schema_version": "iclr-figure3-map-native-candidate-v1",
        "candidate_only": True,
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_images": {str(path.resolve()): sha256(path) for path in sorted(sources.values())},
        "registered_crops_xyxy": crops,
        "case_policy": manifest["figure_contract"]["case_policy"],
        "renderer_reads_test_assets": False,
        "outputs": [path.name for path in sorted(args.output_dir.glob("iclr_figure3_map_native_candidate.*"))],
    }
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
