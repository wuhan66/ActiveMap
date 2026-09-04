#!/usr/bin/env python3
"""Assemble the main-paper map-native and controller qualitative figure.

The left plate shows editable vector candidates produced by the frozen
perception backend.  The right cards show independent, predeclared,
validation-only controller rollouts.  They deliberately remain visually
separate: perception panels are not presented as policy-level outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP_ROOT = ROOT / "docs" / "figures" / "sn7_map_native_qualitative_20260809_full"
DEFAULT_TRACE_ROOT = ROOT / "docs" / "figures" / "safe_commit_qualitative_20260801"
DEFAULT_OUTPUT = ROOT / "docs" / "figures" / "map_native_control_qualitative_20260816_v2"

MAP_CASES = (
    ("008_add_success", "ADD"),
    ("020_reshape_success", "RESHAPE"),
)
TRACE_CASES = (
    {
        "folder": "001_reshape_sn7-e6962896a645c3bb__b4p5__s0",
        "kind": "evidence_revision",
        "title": "Evidence revises the edit",
        "draft": "DELETE",
        "final": "RESHAPE",
        "status": "commit",
        "detail": "one acquisition, two tool calls",
    },
    {
        "folder": "004_keep_sn7-9d57628ebca7de9b__b4p5__s0",
        "kind": "safe_rejection",
        "title": "Safe Commit blocks a false write",
        "draft": "DELETE",
        "final": "KEEP",
        "status": "reject",
        "detail": "risk 0.907 > threshold 0.904",
    },
)

INK = "#1C232A"
MUTED = "#59636E"
RULE = "#D3D8DC"
PRIOR = "#F4A200"
REFERENCE = "#009E73"
WRITEBACK = "#0072B2"
TP = "#009E73"
FP = "#D55E00"
FN = "#56B4E9"
ACTION = "#3977A8"
SAFE = "#2E8B57"
RISK = "#C03D3E"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.65,
        }
    )


def image_axis(axis: plt.Axes, path: Path, *, title: str | None = None) -> None:
    axis.imshow(plt.imread(path))
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_color(RULE)
    if title:
        axis.set_title(title, fontsize=9.5, fontweight="bold", color=INK, pad=4)


def require_map_case(root: Path, case: str) -> list[Path]:
    case_dir = root / case
    manifest_path = case_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("test_assets_read") is not False:
        raise ValueError(f"{case} is not validation-only")
    if "changemamba" not in set(manifest.get("methods", ())):
        raise ValueError(f"{case} does not provide the frozen ChangeMamba candidate")
    required = [
        case_dir / "02_prior_vector.png",
        case_dir / "03_reference_vector.png",
        case_dir / "04_changemamba_writeback.png",
        case_dir / "05_changemamba_residual.png",
        manifest_path,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    return required


def require_trace_case(root: Path, descriptor: dict[str, str]) -> tuple[Path, dict[str, Any], list[Path]]:
    summary_path = root / "summary.json"
    summary = read_json(summary_path)
    if summary.get("test_assets_read") is not False:
        raise ValueError("controller visual source must be validation-only")
    row = next((item for item in summary.get("samples", []) if item.get("folder") == descriptor["folder"]), None)
    if row is None:
        raise ValueError(f"trace case is not registered: {descriptor['folder']}")
    case_dir = root / descriptor["folder"]
    metadata_path = case_dir / "metadata.json"
    metadata = read_json(metadata_path)
    if descriptor["kind"] == "evidence_revision":
        if not (metadata.get("terminal_correct") and metadata.get("target_edit") == "RESHAPE"):
            raise ValueError("evidence-revision case no longer matches the fixed protocol")
        actions = [event.get("action") for event in metadata.get("events", [])]
        if "ACQUIRE" not in actions or "COMMIT" not in actions:
            raise ValueError("evidence-revision case is missing acquisition or commit")
    else:
        if not (metadata.get("terminal_correct") and metadata.get("target_edit") == "KEEP"):
            raise ValueError("safe-rejection case no longer matches the fixed protocol")
    anchor = case_dir / "01_anchor_rgb.png"
    for path in (anchor, metadata_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return anchor, metadata, [anchor, metadata_path, summary_path]


def draw_arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=1.2,
            color=color,
            transform=axis.transAxes,
        )
    )


def trace_card(
    figure: plt.Figure,
    slot: Any,
    panel: str,
    descriptor: dict[str, str],
    anchor: Path,
    metadata: dict[str, Any],
) -> None:
    card_grid = slot.subgridspec(1, 2, width_ratios=[0.84, 1.76], wspace=0.12)
    image_axis = figure.add_subplot(card_grid[0, 0])
    image_axis.imshow(plt.imread(anchor))
    image_axis.set_xticks([])
    image_axis.set_yticks([])
    for spine in image_axis.spines.values():
        spine.set_color(RULE)
        spine.set_linewidth(0.55)
    image_axis.text(
        0.03,
        0.95,
        panel,
        transform=image_axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="white",
        bbox={"facecolor": INK, "edgecolor": "none", "pad": 1.4},
    )

    axis = figure.add_subplot(card_grid[0, 1])
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    title = "Evidence revises\nthe edit" if descriptor["kind"] == "evidence_revision" else "Safe Commit blocks\na false write"
    axis.text(0.0, 0.96, title, fontsize=9.6, fontweight="bold", color=INK, va="top", linespacing=1.05)
    axis.text(0.0, 0.72, f"Direct draft: {descriptor['draft']}", fontsize=8.1, color=MUTED, va="top")

    if descriptor["kind"] == "evidence_revision":
        axis.text(0.01, 0.47, "ACQUIRE", fontsize=7.45, color=ACTION, fontweight="bold", va="center")
        axis.text(0.36, 0.47, "TOOL", fontsize=7.45, color=ACTION, fontweight="bold", va="center")
        axis.text(0.67, 0.47, "COMMIT", fontsize=7.45, color=SAFE, fontweight="bold", va="center")
        draw_arrow(axis, (0.25, 0.47), (0.33, 0.47), ACTION)
        draw_arrow(axis, (0.54, 0.47), (0.63, 0.47), ACTION)
        axis.text(0.0, 0.23, f"Final map edit: {descriptor['final']}", fontsize=8.1, color=SAFE, fontweight="bold", va="center")
        axis.text(0.0, 0.07, descriptor["detail"], fontsize=7.0, color=MUTED, va="center")
    else:
        axis.text(0.01, 0.47, "SAFE COMMIT", fontsize=7.45, color=RISK, fontweight="bold", va="center")
        draw_arrow(axis, (0.54, 0.47), (0.70, 0.47), RISK)
        axis.text(0.73, 0.47, "REJECT", fontsize=7.45, color=RISK, fontweight="bold", va="center")
        axis.text(0.0, 0.23, f"Final map edit: {descriptor['final']}", fontsize=8.1, color=SAFE, fontweight="bold", va="center")
        axis.text(0.0, 0.07, descriptor["detail"], fontsize=7.0, color=MUTED, va="center")


def assemble(map_root: Path, trace_root: Path, output_dir: Path) -> dict[str, Any]:
    setup_style()
    map_inputs: list[Path] = []
    for case, _ in MAP_CASES:
        map_inputs.extend(require_map_case(map_root, case))
    trace_inputs: list[Path] = []
    resolved_traces: list[tuple[dict[str, str], Path, dict[str, Any]]] = []
    for descriptor in TRACE_CASES:
        anchor, metadata, inputs = require_trace_case(trace_root, descriptor)
        trace_inputs.extend(inputs)
        resolved_traces.append((descriptor, anchor, metadata))

    output_dir.mkdir(parents=True, exist_ok=False)
    fig = plt.figure(figsize=(7.16, 5.35), constrained_layout=False)
    fig.patch.set_facecolor("white")
    grid = GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[1.0, 1.0, 0.94],
        left=0.055,
        right=0.985,
        top=0.890,
        bottom=0.055,
        wspace=0.12,
        hspace=0.38,
    )

    fig.text(0.055, 0.972, "Map-native writeback and audited decisions", fontsize=12.5, fontweight="bold", color=INK, va="top")

    map_headers = ("Prior map", "Direct writeback", "Reference map", "Residual")
    map_files = ("02_prior_vector.png", "04_changemamba_writeback.png", "03_reference_vector.png", "05_changemamba_residual.png")
    for row, (case, operation) in enumerate(MAP_CASES):
        for col, (header, name) in enumerate(zip(map_headers, map_files)):
            axis = fig.add_subplot(grid[row, col])
            image_axis(axis, map_root / case / name, title=header if row == 0 else None)
            if col == 0:
                axis.text(
                    -0.12,
                    0.95,
                    "a" if row == 0 else "b",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10.5,
                    fontweight="bold",
                    color="white",
                    bbox={"facecolor": INK, "edgecolor": "none", "pad": 1.4},
                )
                axis.text(-0.10, -0.16, operation, transform=axis.transAxes, ha="left", va="top", fontsize=8.7, fontweight="bold", color=INK)

    for column, (descriptor, anchor, metadata) in enumerate(resolved_traces):
        trace_card(
            fig,
            grid[2, 0:2] if column == 0 else grid[2, 2:4],
            "c" if column == 0 else "d",
            descriptor,
            anchor,
            metadata,
        )

    stem = output_dir / "map_native_control_qualitative"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    all_inputs = map_inputs + trace_inputs
    manifest = {
        "schema_version": "map-native-control-qualitative-v2",
        "split": "val",
        "test_assets_read": False,
        "figure_contract": {
            "core_conclusion": "Editable map drafts and controller decisions remain separately auditable: evidence can revise an edit, and Safe Commit can prevent a high-risk false write.",
            "archetype": "asymmetric mixed-modality figure",
            "backend": "python",
            "case_policy": "fixed predeclared validation-only inputs; no rank-based reselection at assembly time",
        },
        "map_cases": [{"case": case, "operation": operation, "backend": "ChangeMamba"} for case, operation in MAP_CASES],
        "controller_cases": [
            {
                "folder": descriptor["folder"],
                "kind": descriptor["kind"],
                "target_edit": metadata["target_edit"],
                "predicted_edit": metadata["predicted_edit"],
                "terminal_correct": metadata["terminal_correct"],
                "actions": [event["action"] for event in metadata["events"]],
            }
            for descriptor, _, metadata in resolved_traces
        ],
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(all_inputs))},
        "outputs": [
            path.name
            for path in (
                stem.with_suffix(".pdf"),
                stem.with_suffix(".svg"),
                stem.with_suffix(".png"),
                stem.with_suffix(".tiff"),
            )
        ],
        "image_integrity": "Existing validation-only raster panels are reused without crop, contrast, or geometry modification; this script only composes labelled panels and vector annotations.",
    }
    (output_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(assemble(args.map_root, args.trace_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
