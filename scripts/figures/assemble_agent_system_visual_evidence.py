#!/usr/bin/env python3
"""Assemble the operation-aligned SN7 and Agent/System evidence figures.

The visual plate uses three fixed SN7 validation cases rerun with the registered
updater checkpoints. A separate audited rollout exposes candidate value, belief
revision, and the structured action sequence. Every panel is traceable to a
validation artifact and uses a common crop within each case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SN7_CASE_ROOT = (
    ROOT
    / "docs"
    / "figures"
    / "sn7_latest_checkpoint_qualitative_20260903"
)
SN7_CASES = {
    "ADD": SN7_CASE_ROOT / "01_add_clean",
    "DELETE": SN7_CASE_ROOT / "02_delete_clean",
    "RESHAPE": SN7_CASE_ROOT / "03_reshape_clean",
}
SN7_CASE_METRICS = {
    "ADD": {"frozen": 0.906250, "latest": 0.146239},
    "DELETE": {"frozen": 1.000000, "latest": 0.000000},
    "RESHAPE": {"frozen": 0.853846, "latest": 0.758709},
}
SN7_SELECTION_MANIFEST = SN7_CASE_ROOT / "selection_manifest.json"
SN7_TRACE = (
    ROOT
    / ".codex-results"
    / "safe_commit_qualitative_seed21_v2"
    / "selected_traces.jsonl"
)
SN7_VISUAL = (
    ROOT
    / "docs"
    / "figures"
    / "safe_commit_qualitative_20260801"
    / "001_reshape_sn7-e6962896a645c3bb__b4p5__s0"
)
DEFAULT_OUTPUT = ROOT / "docs" / "figures" / "agent_system_visual_evidence_20260903_v4"

TRACE_SAMPLE = "sn7-e6962896a645c3bb__b4p5__s0"
INK = "#18212A"
MUTED = "#5C6670"
RULE = "#D5DADF"
BLUE = "#1877B9"
RED = "#D64F45"
GREEN = "#159A68"
ORANGE = "#E69F00"
LIGHT_BLUE = "#E8F3F8"
LIGHT_GREEN = "#E5F4ED"
LIGHT_RED = "#FAE9E7"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_trace(path: Path, sample_id: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("sample_id") == sample_id:
                return row
    raise ValueError(f"trace sample not found: {sample_id}")


def require_sources() -> None:
    required = [SN7_SELECTION_MANIFEST, SN7_TRACE, SN7_VISUAL / "01_anchor_rgb.png"]
    for case_dir in SN7_CASES.values():
        for filename in (
            "01_current_rgb.png",
            "02_prior_mask.png",
            "03_reference_final_mask.png",
            "07_frozen_overlay.png",
            "08_frozen_tp_fp_fn.png",
            "11_latest_overlay.png",
            "12_latest_tp_fp_fn.png",
        ):
            required.append(case_dir / "zoom" / filename)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing inputs:\n" + "\n".join(missing))
    manifest = read_json(SN7_SELECTION_MANIFEST)
    if manifest.get("test_assets_read") is not False:
        raise ValueError("SN7 figure inputs must remain validation-only")
    if set(manifest.get("operations", ())) != set(SN7_CASES):
        raise ValueError("SN7 selection manifest must register ADD/DELETE/RESHAPE")


def sn7_map_panels(case_dir: Path) -> dict[str, np.ndarray]:
    zoom = case_dir / "zoom"
    return {
        "current": load_rgb(zoom / "01_current_rgb.png"),
        "prior": load_rgb(zoom / "02_prior_mask.png"),
        "reference": load_rgb(zoom / "03_reference_final_mask.png"),
        "proposal": load_rgb(zoom / "07_frozen_overlay.png"),
        "residual": load_rgb(zoom / "08_frozen_tp_fp_fn.png"),
        "latest": load_rgb(zoom / "11_latest_overlay.png"),
        "latest_residual": load_rgb(zoom / "12_latest_tp_fp_fn.png"),
    }


def candidate_matrix(trace: dict[str, Any]) -> tuple[np.ndarray, list[str], list[int], tuple[int, int]]:
    event = trace["events"][0]
    scores = event["ranker_scores"]
    candidates = event["observable_state"]["candidate_evidence"]
    timestamps = sorted({str(item["timestamp"]) for item in candidates})
    scales = sorted({int(item["scale"]) for item in candidates})
    matrix = np.full((len(scales), len(timestamps)), np.nan, dtype=np.float32)
    selected = str(event["executed_action"]["evidence_id"])
    selected_cell = (-1, -1)
    for item in candidates:
        evidence_id = str(item["evidence_id"])
        if evidence_id not in scores:
            continue
        row = scales.index(int(item["scale"]))
        column = timestamps.index(str(item["timestamp"]))
        matrix[row, column] = float(scores[evidence_id])
        if evidence_id == selected:
            selected_cell = (row, column)
    if selected_cell == (-1, -1):
        raise ValueError("selected evidence is absent from the candidate matrix")
    return matrix, timestamps, scales, selected_cell


def belief_matrix(trace: dict[str, Any]) -> np.ndarray:
    select_events = [event for event in trace["events"] if "observable_state" in event]
    if len(select_events) < 2:
        raise ValueError("belief panel requires pre- and post-evidence states")
    before = select_events[0]["observable_state"]["belief"]["edit_probabilities"]
    after = select_events[-1]["observable_state"]["belief"]["edit_probabilities"]
    return np.asarray([before, after], dtype=np.float32).T


def image_panel(axis: plt.Axes, image: np.ndarray, title: str, note: str) -> None:
    axis.imshow(image, interpolation="antialiased")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=7.7, fontweight="bold", loc="left", color=INK, pad=3)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(RULE)
        spine.set_linewidth(0.6)
    axis.text(
        0.03,
        0.04,
        note,
        transform=axis.transAxes,
        color="white",
        fontsize=6.2,
        fontweight="bold",
        ha="left",
        va="bottom",
        bbox={"facecolor": INK, "edgecolor": "none", "pad": 1.2, "alpha": 0.88},
    )


def candidate_panel(axis: plt.Axes, trace: dict[str, Any]) -> None:
    matrix, timestamps, scales, selected = candidate_matrix(trace)
    finite = matrix[np.isfinite(matrix)]
    norm = TwoSlopeNorm(vmin=float(finite.min()), vcenter=0.0, vmax=float(finite.max()))
    masked = np.ma.masked_invalid(matrix)
    image = axis.imshow(masked, cmap="RdBu", norm=norm, aspect="auto", interpolation="nearest")
    axis.set_title("Candidate value surface", fontsize=7.7, fontweight="bold", loc="left", color=INK, pad=3)
    labels = [value.replace("_", "\n") for value in timestamps]
    axis.set_xticks(range(len(labels)), labels, fontsize=5.1)
    axis.set_yticks(range(len(scales)), [f"{scale}x" for scale in scales], fontsize=5.6)
    axis.tick_params(length=0, pad=1.5)
    row, column = selected
    axis.add_patch(Rectangle((column - 0.47, row - 0.47), 0.94, 0.94, fill=False, edgecolor=GREEN, linewidth=2.0))
    axis.text(column, row, "*", ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
    colorbar = axis.figure.colorbar(image, ax=axis, orientation="horizontal", fraction=0.10, pad=0.22, aspect=24)
    colorbar.set_label("predicted evidence value", fontsize=5.4, labelpad=1)
    colorbar.ax.tick_params(labelsize=5.2, length=2)
    colorbar.outline.set_linewidth(0.45)
    axis.text(0.99, 1.03, "selected", transform=axis.transAxes, ha="right", va="bottom", fontsize=5.7, color=GREEN, fontweight="bold")


def belief_panel(axis: plt.Axes, trace: dict[str, Any]) -> None:
    belief = belief_matrix(trace)
    image = axis.imshow(belief, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
    axis.set_title("Belief revision", fontsize=7.7, fontweight="bold", loc="left", color=INK, pad=3)
    axis.set_xticks([0, 1], ["before", "after"], fontsize=5.8)
    axis.set_yticks(range(4), ["K", "A", "D", "R"], fontsize=5.7)
    axis.tick_params(length=0, pad=1.5)
    for row in range(4):
        for column in range(2):
            value = float(belief[row, column])
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=5.5, color="white" if value > 0.45 else INK, fontweight="bold")
    axis.add_patch(Rectangle((0.53, 2.53), 0.94, 0.94, fill=False, edgecolor=GREEN, linewidth=1.6))
    colorbar = axis.figure.colorbar(image, ax=axis, orientation="horizontal", fraction=0.10, pad=0.22, aspect=15)
    colorbar.set_label("K/A/D/R: KEEP, ADD, DELETE, RESHAPE", fontsize=5.2, labelpad=1)
    colorbar.ax.tick_params(labelsize=5.2, length=2)


def action_panel(axis: plt.Axes, trace: dict[str, Any]) -> None:
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title("Structured decision", fontsize=7.7, fontweight="bold", loc="left", color=INK, pad=3)
    labels = ["ACQUIRE", "TOOL", "COMMIT"]
    colors = [BLUE, ORANGE, GREEN]
    xs = [0.16, 0.50, 0.84]
    for index, (x, label, color) in enumerate(zip(xs, labels, colors)):
        axis.scatter([x], [0.66], s=410, color=color, edgecolor="white", linewidth=1.1, zorder=3)
        axis.text(x, 0.66, str(index + 1), color="white", ha="center", va="center", fontsize=7.0, fontweight="bold")
        axis.text(x, 0.46, label, color=color, ha="center", va="center", fontsize=5.9, fontweight="bold")
        if index < len(xs) - 1:
            axis.add_patch(FancyArrowPatch((x + 0.09, 0.66), (xs[index + 1] - 0.09, 0.66), arrowstyle="-|>", mutation_scale=7, linewidth=1.0, color="#8A949C"))
    before = trace["events"][0]["observable_state"]["direct_draft"]["edit"]
    final = trace["predicted_edit"]
    axis.text(0.03, 0.22, before, fontsize=7.4, color=RED, fontweight="bold", va="center")
    axis.add_patch(FancyArrowPatch((0.28, 0.22), (0.55, 0.22), arrowstyle="-|>", mutation_scale=8, linewidth=1.2, color="#8A949C"))
    axis.text(0.62, 0.22, final, fontsize=7.4, color=GREEN, fontweight="bold", va="center")
    axis.text(0.03, 0.08, f"quality +{float(trace['quality_gain']):.3f}  |  cost {float(trace['spent_cost']):.2f}", fontsize=5.8, color=MUTED, va="center")


def panel_label(figure: plt.Figure, x: float, y: float, label: str, title: str) -> None:
    figure.text(x, y, label, fontsize=9.0, fontweight="bold", color="white", ha="center", va="center", bbox={"facecolor": INK, "edgecolor": "none", "pad": 1.1})
    figure.text(x + 0.022, y, title, fontsize=8.4, fontweight="bold", color=INK, ha="left", va="center")


def save_asset(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image).save(path)


def render_checkpoint_diagnostic(
    output_dir: Path,
    operation_panels: dict[str, dict[str, np.ndarray]],
) -> list[str]:
    figure, axes = plt.subplots(3, 6, figsize=(7.16, 4.35), constrained_layout=False)
    figure.subplots_adjust(left=0.075, right=0.992, top=0.91, bottom=0.055, wspace=0.10, hspace=0.24)
    titles = ["Current image", "Reference", "Frozen V5-B", "V5-B residual", "Latest repair", "Latest residual"]
    keys = ["current", "reference", "proposal", "residual", "latest", "latest_residual"]
    for row, operation in enumerate(SN7_CASES):
        metrics = SN7_CASE_METRICS[operation]
        notes = [
            "observation",
            "target final map",
            f"IoU {metrics['frozen']:.3f}",
            "TP / FP / FN",
            f"IoU {metrics['latest']:.3f}",
            "TP / FP / FN",
        ]
        for column, (key, title, note) in enumerate(zip(keys, titles, notes)):
            image_panel(axes[row, column], operation_panels[operation][key], title if row == 0 else "", note)
        axes[row, 0].text(
            -0.18,
            0.5,
            operation,
            transform=axes[row, 0].transAxes,
            rotation=90,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color=INK,
        )
    figure.suptitle(
        "Full-validation checkpoint audit: the latest temporal repair is over-conservative",
        x=0.075,
        y=0.975,
        ha="left",
        fontsize=8.5,
        fontweight="bold",
        color=INK,
    )
    stem = output_dir / "sn7_checkpoint_diagnostic"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)
    return [f"{stem.name}{suffix}" for suffix in (".pdf", ".svg", ".png", ".tiff")]


def render(output_dir: Path) -> dict[str, Any]:
    require_sources()
    setup_style()
    operation_panels = {operation: sn7_map_panels(case_dir) for operation, case_dir in SN7_CASES.items()}
    sn7_trace = read_trace(SN7_TRACE, TRACE_SAMPLE)
    if sn7_trace.get("split") != "val" or sn7_trace.get("test_assets_read") is not False:
        raise ValueError("SN7 mechanism case must remain validation-only")

    output_dir.mkdir(parents=True, exist_ok=True)
    assets = output_dir / "assets"
    assets.mkdir(exist_ok=True)
    for operation, panels in operation_panels.items():
        for name, image in panels.items():
            save_asset(assets / f"sn7_{operation.lower()}_{name}.png", image)

    figure = plt.figure(figsize=(7.16, 7.05), constrained_layout=False)
    figure.patch.set_facecolor("white")
    grid = figure.add_gridspec(
        4,
        5,
        height_ratios=[1.0, 1.0, 1.0, 0.86],
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0],
        left=0.075,
        right=0.985,
        top=0.935,
        bottom=0.055,
        wspace=0.12,
        hspace=0.35,
    )

    panel_label(figure, 0.028, 0.972, "a", "Operation-aligned visual proposals (SN7 validation)")
    titles = ["Current image", "Editable prior", "Reference map", "V5-B writeback", "Pixel residual"]
    keys = ["current", "prior", "reference", "proposal", "residual"]
    for row, operation in enumerate(SN7_CASES):
        metrics = SN7_CASE_METRICS[operation]
        notes = ["observation", "before write", "target final map", "frozen updater", f"TP / FP / FN | IoU {metrics['frozen']:.3f}"]
        for column, (key, title, note) in enumerate(zip(keys, titles, notes)):
            axis = figure.add_subplot(grid[row, column])
            image_panel(axis, operation_panels[operation][key], title if row == 0 else "", note)
        figure.axes[-5].text(
            -0.18,
            0.5,
            operation,
            transform=figure.axes[-5].transAxes,
            rotation=90,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color=INK,
        )

    panel_label(figure, 0.028, 0.260, "b", "Agent decision (SN7 validation: DELETE to RESHAPE)")
    anchor_axis = figure.add_subplot(grid[3, 0])
    anchor = load_rgb(SN7_VISUAL / "01_anchor_rgb.png")
    image_panel(anchor_axis, anchor, "Visual state", "direct draft: DELETE")
    candidate_panel(figure.add_subplot(grid[3, 1:3]), sn7_trace)
    belief_panel(figure.add_subplot(grid[3, 3]), sn7_trace)
    action_panel(figure.add_subplot(grid[3, 4]), sn7_trace)

    stem = output_dir / "agent_system_visual_evidence"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)
    diagnostic_outputs = render_checkpoint_diagnostic(output_dir, operation_panels)

    manifest = {
        "schema_version": "agent-system-visual-evidence-v3",
        "split": "val",
        "test_assets_read": False,
        "figure_contract": {
            "core_conclusion": "A visual map-update proposal is an input to the controller, not permission to modify the editable map.",
            "archetype": "asymmetric mixed-modality image plate plus quantitative mechanism",
            "backend": "python",
            "hero_evidence": "SN7 operation-aligned ADD/DELETE/RESHAPE proposals and pixel residuals",
            "mechanism_evidence": "SN7 candidate values, belief transition, and structured action trace",
        },
        "cases": {
            "SN7_visual_cases": {
                "ADD": "01_add_clean",
                "DELETE": "02_delete_clean",
                "RESHAPE": "03_reshape_clean",
            },
            "SN7_controller_case": TRACE_SAMPLE,
        },
        "checkpoint_decision": {
            "paper_visual_checkpoint": "frozen V5-B",
            "latest_temporal_repair": "rejected after 6,098-sample validation audit",
        },
        "visual_amplification": "none",
        "outputs": [f"{stem.name}{suffix}" for suffix in (".pdf", ".svg", ".png", ".tiff")] + diagnostic_outputs,
        "asset_directory": "assets",
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
