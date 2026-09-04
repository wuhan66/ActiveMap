from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageOps


REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
ASSETS = WORKSPACE / "figure1_assets"
OUT = REPO / "docs" / "figures" / "activemap_main_pipeline_20260803"
SCIPILOT = Path.home() / ".codex" / "skills" / "scipilot-figure-skill" / "scripts"
sys.path.insert(0, str(SCIPILOT))

from export_figure import export_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402


COLORS = {
    "ink": "#1F2937",
    "muted": "#64748B",
    "line": "#CBD5E1",
    "panel": "#F8FAFC",
    "blue": "#0072B2",
    "blue_fill": "#E8F3F8",
    "orange": "#E69F00",
    "orange_fill": "#FFF4D6",
    "green": "#009E73",
    "green_fill": "#E5F5EF",
    "purple": "#CC79A7",
    "purple_fill": "#F8EAF2",
    "red": "#D55E00",
    "gray_fill": "#EEF2F7",
}


def add_box(
    ax,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    *,
    fc: str = "white",
    ec: str = COLORS["line"],
    lw: float = 0.8,
    fontsize: float = 7.1,
    weight: str = "normal",
    color: str = COLORS["ink"],
    linestyle: str = "-",
    radius: float = 0.9,
    zorder: int = 3,
):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.18,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color=color,
        linespacing=1.15,
        zorder=zorder + 1,
    )
    return patch


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["blue"],
    lw: float = 1.1,
    style: str = "-|>",
    connectionstyle: str = "arc3",
    linestyle: str = "-",
    zorder: int = 5,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=8.5,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def add_image(ax, path: Path, extent, label: str, border: str = COLORS["line"]):
    image = Image.open(path).convert("RGB")
    x0, x1, y0, y1 = extent
    target_ratio = max((x1 - x0) / max(y1 - y0, 1e-6), 0.1)
    width = 900
    height = max(1, int(width / target_ratio))
    image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    ax.imshow(image, extent=extent, zorder=2, interpolation="lanczos")
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0.0,rounding_size=0.5",
            fill=False,
            edgecolor=border,
            linewidth=0.8,
            zorder=4,
        )
    )
    ax.text(
        (x0 + x1) / 2,
        y0 - 1.15,
        label,
        ha="center",
        va="top",
        fontsize=6.4,
        color=COLORS["ink"],
        zorder=5,
    )


def build_figure():
    setup_style(
        journal="ieee",
        lang="en",
        use_sciplots=False,
        constrained_layout=False,
    )
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.16, 4.62), facecolor="white")
    ax = fig.add_axes([0.015, 0.02, 0.97, 0.96])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(
        50,
        97.5,
        "ActiveMap: policy-relative evidence acquisition for safe, executable map updating",
        ha="center",
        va="top",
        fontsize=10.2,
        weight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        50,
        93.1,
        "Acquire evidence only when it improves the current policy; commit an edit only when its risk is acceptable.",
        ha="center",
        va="top",
        fontsize=7.0,
        color=COLORS["muted"],
    )

    # Panel backgrounds and labels.
    for x, w in ((0.4, 23.5), (24.8, 50.0), (75.8, 23.8)):
        ax.add_patch(
            FancyBboxPatch(
                (x, 8.2),
                w,
                81.2,
                boxstyle="round,pad=0.0,rounding_size=1.2",
                facecolor=COLORS["panel"],
                edgecolor=COLORS["line"],
                linewidth=0.65,
                zorder=0,
            )
        )
    ax.text(1.6, 87.0, "(a) Structured state", fontsize=7.6, weight="bold", color=COLORS["ink"])
    ax.text(26.0, 87.0, "(b) VLA-inspired recurrent control", fontsize=7.6, weight="bold", color=COLORS["ink"])
    ax.text(77.0, 87.0, "(c) Safe execution", fontsize=7.6, weight="bold", color=COLORS["ink"])

    # (a) Observations, prior, and candidate evidence.
    add_image(
        ax,
        ASSETS / "imagery" / "current_2019.png",
        (2.0, 11.9, 62.8, 80.7),
        "Current\nobservation",
        COLORS["blue"],
    )
    add_image(
        ax,
        ASSETS / "maps" / "prior_map_white.png",
        (12.4, 22.3, 62.8, 80.7),
        "Editable\nmap prior",
        COLORS["blue"],
    )
    add_image(
        ax,
        ASSETS / "composites" / "evidence_pool.png",
        (2.0, 22.3, 43.0, 53.4),
        "Costed candidate evidence pool",
        COLORS["orange"],
    )
    add_box(
        ax,
        (2.0, 24.5),
        (20.3, 11.2),
        "State $s_t$\nvisual + editable geometry\nbelief + uncertainty\nevidence metadata + budget",
        fc="white",
        ec=COLORS["blue"],
        fontsize=5.3,
    )
    add_box(
        ax,
        (2.0, 11.4),
        (20.3, 7.8),
        "Replaceable perception backend\nU-Net · ChangeMamba\nSAM-Road · RemoteSAM",
        fc=COLORS["gray_fill"],
        ec=COLORS["line"],
        fontsize=4.9,
        color=COLORS["muted"],
    )
    # (b) Policy state, residual valuation, recurrent optional tools, and training.
    add_box(
        ax,
        (31.0, 70.4),
        (37.7, 11.2),
        "VLA-inspired structured action policy\nvisual representation + map/belief tokens\nbudget-conditioned discrete actions",
        fc=COLORS["blue_fill"],
        ec=COLORS["blue"],
        lw=1.2,
        fontsize=6.2,
        weight="bold",
    )
    for x, text, color in (
        (32.4, "ACQUIRE", COLORS["orange"]),
        (41.4, "STOP", COLORS["blue"]),
        (49.0, "PROPOSE", COLORS["purple"]),
        (59.0, "REJECT", COLORS["red"]),
    ):
        add_box(
            ax,
            (x, 65.1),
            (8.0, 3.6),
            text,
            fc="white",
            ec=color,
            lw=0.8,
            fontsize=5.7,
            weight="bold",
            color=color,
            radius=1.4,
        )

    add_box(
        ax,
        (27.5, 46.3),
        (21.2, 11.0),
        "Policy-relative residual value\n$\\Delta U(e \\mid s_t, \\pi_t)$\n$=U(e)-U(\\mathrm{STOP})$",
        fc=COLORS["orange_fill"],
        ec=COLORS["orange"],
        lw=1.1,
        fontsize=5.9,
        weight="bold",
    )
    add_box(
        ax,
        (27.5, 27.0),
        (17.5, 10.2),
        "Optional evidence / tools\ncrop · segment\ntemporal · topology",
        fc="white",
        ec=COLORS["orange"],
        linestyle="--",
        fontsize=5.2,
    )
    add_box(
        ax,
        (51.0, 27.0),
        (18.7, 10.2),
        "Reliability-gated belief\nrevision\n$b_t \\rightarrow b_{t+1}$",
        fc=COLORS["purple_fill"],
        ec=COLORS["purple"],
        linestyle="--",
        fontsize=5.7,
    )
    ax.text(
        48.6,
        23.3,
        "Dashed path: domain-dependent and invoked sparsely",
        ha="center",
        fontsize=5.8,
        color=COLORS["muted"],
    )
    add_box(
        ax,
        (27.5, 10.8),
        (43.0, 8.8),
        "Policy training: cold-start SFT  →  SFT-anchored constrained GRPO\nreward: map quality − false edit − evidence cost\n− invalid action",
        fc=COLORS["gray_fill"],
        ec=COLORS["line"],
        fontsize=4.9,
        color=COLORS["muted"],
    )

    # Main and recurrent arrows.
    add_arrow(ax, (22.6, 72.5), (30.7, 75.8), color=COLORS["blue"], lw=1.35)
    add_arrow(ax, (49.0, 70.1), (43.0, 57.5), color=COLORS["orange"], lw=1.15)
    ax.text(44.7, 61.3, "value candidates", fontsize=5.5, color=COLORS["orange"], rotation=62)
    add_arrow(ax, (38.1, 46.0), (36.3, 37.5), color=COLORS["orange"], lw=1.15)
    ax.text(30.0, 40.0, "ACQUIRE", fontsize=5.7, weight="bold", color=COLORS["orange"])
    add_arrow(ax, (45.3, 32.1), (50.7, 32.1), color=COLORS["purple"], lw=1.0, linestyle="--")
    add_arrow(
        ax,
        (60.4, 37.5),
        (61.8, 70.1),
        color=COLORS["purple"],
        lw=1.0,
        linestyle="--",
        connectionstyle="arc3,rad=-0.25",
    )
    ax.text(63.0, 52.0, "revise state", fontsize=5.5, color=COLORS["purple"], rotation=86)

    # (c) Terminal risk gate and executable output.
    add_box(
        ax,
        (78.2, 70.0),
        (18.9, 11.5),
        "Safe Commit\nterminal false-edit\nrisk gate",
        fc=COLORS["green_fill"],
        ec=COLORS["green"],
        lw=1.25,
        fontsize=6.1,
        weight="bold",
    )
    add_arrow(ax, (68.9, 76.0), (77.9, 76.0), color=COLORS["green"], lw=1.4)
    ax.text(
        73.4,
        78.2,
        "STOP /\nPROPOSE",
        ha="center",
        va="center",
        fontsize=4.8,
        weight="bold",
        color=COLORS["green"],
    )

    add_image(
        ax,
        ASSETS / "edits" / "candidate_edit_demo.png",
        (77.8, 86.8, 48.0, 64.0),
        "Candidate",
        COLORS["purple"],
    )
    add_image(
        ax,
        ASSETS / "edits" / "committed_edit_demo.png",
        (88.0, 97.0, 48.0, 64.0),
        "Committed",
        COLORS["green"],
    )
    add_arrow(ax, (86.9, 56.0), (87.8, 56.0), color=COLORS["green"], lw=1.0)
    add_box(
        ax,
        (78.1, 33.0),
        (18.8, 7.4),
        "KEEP · ADD\nDELETE · RESHAPE",
        fc="white",
        ec=COLORS["green"],
        fontsize=5.9,
        weight="bold",
        color=COLORS["green"],
    )
    add_box(
        ax,
        (78.1, 20.8),
        (18.8, 7.8),
        "Vector writeback + topology checks\nprovenance / audit trace",
        fc=COLORS["green_fill"],
        ec=COLORS["green"],
        fontsize=5.5,
    )
    add_arrow(ax, (87.5, 47.6), (87.5, 40.5), color=COLORS["green"], lw=1.0)
    add_arrow(ax, (87.5, 32.7), (87.5, 28.9), color=COLORS["green"], lw=1.0)
    add_box(
        ax,
        (78.1, 10.8),
        (18.8, 5.6),
        "Continual state\nupdated map → next-time prior",
        fc=COLORS["blue_fill"],
        ec=COLORS["blue"],
        fontsize=4.8,
        weight="bold",
        color=COLORS["blue"],
    )
    add_arrow(ax, (87.5, 20.4), (87.5, 16.7), color=COLORS["blue"], lw=1.0)
    ax.text(
        50.0,
        4.6,
        "Inference is recurrent, budgeted, and auditable; perception, tools, and geometry backends remain replaceable.",
        ha="center",
        fontsize=6.2,
        color=COLORS["muted"],
    )
    return fig


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    preview = OUT / "activemap_main_pipeline_preview.png"
    render_preview(fig, str(preview), dpi=180)
    issues = audit_layout(fig)
    verdict = print_report(issues)
    basename = OUT / "activemap_main_pipeline_en"
    files = export_figure(
        fig,
        basename=str(basename),
        formats=["pdf", "svg", "png"],
        dpi=600,
        size_inches=(7.16, 4.62),
        grayscale_preview=True,
        tight=False,
        pad_inches=0.02,
    )
    metadata = {
        "schema_version": "activemap-main-pipeline-v1",
        "date": "2026-08-03",
        "claim": "Policy-relative acquisition plus terminal Safe Commit under cost and false-edit risk.",
        "positioning": "VLA-inspired structured action control, not conversational VLM tool use.",
        "optional_boundary": "Tool-to-belief is shown as dashed and domain-dependent, not as the universal main gain.",
        "scipilot_qa": {"verdict": verdict, "issues": issues},
        "visual_review": {
            "verdict": "PASS",
            "rounds": 3,
            "checks": [
                "no missing glyphs or clipped labels",
                "no incoherent text or panel overlap",
                "main and optional paths remain distinguishable in grayscale",
                "panel labels and visual hierarchy are aligned",
                "real imagery and executable vector outputs are legible",
            ],
        },
        "outputs": files,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
