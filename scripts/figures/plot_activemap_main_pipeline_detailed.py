from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from PIL import Image, ImageOps


REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
ASSETS = WORKSPACE / "figure1_assets"
OUT = REPO / "docs" / "figures" / "activemap_main_pipeline_detailed_20260803"
SCIPILOT = Path.home() / ".codex" / "skills" / "scipilot-figure-skill" / "scripts"
sys.path.insert(0, str(SCIPILOT))

from export_figure import export_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402


C = {
    "ink": "#17212B",
    "muted": "#5F6B78",
    "line": "#BFC9D4",
    "panel": "#F7F9FB",
    "blue": "#0072B2",
    "blue_fill": "#E8F3F8",
    "orange": "#D98E00",
    "orange_fill": "#FFF3D2",
    "green": "#009E73",
    "green_fill": "#E4F4EE",
    "purple": "#B85C9B",
    "purple_fill": "#F7EAF2",
    "red": "#C64E13",
    "red_fill": "#FBECE5",
    "gray_fill": "#EDF1F5",
    "white": "#FFFFFF",
}


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str = C["white"],
    ec: str = C["line"],
    lw: float = 0.75,
    fs: float = 5.8,
    weight: str = "normal",
    color: str = C["ink"],
    ls: str = "-",
    radius: float = 0.65,
    align: str = "center",
    z: int = 3,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.13,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(patch)
    tx = x + w / 2 if align == "center" else x + 0.8
    ax.text(
        tx,
        y + h / 2,
        text,
        ha=align,
        va="center",
        fontsize=fs,
        weight=weight,
        color=color,
        linespacing=1.12,
        zorder=z + 1,
    )
    return patch


def arrow(
    ax,
    start,
    end,
    *,
    color=C["blue"],
    lw=1.05,
    ls="-",
    connection="arc3",
    head=7.5,
    z=6,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=head,
        color=color,
        linewidth=lw,
        linestyle=ls,
        connectionstyle=connection,
        shrinkA=1.2,
        shrinkB=1.2,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def image_tile(ax, path: Path, extent, label: str, *, ec=C["line"], fs=5.0):
    im = Image.open(path).convert("RGB")
    x0, x1, y0, y1 = extent
    ratio = (x1 - x0) / max(y1 - y0, 1e-6)
    im = ImageOps.fit(im, (700, max(1, int(700 / ratio))), Image.Resampling.LANCZOS)
    ax.imshow(im, extent=extent, zorder=2, interpolation="lanczos")
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0.0,rounding_size=0.35",
            fill=False,
            edgecolor=ec,
            linewidth=0.75,
            zorder=4,
        )
    )
    ax.text((x0 + x1) / 2, y0 - 0.55, label, ha="center", va="top", fontsize=fs)


def section(ax, y: float, h: float, label: str, subtitle: str, color: str):
    ax.add_patch(
        FancyBboxPatch(
            (0.7, y),
            98.6,
            h,
            boxstyle="round,pad=0.0,rounding_size=0.85",
            facecolor=C["panel"],
            edgecolor=C["line"],
            linewidth=0.65,
            zorder=0,
        )
    )
    ax.text(1.5, y + h - 1.35, label, fontsize=6.5, weight="bold", color=color, va="top")
    ax.text(29.0, y + h - 1.35, subtitle, fontsize=5.3, color=C["muted"], va="top")


def tag(ax, x, y, w, text, color, *, filled=False, fs=4.7):
    return box(
        ax,
        x,
        y,
        w,
        3.1,
        text,
        fc=color if filled else C["white"],
        ec=color,
        lw=0.7,
        fs=fs,
        weight="bold",
        color=C["white"] if filled else color,
        radius=1.2,
    )


def build_figure():
    setup_style(journal="ieee", lang="en", use_sciplots=False, constrained_layout=False)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    fig = plt.figure(figsize=(7.16, 6.45), facecolor="white")
    ax = fig.add_axes([0.012, 0.012, 0.976, 0.976])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(
        50,
        98.3,
        "ActiveMap: recurrent evidence control for safe, executable map updating",
        ha="center",
        va="top",
        fontsize=10.0,
        weight="bold",
        color=C["ink"],
    )
    ax.text(
        50,
        95.0,
        "A policy-relative controller acquires only decision-changing evidence, then verifies typed edits before persistent writeback.",
        ha="center",
        va="top",
        fontsize=6.2,
        color=C["muted"],
    )

    section(ax, 80.0, 12.5, "1  PERCEPTION SUBSTRATE", "replaceable visual and geospatial backends", C["blue"])
    section(ax, 43.0, 35.5, "2  RECURRENT CONTROLLER", "policy-relative evidence valuation and ordered action state machine", C["orange"])
    section(ax, 21.0, 20.0, "3  SAFETY + EXECUTION", "probabilistic risk control followed by deterministic map verification", C["green"])
    section(ax, 2.0, 17.0, "4  LEARNING SIGNALS", "executable counterfactual supervision and recurrent post-training", C["purple"])

    # 1. Perception substrate.
    image_tile(ax, ASSETS / "imagery" / "current_2019.png", (3.0, 10.0, 82.2, 89.0), "$I_t$ current image", ec=C["blue"], fs=4.5)
    image_tile(ax, ASSETS / "maps" / "prior_map_white.png", (11.0, 18.0, 82.2, 89.0), "$M_t$ editable prior", ec=C["blue"], fs=4.5)
    box(ax, 20.0, 82.0, 16.0, 7.2, "Visual/map encoder\n+ direct edit draft\n$\hat{o}_t,\; c_t,\; z_t$", fc=C["blue_fill"], ec=C["blue"], fs=5.2, weight="bold")
    ax.text(28.0, 82.45, "U-Net / ChangeMamba / SAM-Road / RemoteSAM", ha="center", fontsize=3.65, color=C["muted"])

    image_tile(ax, ASSETS / "composites" / "evidence_pool.png", (41.0, 55.0, 82.2, 89.0), "candidate evidence $\mathcal{E}_t$", ec=C["orange"], fs=4.5)
    box(ax, 57.0, 82.0, 18.0, 7.2, "Observable catalog\ntime / scale / clarity\ncost / temporal offset", fc=C["orange_fill"], ec=C["orange"], fs=5.1, weight="bold")
    box(ax, 78.0, 82.0, 18.5, 7.2, "Public belief $b_t$\n$p$(KEEP, ADD, DELETE, RESHAPE)\nconfidence / uncertainty / $\Delta g$", fc=C["purple_fill"], ec=C["purple"], fs=4.8, weight="bold")
    for x0, x1 in ((18.2, 19.7), (36.3, 40.7), (55.3, 56.7), (75.3, 77.7)):
        arrow(ax, (x0, 85.6), (x1, 85.6), color=C["blue"] if x0 < 40 else C["orange"], lw=0.9)

    # 2. Recurrent controller: structured state, evidence-value head, and ordered stages.
    box(
        ax,
        3.0,
        61.0,
        17.0,
        12.2,
        "ActiveCatalogState $s_t$\n\nvisual state $z_t$\ndirect draft $(\hat{o}_t,c_t)$\nbelief $b_t$ + budget $B_t$\nselected IDs + candidates",
        fc=C["blue_fill"],
        ec=C["blue"],
        fs=4.9,
        weight="bold",
    )
    ax.text(11.5, 59.95, "auditable, test-free public state", ha="center", fontsize=4.4, color=C["muted"])

    box(ax, 23.0, 68.0, 13.0, 6.6, "Context encoder\n$h_s=f_s(s_t)$", fc=C["white"], ec=C["blue"], fs=5.1, weight="bold")
    box(ax, 23.0, 59.2, 13.0, 6.6, "Candidate encoder\n$h_e=f_e(e_i,c_i)$", fc=C["white"], ec=C["orange"], fs=5.1, weight="bold")
    box(ax, 39.0, 62.0, 14.0, 9.4, "State-candidate interaction\n$[h_s,\;h_e,\;h_s\odot h_e]$\n\npolicy-relative representation", fc=C["orange_fill"], ec=C["orange"], fs=5.0, weight="bold")
    arrow(ax, (20.3, 68.7), (22.7, 71.1), color=C["blue"])
    arrow(ax, (20.3, 64.0), (22.7, 62.5), color=C["orange"])
    arrow(ax, (36.3, 71.0), (38.7, 68.5), color=C["blue"])
    arrow(ax, (36.3, 62.5), (38.7, 65.3), color=C["orange"])

    box(ax, 56.0, 66.2, 14.5, 8.4, "Multi-task action heads\n$\Delta U_i$  |  $\Delta Q_i$  |  beneficial\nunsafe  |  missed  |  terminal edit", fc=C["white"], ec=C["orange"], fs=4.75, weight="bold")
    box(ax, 56.0, 55.8, 14.5, 8.0, "Risk-adjusted score\n$V_i=\Delta U_i-\lambda_f p_i^{unsafe}$\n$\quad-\lambda_m p_i^{missed}$", fc=C["red_fill"], ec=C["red"], fs=4.65, weight="bold")
    arrow(ax, (53.3, 66.7), (55.7, 70.2), color=C["orange"])
    arrow(ax, (63.2, 65.9), (63.2, 64.1), color=C["red"])

    box(ax, 73.5, 61.0, 23.0, 13.6, "", fc=C["gray_fill"], ec=C["ink"], fs=6.0, weight="bold")
    ax.text(85.0, 73.2, "Ordered controller", ha="center", fontsize=5.6, weight="bold", color=C["ink"])
    tag(ax, 75.0, 68.5, 5.0, "DRAFT", C["blue"], fs=4.2)
    tag(ax, 81.0, 68.5, 5.0, "SELECT", C["orange"], filled=True, fs=4.2)
    tag(ax, 87.0, 68.5, 4.0, "TOOL", C["purple"], fs=4.1)
    tag(ax, 92.0, 68.5, 3.4, "TERM", C["green"], fs=4.0)
    arrow(ax, (80.2, 70.05), (80.7, 70.05), color=C["ink"], head=5)
    arrow(ax, (86.2, 70.05), (86.7, 70.05), color=C["ink"], head=5)
    arrow(ax, (91.2, 70.05), (91.7, 70.05), color=C["ink"], head=5)
    ax.text(85.0, 65.9, "$\\max_i V_i > \\tau_{safe}$ ?", ha="center", fontsize=5.2, weight="bold", color=C["red"])
    tag(ax, 75.5, 62.3, 8.5, "ACQUIRE($e_i$)", C["orange"], fs=4.4)
    tag(ax, 85.0, 62.3, 4.8, "STOP", C["blue"], fs=4.4)
    tag(ax, 90.7, 62.3, 4.5, "REPLAN", C["purple"], fs=4.1)

    # Environment transitions, optional tool path, and action semantics.
    box(ax, 3.0, 45.8, 24.5, 9.3, "Budgeted evidence environment\nACQUIRE: pay $c_i$ + add evidence + fuse mask\n$B_{t+1}=B_t-c_i$; update candidates\nreward = marginal utility; emit $s_{t+1}$", fc=C["orange_fill"], ec=C["orange"], fs=4.65, weight="bold")
    box(ax, 30.0, 45.8, 25.0, 9.3, "Optional grounded tool branch\nUSE_TOOL(tool, acquired $e_i$)\ncrop / quality / temporal / segmentation / topology\nstable result features + cost + provenance", fc=C["white"], ec=C["purple"], ls="--", fs=4.55, weight="bold")
    box(ax, 58.0, 45.8, 20.0, 9.3, "Reliability-gated residual belief\n$r=\sigma(f_{gate})$\n$b_{t+1}=b_t+r\,\Delta b$\nedit prob. / confidence / geometry", fc=C["purple_fill"], ec=C["purple"], ls="--", fs=4.65, weight="bold")
    box(ax, 81.0, 45.8, 15.5, 9.3, "Terminal semantics\nREJECT $\\Rightarrow$ KEEP\nCOMMIT($o$), $o\\ne$ KEEP\n$o\\in\\{$ADD, DELETE,\nRESHAPE$\\}$", fc=C["green_fill"], ec=C["green"], fs=4.6, weight="bold")
    arrow(ax, (78.2, 50.45), (80.7, 50.45), color=C["green"])
    arrow(ax, (27.8, 50.45), (29.7, 50.45), color=C["purple"], ls="--")
    arrow(ax, (55.3, 50.45), (57.7, 50.45), color=C["purple"], ls="--")
    ax.plot([78.2, 79.5, 79.5, 93.0], [50.45, 50.45, 58.0, 58.0], color=C["purple"], lw=1.0, ls="--", zorder=5)
    arrow(ax, (93.0, 58.0), (93.0, 62.0), color=C["purple"], ls="--")
    ax.text(85.8, 58.35, "update belief, then replan", ha="center", fontsize=4.05, color=C["purple"])
    ax.plot([79.2, 79.2, 15.2], [62.1, 55.45, 55.45], color=C["orange"], lw=1.0, zorder=5)
    arrow(ax, (15.2, 55.45), (15.2, 55.2), color=C["orange"], lw=1.0)
    ax.text(42.0, 55.75, "selected evidence", ha="center", fontsize=4.05, color=C["orange"])

    # 3. Safety and executable map transaction.
    box(ax, 3.0, 25.6, 15.5, 10.1, "Probabilistic Safe Commit\nvalidation-calibrated threshold\nfalse-edit cap + KEEP guard\nlow confidence $\\Rightarrow$ REJECT", fc=C["green_fill"], ec=C["green"], fs=4.75, weight="bold")
    box(ax, 21.0, 25.6, 20.0, 10.1, "Typed delta writeback\nfuse selected masks by confidence\nADD: $M\cup\Delta^+$  |  DELETE: $M\setminus\Delta^-$\nRESHAPE: add + remove; filter tiny components", fc=C["white"], ec=C["green"], fs=4.45, weight="bold")
    image_tile(ax, ASSETS / "edits" / "candidate_edit_demo.png", (43.5, 52.0, 27.0, 34.3), "candidate vector delta", ec=C["purple"], fs=4.4)
    box(ax, 54.0, 25.6, 18.5, 10.1, "Deterministic transaction verifier\nconfidence / object ID / geometry family\nvalidity / topology conflict\nAPPROVE  |  REVISE  |  REJECT", fc=C["white"], ec=C["green"], fs=4.45, weight="bold")
    image_tile(ax, ASSETS / "edits" / "committed_edit_demo.png", (75.0, 83.5, 27.0, 34.3), "committed map", ec=C["green"], fs=4.4)
    box(ax, 86.0, 25.6, 10.5, 10.1, "Persistent state\ncommit / rollback\naudit + provenance\n$M_t\\rightarrow M_{t+1}$", fc=C["blue_fill"], ec=C["blue"], fs=4.55, weight="bold")
    for a, b in (((18.8, 30.7), (20.7, 30.7)), ((41.3, 30.7), (43.2, 30.7)), ((52.3, 30.7), (53.7, 30.7)), ((72.8, 30.7), (74.7, 30.7)), ((83.8, 30.7), (85.7, 30.7))):
        arrow(ax, a, b, color=C["green"], lw=1.0)
    ax.text(50, 22.75, "Executable metrics: raster / polygon IoU, added-removed change IoU, vector replay, topology validity, false/missed/wrong edit, cost", ha="center", fontsize=4.7, color=C["muted"])
    ax.plot([91.2, 98.0, 98.0], [35.9, 39.0, 85.6], color=C["blue"], lw=1.15, zorder=5)
    arrow(ax, (98.0, 85.6), (96.7, 85.6), color=C["blue"], lw=1.15)
    ax.text(99.0, 59.5, "continual loop: updated map becomes next-time prior", ha="center", fontsize=4.25, weight="bold", color=C["blue"], rotation=90)

    # 4. Learning signals and claim boundary.
    box(ax, 3.0, 5.1, 18.0, 9.4, "Audited training data\ntrain/val only; no test assets\nexecutable counterfactual outcomes\nterminal replay + rollout traces", fc=C["gray_fill"], ec=C["ink"], fs=4.55, weight="bold")
    box(ax, 24.0, 5.1, 24.5, 9.4, "Structured cold-start pretraining\n$L=L_{utility}+L_{quality}+L_{beneficial}$\n$\\quad+L_{unsafe}+L_{missed}+L_{terminal}$\nbalanced ACQUIRE/STOP + calibrated $\\tau_{safe}$", fc=C["orange_fill"], ec=C["orange"], fs=4.45, weight="bold")
    box(ax, 51.5, 5.1, 19.0, 9.4, "Implemented post-training\nSFT-anchored proxy GRPO\ngroup-relative advantages\nsequence clipping + KL + replay", fc=C["purple_fill"], ec=C["purple"], fs=4.55, weight="bold")
    box(ax, 73.5, 5.1, 23.0, 9.4, "Final paper-primary protocol\nexecutable-map constrained GRPO\n$R=\Delta Q_{map}+\lambda_t\Delta Q_{topo}$\n$-\lambda_fE_f-\lambda_mE_m-\lambda_c C-\lambda_iE_i$", fc=C["white"], ec=C["purple"], ls="--", fs=4.35, weight="bold")
    arrow(ax, (21.3, 9.8), (23.7, 9.8), color=C["orange"])
    arrow(ax, (48.8, 9.8), (51.2, 9.8), color=C["purple"])
    arrow(ax, (70.8, 9.8), (73.2, 9.8), color=C["purple"], ls="--")
    ax.text(85.0, 3.65, "dashed = target protocol; current proxy GRPO is not presented as final executable RL", ha="center", fontsize=4.25, color=C["muted"])

    # Compact visual grammar legend.
    ax.plot([3.0, 7.0], [1.0, 1.0], color=C["blue"], lw=1.4)
    ax.text(7.7, 1.0, "main recurrent path", va="center", fontsize=4.2, color=C["muted"])
    ax.plot([21.5, 25.5], [1.0, 1.0], color=C["purple"], lw=1.2, ls="--")
    ax.text(26.2, 1.0, "optional tool / target protocol", va="center", fontsize=4.2, color=C["muted"])
    ax.plot([46.0, 50.0], [1.0, 1.0], color=C["green"], lw=1.4)
    ax.text(50.7, 1.0, "safe executable path", va="center", fontsize=4.2, color=C["muted"])

    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    preview = OUT / "activemap_main_pipeline_detailed_preview.png"
    render_preview(fig, preview, dpi=180)
    report = audit_layout(fig)
    print_report(report)
    export_figure(
        fig,
        basename=OUT / "activemap_main_pipeline_detailed_en",
        formats=["pdf", "svg", "png"],
        size_inches=(7.16, 6.45),
        dpi=300,
        grayscale_preview=True,
    )
    metadata = {
        "figure": "ActiveMap detailed code-grounded method overview",
        "claim": "Policy-relative evidence control plus two-stage safe executable writeback.",
        "code_grounding": [
            "agent/active_catalog.py",
            "agent/evidence_value_head.py",
            "agent/sequential_controller.py",
            "agent/environment.py",
            "agent/tool_belief_model.py",
            "safe_commit.py",
            "agent/writeback.py",
            "agent/map_transaction.py",
            "agent/recurrent_grpo.py",
        ],
        "claim_boundary": "Proxy GRPO is implemented; executable-map GRPO is shown as the final target protocol.",
        "visual_qa": report,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
