#!/usr/bin/env python3
"""Create the ActiveMap literature-derived problem-origin figure."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image, ImageDraw


COLORS = {
    "ink": "#172033",
    "muted": "#596579",
    "line": "#C9D2DF",
    "blue": "#2864B4",
    "teal": "#058C8C",
    "green": "#25804A",
    "amber": "#B96B12",
    "red": "#B63C3C",
    "blue_bg": "#EEF5FC",
    "teal_bg": "#ECF8F7",
    "green_bg": "#EFF8F1",
    "amber_bg": "#FFF6E8",
    "paper": "#FFFFFF",
}


TEXT = {
    "en": {
        "title": "ActiveMap arises at the intersection of established research lines",
        "subtitle": "The task is literature-derived; policy-relative acquisition and Safe Commit are the proposed solution.",
        "l1": "Editable Map Update",
        "l1ref": "MUNO21 (ICCV'21) · ArgoTweak (ICCV'25)",
        "l1q": "Given observations, how should an old map be updated?",
        "l2": "Selective Map Construction",
        "l2ref": "OptiMVMap (CVPR'26)",
        "l2q": "Which costly, redundant or noisy views should be fused?",
        "l3": "Active Visual Search",
        "l3ref": "MapEx · SenseSearch · GeoMMAgent",
        "l3q": "What should the visual system inspect next?",
        "gap": "Missing intersection",
        "gaptext": "Sequential evidence choice + stopping + risk-aware executable map writeback",
        "ours": "ActiveMap: Budgeted Active Evidence Acquisition for Editable Map Updating",
        "catalog": "Candidate evidence",
        "residual": "Current-policy\nresidual value",
        "decision": "ACQUIRE / STOP",
        "safe": "Safe Commit",
        "write": "Executable edit",
        "actions": "KEEP · ADD\nDELETE · RESHAPE",
        "cost": "cost / redundancy / noise",
        "risk": "false-edit risk",
    },
    "zh": {
        "title": "ActiveMap 来自三条既有研究线的交叉缺口",
        "subtitle": "问题由文献与真实约束推出；策略相对取证和 Safe Commit 是我们的解法。",
        "l1": "可编辑地图更新",
        "l1ref": "MUNO21（ICCV'21）· ArgoTweak（ICCV'25）",
        "l1q": "观测已给定时，如何增量更新旧地图？",
        "l2": "选择式地图构建",
        "l2ref": "OptiMVMap（CVPR'26）",
        "l2q": "哪些昂贵、冗余或有噪声的视角值得融合？",
        "l3": "主动视觉搜索",
        "l3ref": "MapEx · SenseSearch · GeoMMAgent",
        "l3q": "视觉系统下一步应该查看什么？",
        "gap": "尚未联合解决的交叉缺口",
        "gaptext": "连续证据选择 + 停止决策 + 风险感知的可执行地图写回",
        "ours": "ActiveMap：预算约束下的可编辑地图主动证据获取",
        "catalog": "候选证据",
        "residual": "当前策略条件下的\n剩余证据价值",
        "decision": "获取 / 停止",
        "safe": "安全提交",
        "write": "可执行编辑",
        "actions": "保持 · 新增 · 删除 · 重塑",
        "cost": "成本 / 冗余 / 噪声",
        "risk": "错误编辑风险",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--anchor",
        type=Path,
        default=Path(".codex-results/rollout_visual_qc/01_anchor_rgb.png"),
    )
    parser.add_argument(
        "--utility",
        type=Path,
        default=Path(".codex-results/rollout_visual_qc/utility_overlay_step00.png"),
    )
    return parser.parse_args()


def configure_fonts() -> dict[str, font_manager.FontProperties]:
    candidates = {
        "en": Path("C:/Windows/Fonts/arial.ttf"),
        "en_bold": Path("C:/Windows/Fonts/arialbd.ttf"),
        "zh": Path("C:/Windows/Fonts/msyh.ttc"),
        "zh_bold": Path("C:/Windows/Fonts/msyhbd.ttc"),
    }
    return {
        key: font_manager.FontProperties(fname=str(path))
        for key, path in candidates.items()
        if path.is_file()
    }


def crop_square(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return np.asarray(image.crop((left, top, left + side, top + side)))


def map_symbol(size: int = 512) -> Image.Image:
    image = Image.new("RGBA", (size, size), (246, 248, 251, 255))
    draw = ImageDraw.Draw(image)
    roads = [
        [(30, 400), (160, 310), (275, 325), (470, 150)],
        [(65, 90), (180, 205), (300, 215), (455, 350)],
        [(250, 25), (255, 165), (275, 325), (305, 490)],
    ]
    for points in roads:
        draw.line(points, fill=(255, 255, 255, 255), width=34, joint="curve")
        draw.line(points, fill=(79, 101, 129, 255), width=9, joint="curve")
    buildings = [
        (70, 235, 135, 295),
        (340, 65, 420, 130),
        (330, 315, 430, 405),
        (115, 365, 180, 435),
    ]
    for index, box in enumerate(buildings):
        fill = (37, 128, 74, 85) if index != 2 else (182, 60, 60, 72)
        outline = (37, 128, 74, 255) if index != 2 else (182, 60, 60, 255)
        draw.rectangle(box, fill=fill, outline=outline, width=7)
    draw.rectangle((360, 345, 460, 435), outline=(182, 60, 60, 255), width=7)
    draw.line((360, 345, 460, 435), fill=(182, 60, 60, 255), width=7)
    draw.line((460, 345, 360, 435), fill=(182, 60, 60, 255), width=7)
    return image


def add_image(ax: plt.Axes, image: np.ndarray, box: tuple[float, float, float, float], border: str) -> None:
    x, y, width, height = box
    ax.imshow(image, extent=(x, x + width, y, y + height), zorder=2, aspect="auto")
    ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=border, linewidth=1.4, zorder=3))


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#7B8798") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            color=color,
            shrinkA=2,
            shrinkB=2,
            zorder=5,
        )
    )


def label(ax: plt.Axes, x: float, y: float, text: str, fp: font_manager.FontProperties, size: float, color: str, **kwargs: object) -> None:
    ax.text(x, y, text, fontproperties=fp, fontsize=size, color=color, **kwargs)


def panel(ax: plt.Axes, x: float, title: str, ref: str, question: str, color: str, bg: str, fp: dict[str, font_manager.FontProperties], lang: str, anchor: np.ndarray, utility: np.ndarray, map_img: np.ndarray, kind: int) -> None:
    width = 30.1
    ax.add_patch(Rectangle((x, 45.5), width, 36.5, facecolor=bg, edgecolor=COLORS["line"], linewidth=1.1))
    ax.add_patch(Rectangle((x, 78.9), width, 3.1, facecolor=color, edgecolor="none"))
    bold = fp[f"{lang}_bold"]
    regular = fp[lang]
    label(ax, x + 2.0, 75.7, title, bold, 13, COLORS["ink"], ha="left", va="top")
    label(ax, x + 2.0, 71.8, ref, regular, 8.4, color, ha="left", va="top")

    if kind == 0:
        add_image(ax, map_img, (x + 2.0, 53.2, 8.0, 11.2), COLORS["blue"])
        add_image(ax, anchor, (x + 11.2, 53.2, 8.0, 11.2), COLORS["blue"])
        add_image(ax, utility, (x + 20.4, 53.2, 7.7, 11.2), COLORS["green"])
        arrow(ax, (x + 10.2, 58.8), (x + 11.0, 58.8), COLORS["blue"])
        arrow(ax, (x + 19.4, 58.8), (x + 20.2, 58.8), COLORS["green"])
    elif kind == 1:
        crops = [anchor[0:380, 0:380], anchor[0:380, -380:], anchor[-380:, 0:380], anchor[-380:, -380:]]
        positions = [(x + 2.0, 58.9), (x + 7.8, 58.9), (x + 2.0, 52.8), (x + 7.8, 52.8)]
        for idx, (crop, pos) in enumerate(zip(crops, positions, strict=True)):
            add_image(ax, crop, (pos[0], pos[1], 5.0, 5.0), COLORS["teal"] if idx in {0, 3} else COLORS["line"])
        arrow(ax, (x + 13.2, 58.2), (x + 17.2, 58.2), COLORS["teal"])
        add_image(ax, utility, (x + 18.0, 52.8, 10.1, 11.1), COLORS["teal"])
    else:
        add_image(ax, anchor, (x + 2.0, 52.8, 11.2, 11.2), COLORS["amber"])
        ax.add_patch(Rectangle((x + 6.0, 56.1), 4.0, 4.0, fill=False, edgecolor=COLORS["red"], linewidth=2.3, zorder=5))
        arrow(ax, (x + 13.5, 58.3), (x + 17.0, 58.3), COLORS["amber"])
        add_image(ax, anchor[180:570, 180:570], (x + 17.8, 54.0, 7.7, 8.0), COLORS["amber"])
        arrow(ax, (x + 25.7, 58.0), (x + 27.4, 58.0), COLORS["green"])
        ax.add_patch(Rectangle((x + 27.6, 55.6), 1.7, 4.8, facecolor=COLORS["green"], edgecolor="none", zorder=4))

    label(ax, x + width / 2, 48.8, question, regular, 9.3, COLORS["muted"], ha="center", va="center", wrap=True)


def render(lang: str, output: Path, anchor: np.ndarray, utility: np.ndarray, fp: dict[str, font_manager.FontProperties]) -> None:
    text = TEXT[lang]
    regular = fp[lang]
    bold = fp[f"{lang}_bold"]
    map_img = np.asarray(map_symbol())

    fig = plt.figure(figsize=(16, 9), facecolor=COLORS["paper"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    label(ax, 4, 95.0, text["title"], bold, 21, COLORS["ink"], ha="left", va="top")
    label(ax, 4, 89.7, text["subtitle"], regular, 10.5, COLORS["muted"], ha="left", va="top")

    panel(ax, 3.5, text["l1"], text["l1ref"], text["l1q"], COLORS["blue"], COLORS["blue_bg"], fp, lang, anchor, utility, map_img, 0)
    panel(ax, 35.0, text["l2"], text["l2ref"], text["l2q"], COLORS["teal"], COLORS["teal_bg"], fp, lang, anchor, utility, map_img, 1)
    panel(ax, 66.5, text["l3"], text["l3ref"], text["l3q"], COLORS["amber"], COLORS["amber_bg"], fp, lang, anchor, utility, map_img, 2)

    ax.add_patch(Rectangle((7.0, 37.3), 86.0, 5.4, facecolor="#FFF1F1", edgecolor=COLORS["red"], linewidth=1.2))
    label(ax, 10.0, 40.0, text["gap"], bold, 11, COLORS["red"], ha="left", va="center")
    label(ax, 33.2, 40.0, text["gaptext"], regular, 10.5, COLORS["ink"], ha="left", va="center")
    for x in (18.5, 50.0, 81.5):
        arrow(ax, (x, 45.2), (x, 42.9), COLORS["red"])
    arrow(ax, (50.0, 37.1), (50.0, 33.6), COLORS["red"])

    ax.add_patch(Rectangle((3.5, 4.2), 93.0, 28.8, facecolor="#F7FAFD", edgecolor=COLORS["line"], linewidth=1.2))
    ax.add_patch(Rectangle((3.5, 29.8), 93.0, 3.2, facecolor=COLORS["ink"], edgecolor="none"))
    label(ax, 5.5, 31.3, text["ours"], bold, 13, "white", ha="left", va="center")

    boxes = [(6.0, 9.2, 14.0, 14.2), (26.0, 9.2, 14.0, 14.2), (46.0, 9.2, 12.5, 14.2), (65.0, 9.2, 12.5, 14.2), (83.0, 9.2, 10.5, 14.2)]
    add_image(ax, anchor, boxes[0], COLORS["blue"])
    add_image(ax, utility, boxes[1], COLORS["teal"])
    for box, color in zip(boxes[2:], (COLORS["teal"], COLORS["red"], COLORS["green"]), strict=True):
        ax.add_patch(Rectangle((box[0], box[1]), box[2], box[3], facecolor="white", edgecolor=color, linewidth=1.6))

    label(ax, 13.0, 7.1, text["catalog"], bold, 9.5, COLORS["blue"], ha="center", va="center")
    label(ax, 33.0, 7.1, text["residual"], bold, 9.2, COLORS["teal"], ha="center", va="center")
    label(ax, 52.25, 16.3, text["decision"], bold, 10.8, COLORS["teal"], ha="center", va="center")
    label(ax, 71.25, 16.3, text["safe"], bold, 10.8, COLORS["red"], ha="center", va="center")
    label(ax, 88.25, 16.3, text["write"], bold, 10.8, COLORS["green"], ha="center", va="center")
    label(ax, 88.25, 12.4, text["actions"], regular, 7.0, COLORS["muted"], ha="center", va="center")
    label(ax, 23.0, 25.9, text["cost"], regular, 8.8, COLORS["amber"], ha="center", va="center")
    label(ax, 74.0, 25.9, text["risk"], regular, 8.8, COLORS["red"], ha="center", va="center")

    for start, end in [((20.3, 16.3), (25.7, 16.3)), ((40.3, 16.3), (45.7, 16.3)), ((58.8, 16.3), (64.7, 16.3)), ((77.8, 16.3), (82.7, 16.3))]:
        arrow(ax, start, end)
    ax.add_patch(FancyArrowPatch((52.0, 9.0), (32.8, 8.8), connectionstyle="arc3,rad=-0.28", arrowstyle="-|>", mutation_scale=13, linewidth=1.35, color=COLORS["teal"]))

    fig.savefig(output.with_suffix(".png"), dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output.with_suffix(".svg"), facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchor = crop_square(args.anchor)
    utility = crop_square(args.utility)
    fonts = configure_fonts()
    required = {"en", "en_bold", "zh", "zh_bold"}
    if required - set(fonts):
        raise FileNotFoundError(f"missing fonts: {sorted(required - set(fonts))}")

    render("en", args.output_dir / "activemap_problem_origin_en", anchor, utility, fonts)
    render("zh", args.output_dir / "activemap_problem_origin_zh", anchor, utility, fonts)

    assets = args.output_dir / "assets"
    assets.mkdir(exist_ok=True)
    shutil.copy2(args.anchor, assets / "anchor_rgb.png")
    shutil.copy2(args.utility, assets / "policy_utility_overlay.png")
    map_symbol().save(assets / "editable_map_symbol.png")
    metadata = {
        "schema_version": "activemap-problem-origin-figure-v1",
        "test_assets_read": False,
        "source_anchor": str(args.anchor),
        "source_utility": str(args.utility),
        "literature_lines": [
            "editable_map_update",
            "select_then_fuse_mapping",
            "active_visual_search",
        ],
        "missing_intersection": [
            "sequential_evidence_choice",
            "stopping",
            "risk_aware_executable_writeback",
        ],
        "outputs": [
            "activemap_problem_origin_en.png/pdf/svg",
            "activemap_problem_origin_zh.png/pdf/svg",
        ],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
