#!/usr/bin/env python3
"""Export aligned, editable assets for the ActiveMap Figure 1 paradigm diagram."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

PALETTE = {
    "prior": "#4C78A8",
    "candidate": "#00A6A6",
    "committed": "#15803D",
    "reject": "#DC2626",
    "background_geometry": "#CBD5E1",
    "target": "#334155",
    "halo": "#FFFFFF",
    "canvas": "#FFFFFF",
    "preview_background": "#F8FAFC",
    "rollback": "#64748B",
}

LINE_WIDTHS = {
    "halo": 8,
    "map": 4,
    "edit": 5,
    "selection_border": 3,
}

EVIDENCE_PATTERN = re.compile(r"evidence_(\d{4})\.png$")


@dataclass(frozen=True)
class Viewport:
    bbox: tuple[float, float, float, float]
    size: int
    scale: float
    offset_x: float
    offset_y: float
    content_width: float
    content_height: float

    @classmethod
    def create(cls, bbox: Sequence[float], size: int) -> Viewport:
        xmin, ymin, xmax, ymax = map(float, bbox)
        width = xmax - xmin
        height = ymax - ymin
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid bbox: {bbox}")
        scale = min(size / width, size / height)
        content_width = width * scale
        content_height = height * scale
        return cls(
            bbox=(xmin, ymin, xmax, ymax),
            size=size,
            scale=scale,
            offset_x=(size - content_width) / 2.0,
            offset_y=(size - content_height) / 2.0,
            content_width=content_width,
            content_height=content_height,
        )

    def point(self, x: float, y: float) -> tuple[float, float]:
        xmin, ymin, _, _ = self.bbox
        return (
            self.offset_x + (float(x) - xmin) * self.scale,
            self.offset_y + (float(y) - ymin) * self.scale,
        )

    @property
    def image_box(self) -> tuple[int, int, int, int]:
        left = round(self.offset_x)
        top = round(self.offset_y)
        right = round(self.offset_x + self.content_width)
        bottom = round(self.offset_y + self.content_height)
        return left, top, right, bottom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--selected-evidence-from-trajectory", action="store_true")
    parser.add_argument("--export-svg", action="store_true")
    parser.add_argument("--export-png", action="store_true")
    parser.add_argument("--contact-sheet", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def public_id(kind: str, raw_id: str) -> str:
    digest = hashlib.sha256(f"activemap:{kind}:{raw_id}".encode()).hexdigest()[:16]
    return f"{kind}-{digest}"


def rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4)) + (alpha,)


def geometry_from_geojson(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("type") == "FeatureCollection":
        return {
            "type": "GeometryCollection",
            "geometries": [
                feature["geometry"]
                for feature in payload.get("features", [])
                if feature.get("geometry")
            ],
        }
    if payload.get("type") == "Feature":
        return payload["geometry"]
    return payload


def iter_geometries(geometry: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if geometry.get("type") == "GeometryCollection":
        for child in geometry.get("geometries", []):
            yield from iter_geometries(child)
    else:
        yield geometry


def iter_lines(geometry: dict[str, Any]) -> Iterator[list[list[float]]]:
    for item in iter_geometries(geometry):
        kind = item.get("type")
        coordinates = item.get("coordinates", [])
        if kind == "LineString":
            yield coordinates
        elif kind == "MultiLineString":
            yield from coordinates
        elif kind == "Polygon":
            yield from coordinates
        elif kind == "MultiPolygon":
            for polygon in coordinates:
                yield from polygon


def iter_polygons(geometry: dict[str, Any]) -> Iterator[list[list[list[float]]]]:
    for item in iter_geometries(geometry):
        kind = item.get("type")
        coordinates = item.get("coordinates", [])
        if kind == "Polygon":
            yield coordinates
        elif kind == "MultiPolygon":
            yield from coordinates


def representative_gt_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    lines = list(iter_lines(geometry))
    if not lines:
        return geometry

    def length(line: Sequence[Sequence[float]]) -> float:
        return sum(
            math.dist(start[:2], end[:2]) for start, end in zip(line, line[1:], strict=False)
        )

    ranked = sorted(lines, key=length, reverse=True)
    keep = ranked[: max(1, min(8, len(ranked) // 4 or 1))]
    return {"type": "MultiLineString", "coordinates": keep}


def load_evidence(episode_dir: Path) -> list[tuple[str, Path, np.ndarray]]:
    rows = []
    for path in sorted(episode_dir.glob("evidence_*.png")):
        match = EVIDENCE_PATTERN.match(path.name)
        if match:
            rows.append((match.group(1), path, np.asarray(Image.open(path).convert("RGB"))))
    if not rows:
        raise FileNotFoundError(f"no evidence_YYYY.png files in {episode_dir}")
    shapes = {array.shape for _, _, array in rows}
    if len(shapes) != 1:
        raise ValueError(f"evidence images are not pixel-aligned: {sorted(shapes)}")
    return rows


def global_normalization(
    evidence: Sequence[tuple[str, Path, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.concatenate([array.reshape(-1, 3) for _, _, array in evidence], axis=0)
    low = np.percentile(pixels, 2, axis=0)
    high = np.percentile(pixels, 98, axis=0)
    if np.any(high <= low):
        raise ValueError(f"invalid RGB percentile range: low={low}, high={high}")
    return low, high


def normalize_rgb(array: np.ndarray, low: np.ndarray, high: np.ndarray) -> Image.Image:
    normalized = (array.astype(np.float32) - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0)
    return Image.fromarray(np.uint8(np.round(normalized * 255.0)), mode="RGB")


def letterbox(image: Image.Image, viewport: Viewport, background: str = "#F8FAFC") -> Image.Image:
    canvas = Image.new("RGB", (viewport.size, viewport.size), background)
    left, top, right, bottom = viewport.image_box
    resized = image.resize((right - left, bottom - top), Image.Resampling.LANCZOS)
    canvas.paste(resized, (left, top))
    return canvas


def to_points(line: Sequence[Sequence[float]], viewport: Viewport) -> list[tuple[float, float]]:
    return [viewport.point(point[0], point[1]) for point in line]


def draw_dashed(
    draw: ImageDraw.ImageDraw,
    points: Sequence[tuple[float, float]],
    fill: tuple[int, int, int, int],
    width: int,
    dash: float = 12,
    gap: float = 8,
) -> None:
    for start, end in zip(points, points[1:], strict=False):
        x1, y1 = start
        x2, y2 = end
        dx, dy = x2 - x1, y2 - y1
        distance = math.hypot(dx, dy)
        if distance == 0:
            continue
        cursor = 0.0
        while cursor < distance:
            stop = min(cursor + dash, distance)
            a, b = cursor / distance, stop / distance
            draw.line(
                (
                    x1 + dx * a,
                    y1 + dy * a,
                    x1 + dx * b,
                    y1 + dy * b,
                ),
                fill=fill,
                width=width,
            )
            cursor += dash + gap


def draw_geometry(
    canvas: Image.Image,
    geometry: dict[str, Any],
    viewport: Viewport,
    color: str,
    width: int,
    *,
    halo: bool = True,
    dashed: bool = False,
    polygon_fill_alpha: int = 0,
    opacity: int = 255,
) -> None:
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_color = rgba(color, opacity)
    halo_color = rgba(PALETTE["halo"], min(opacity, 235))
    polygons = list(iter_polygons(geometry))
    if polygon_fill_alpha and polygons:
        for polygon in polygons:
            if not polygon:
                continue
            exterior = to_points(polygon[0], viewport)
            draw.polygon(exterior, fill=rgba(color, polygon_fill_alpha))
            for hole in polygon[1:]:
                draw.polygon(to_points(hole, viewport), fill=(0, 0, 0, 0))
    for line in iter_lines(geometry):
        points = to_points(line, viewport)
        if len(points) < 2:
            continue
        if halo:
            draw.line(
                points,
                fill=halo_color,
                width=LINE_WIDTHS["halo"],
                joint="curve",
            )
        if dashed:
            draw_dashed(draw, points, line_color, width)
        else:
            draw.line(points, fill=line_color, width=width, joint="curve")
    canvas.alpha_composite(overlay)


def svg_path_data(geometry: dict[str, Any], viewport: Viewport) -> str:
    parts = []
    for line in iter_lines(geometry):
        points = to_points(line, viewport)
        if not points:
            continue
        path = [f"M {points[0][0]:.3f} {points[0][1]:.3f}"]
        path.extend(f"L {x:.3f} {y:.3f}" for x, y in points[1:])
        if len(line) > 2 and line[0][:2] == line[-1][:2]:
            path.append("Z")
        parts.append(" ".join(path))
    return " ".join(parts)


def svg_document(
    viewport: Viewport,
    layers: Sequence[dict[str, Any]],
    *,
    transparent: bool = False,
    embedded_image: Path | Image.Image | None = None,
    decorations: str = "",
) -> str:
    elements = []
    if not transparent:
        elements.append(
            f'<rect width="{viewport.size}" height="{viewport.size}" fill="{PALETTE["canvas"]}"/>'
        )
    if embedded_image is not None:
        mime = "image/png"
        if isinstance(embedded_image, Path):
            image_bytes = embedded_image.read_bytes()
        else:
            buffer = io.BytesIO()
            embedded_image.convert("RGB").save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        elements.append(
            f'<image x="0" y="0" width="{viewport.size}" height="{viewport.size}" '
            f'href="data:{mime};base64,{encoded}"/>'
        )
    for layer in layers:
        path_data = svg_path_data(layer["geometry"], viewport)
        if not path_data:
            continue
        if layer.get("fill_alpha", 0):
            elements.append(
                f'<path d="{path_data}" fill="{layer["color"]}" '
                f'fill-opacity="{layer["fill_alpha"]:.3f}" '
                f'stroke="none" fill-rule="evenodd"/>'
            )
        if layer.get("halo", True):
            elements.append(
                f'<path d="{path_data}" fill="none" stroke="{PALETTE["halo"]}" '
                f'stroke-width="{LINE_WIDTHS["halo"]}" stroke-linecap="round" '
                'stroke-linejoin="round"/>'
            )
        dash = ' stroke-dasharray="12 8"' if layer.get("dashed") else ""
        opacity = layer.get("opacity", 1.0)
        elements.append(
            f'<path d="{path_data}" fill="none" stroke="{layer["color"]}" '
            f'stroke-width="{layer["width"]}" stroke-opacity="{opacity:.3f}"'
            f'{dash} stroke-linecap="round" stroke-linejoin="round"/>'
        )
    if decorations:
        elements.append(decorations)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{viewport.size}" '
        f'height="{viewport.size}" viewBox="0 0 {viewport.size} {viewport.size}">'
        + "".join(elements)
        + "</svg>\n"
    )


def save_svg(path: Path, content: str, enabled: bool, generated: list[str], root: Path) -> None:
    if enabled:
        path.write_text(content, encoding="utf-8")
        generated.append(path.relative_to(root).as_posix())


def save_png(
    path: Path, image: Image.Image, enabled: bool, generated: list[str], root: Path
) -> None:
    if enabled:
        image.convert("RGB").save(path, format="PNG", optimize=True)
        generated.append(path.relative_to(root).as_posix())


def save_vector_asset(
    stem: Path,
    viewport: Viewport,
    layers: Sequence[dict[str, Any]],
    generated: list[str],
    root: Path,
    *,
    export_svg: bool,
    export_png: bool,
    transparent: bool = False,
    embedded_image: Path | Image.Image | None = None,
    decorations_svg: str = "",
    decorations_png: Any | None = None,
) -> None:
    svg = svg_document(
        viewport,
        layers,
        transparent=transparent,
        embedded_image=embedded_image,
        decorations=decorations_svg,
    )
    save_svg(stem.with_suffix(".svg"), svg, export_svg, generated, root)
    if export_png:
        background = (255, 255, 255, 0) if transparent else (255, 255, 255, 255)
        if embedded_image is not None:
            canvas = (
                Image.open(embedded_image).convert("RGBA")
                if isinstance(embedded_image, Path)
                else embedded_image.convert("RGBA")
            )
        else:
            canvas = Image.new("RGBA", (viewport.size, viewport.size), background)
        for layer in layers:
            draw_geometry(
                canvas,
                layer["geometry"],
                viewport,
                layer["color"],
                layer["width"],
                halo=layer.get("halo", True),
                dashed=layer.get("dashed", False),
                polygon_fill_alpha=round(layer.get("fill_alpha", 0.0) * 255),
                opacity=round(layer.get("opacity", 1.0) * 255),
            )
        if decorations_png is not None:
            decorations_png(canvas)
        path = stem.with_suffix(".png")
        if transparent:
            canvas.save(path, format="PNG", optimize=True)
            generated.append(path.relative_to(root).as_posix())
        else:
            save_png(path, canvas, True, generated, root)


def first_acquire(
    episode_dir: Path,
    metadata: dict[str, Any],
) -> tuple[str | None, str]:
    trajectory_path = episode_dir / "trajectory.jsonl"
    if not trajectory_path.exists():
        return None, "fallback_latest_timestamp"
    expected_task = public_id("task", str(metadata.get("episode_id", "")))
    for row in read_jsonl(trajectory_path):
        if row.get("task_id") not in {None, expected_task}:
            continue
        for action in row.get("actions", []):
            if action.get("stage") == "SELECT" and action.get("selection") == "ACQUIRE":
                return str(action.get("evidence_id")), "trajectory_first_acquire"
    return None, "fallback_latest_timestamp"


def selected_year(
    episode_dir: Path,
    metadata: dict[str, Any],
    years: Sequence[str],
    from_trajectory: bool,
) -> tuple[str, str, str | None]:
    if from_trajectory:
        evidence_id, source = first_acquire(episode_dir, metadata)
        if evidence_id:
            for item in metadata.get("evidence", []):
                raw_id = item.get("evidence_id") or item.get("raw_id")
                output = str(item.get("output", ""))
                year = str(item.get("year") or item.get("timestamp", "")[:4])
                candidates = {raw_id, public_id("evidence", raw_id) if raw_id else None}
                if evidence_id in candidates and year in years:
                    return year, source, evidence_id
                if year in years and year in evidence_id:
                    return year, source, evidence_id
                if (
                    output
                    and year in output
                    and public_id(
                        "evidence", f"{metadata['episode_id'].removesuffix('__temporal')}__y{year}"
                    )
                    == evidence_id
                ):
                    return year, source, evidence_id
            resource = metadata.get("real_trajectory_resources", {})
            raw_selected = str(resource.get("selected_evidence_raw_id", ""))
            for year in years:
                if year in raw_selected:
                    return year, source, evidence_id
    return max(years), "latest_timestamp_fallback", None


def compose_horizontal_pool(
    images: Sequence[tuple[str, Image.Image]],
    size: int,
    selected: str | None = None,
) -> Image.Image:
    canvas = Image.new("RGB", (size, size), PALETTE["preview_background"])
    gap = max(6, size // 64)
    margin = max(8, size // 40)
    tile_width = (size - 2 * margin - gap * (len(images) - 1)) // len(images)
    tile_height = round(tile_width * 530 / 681)
    top = (size - tile_height) // 2
    draw = ImageDraw.Draw(canvas)
    for index, (year, image) in enumerate(images):
        tile = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        if selected is not None and year != selected:
            tile = Image.blend(tile, Image.new("RGB", tile.size, "white"), 0.60)
        left = margin + index * (tile_width + gap)
        canvas.paste(tile, (left, top))
        border = PALETTE["committed"] if year == selected else "#CBD5E1"
        border_width = LINE_WIDTHS["selection_border"] if year == selected else 1
        draw.rectangle(
            (left - border_width, top - border_width, left + tile_width, top + tile_height),
            outline=border,
            width=border_width,
        )
        if year == selected:
            radius = max(7, size // 40)
            cx, cy = left + tile_width - radius - 4, top + radius + 4
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius), fill=PALETTE["committed"]
            )
            draw.line((cx - radius // 2, cy, cx - 1, cy + radius // 2), fill="white", width=3)
            draw.line(
                (cx - 1, cy + radius // 2, cx + radius // 2, cy - radius // 2),
                fill="white",
                width=3,
            )
    return canvas


def compose_stacked_pool(images: Sequence[tuple[str, Image.Image]], size: int) -> Image.Image:
    canvas = Image.new("RGB", (size, size), PALETTE["preview_background"])
    card_width = round(size * 0.72)
    card_height = round(card_width * 530 / 681)
    offsets = [(-34, -30), (-18, -14), (0, 2), (18, 18)]
    center_x, center_y = size // 2, size // 2
    for index, (_, image) in enumerate(images):
        card = image.resize((card_width, card_height), Image.Resampling.LANCZOS)
        card = ImageEnhance.Brightness(card).enhance(0.78 + 0.07 * index)
        framed = Image.new("RGB", (card_width + 8, card_height + 8), "white")
        framed.paste(card, (4, 4))
        left = center_x - framed.width // 2 + offsets[index][0]
        top = center_y - framed.height // 2 + offsets[index][1]
        canvas.paste(framed, (left, top))
    return canvas


def belief_values(episode_dir: Path) -> tuple[float | None, float | None, str]:
    trajectory = episode_dir / "trajectory.jsonl"
    if trajectory.exists():
        for row in read_jsonl(trajectory):
            before = None
            after = None
            for action in row.get("actions", []):
                if action.get("stage") == "DRAFT" and action.get("confidence") is not None:
                    before = float(action["confidence"])
                if action.get("stage") == "BELIEF_UPDATE" and action.get("confidence") is not None:
                    after = float(action["confidence"])
            if before is not None and after is not None:
                return before, after, "trajectory_action_confidence"
    return None, None, "unavailable"


def belief_svg(size: int, before: float, after: float) -> str:
    baseline = size * 0.82
    max_height = size * 0.62
    bar_width = size * 0.18
    x1, x2 = size * 0.20, size * 0.62
    h1, h2 = max_height * before, max_height * after
    arrow = (
        f'<path d="M {size * 0.43:.1f} {size * 0.50:.1f} H {size * 0.56:.1f} '
        f"M {size * 0.51:.1f} {size * 0.45:.1f} L {size * 0.56:.1f} {size * 0.50:.1f} "
        f'L {size * 0.51:.1f} {size * 0.55:.1f}" fill="none" stroke="#64748B" '
        'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">'
        f'<rect width="{size}" height="{size}" fill="#FFFFFF"/>'
        f'<line x1="{size * 0.12}" y1="{baseline}" x2="{size * 0.88}" '
        f'y2="{baseline}" stroke="#CBD5E1" stroke-width="3"/>'
        f'<rect x="{x1}" y="{baseline - h1}" width="{bar_width}" '
        f'height="{h1}" rx="4" fill="#94A3B8"/>'
        f'<rect x="{x2}" y="{baseline - h2}" width="{bar_width}" '
        f'height="{h2}" rx="4" fill="{PALETTE["committed"]}"/>'
        f"{arrow}</svg>\n"
    )


def belief_png(size: int, before: float, after: float) -> Image.Image:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    baseline = round(size * 0.82)
    max_height = size * 0.62
    bar_width = round(size * 0.18)
    x1, x2 = round(size * 0.20), round(size * 0.62)
    draw.line((size * 0.12, baseline, size * 0.88, baseline), fill="#CBD5E1", width=3)
    for x, value, color in ((x1, before, "#94A3B8"), (x2, after, PALETTE["committed"])):
        height = round(max_height * value)
        draw.rounded_rectangle(
            (x, baseline - height, x + bar_width, baseline), radius=4, fill=color
        )
    draw.line((size * 0.43, size * 0.50, size * 0.56, size * 0.50), fill="#64748B", width=4)
    draw.line((size * 0.51, size * 0.45, size * 0.56, size * 0.50), fill="#64748B", width=4)
    draw.line((size * 0.51, size * 0.55, size * 0.56, size * 0.50), fill="#64748B", width=4)
    return image


def arrow_preview(
    draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color="#2563EB"
) -> None:
    draw.line((*start, *end), fill=color, width=5)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    for offset in (-0.55, 0.55):
        tip = (
            end[0] - 15 * math.cos(angle + offset),
            end[1] - 15 * math.sin(angle + offset),
        )
        draw.line((*end, *tip), fill=color, width=5)


def preview_strip(paths: Sequence[Path], output: Path, title: str) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    tile = 260
    gap = 54
    top = 58
    canvas = Image.new(
        "RGB", (tile * len(images) + gap * (len(images) - 1) + 40, tile + 90), "white"
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((20, 18), title, fill="#0F172A", font=font)
    for index, image in enumerate(images):
        left = 20 + index * (tile + gap)
        canvas.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (left, top))
        if index + 1 < len(images):
            arrow_preview(
                draw, (left + tile + 10, top + tile // 2), (left + tile + gap - 10, top + tile // 2)
            )
    canvas.save(output, format="PNG", optimize=True)


def make_contact_sheet(paths: Sequence[Path], output: Path) -> None:
    entries = [(path, Image.open(path).convert("RGB")) for path in paths if path.exists()]
    columns = 4
    tile = 220
    label_height = 30
    rows = math.ceil(len(entries) / columns)
    canvas = Image.new("RGB", (columns * tile, rows * (tile + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (path, image) in enumerate(entries):
        column = index % columns
        row = index // columns
        left = column * tile
        top = row * (tile + label_height)
        canvas.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (left, top))
        draw.text((left + 6, top + tile + 8), path.stem[:34], fill="#334155", font=font)
    canvas.save(output, format="PNG", optimize=True)


def main() -> None:
    args = parse_args()
    episode_dir = args.episode_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.size <= 0:
        raise ValueError("--size must be positive")
    export_svg = args.export_svg or not (args.export_svg or args.export_png)
    export_png = args.export_png or not (args.export_svg or args.export_png)

    required = ["metadata.json", "prior.geojson", "target.geojson", "ground_truth_add.geojson"]
    missing = [name for name in required if not (episode_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing required episode assets: {missing}")

    metadata = read_json(episode_dir / "metadata.json")
    bbox = metadata.get("bbox") or metadata.get("region_xyxy")
    if not bbox or len(bbox) != 4:
        raise ValueError("metadata.json must contain bbox or region_xyxy")
    viewport = Viewport.create(bbox, args.size)
    evidence = load_evidence(episode_dir)
    low, high = global_normalization(evidence)
    years = [year for year, _, _ in evidence]
    chosen_year, selection_source, selected_public_id = selected_year(
        episode_dir,
        metadata,
        years,
        args.selected_evidence_from_trajectory,
    )

    for folder in ("imagery", "maps", "edits", "belief", "previews"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    asset_status: dict[str, dict[str, Any]] = {}
    normalized_images: dict[str, Image.Image] = {}
    for year, source, array in evidence:
        normalized = letterbox(normalize_rgb(array, low, high), viewport)
        normalized_images[year] = normalized
        destination = output_dir / "imagery" / f"candidate_evidence_{year}.png"
        save_png(destination, normalized, export_png, generated, output_dir)
        asset_status[destination.stem] = {"real": True, "schematic": False, "source": str(source)}

    visible = normalized_images[max(years)]
    visible_path = output_dir / "imagery" / "visible_evidence.png"
    fixed_path = output_dir / "imagery" / "fixed_observation_thumbnail.png"
    save_png(visible_path, visible, export_png, generated, output_dir)
    save_png(fixed_path, visible, export_png, generated, output_dir)
    asset_status[visible_path.stem] = {
        "real": True,
        "schematic": False,
        "source": f"evidence_{max(years)}.png",
    }
    asset_status[fixed_path.stem] = {
        "real": True,
        "schematic": False,
        "source": f"evidence_{max(years)}.png",
    }

    pool_images = [(year, normalized_images[year]) for year in years]
    horizontal = compose_horizontal_pool(pool_images, args.size)
    stacked = compose_stacked_pool(pool_images, args.size)
    selected = compose_horizontal_pool(pool_images, args.size, selected=chosen_year)
    for name, image in (
        ("candidate_evidence_pool_horizontal", horizontal),
        ("candidate_evidence_pool_stacked", stacked),
        ("selected_evidence", selected),
    ):
        path = output_dir / "imagery" / f"{name}.png"
        save_png(path, image, export_png, generated, output_dir)
        asset_status[name] = {
            "real": True,
            "schematic": False,
            "source": "normalized evidence pool",
        }

    prior = geometry_from_geojson(episode_dir / "prior.geojson")
    target = geometry_from_geojson(episode_dir / "target.geojson")
    gt_add = geometry_from_geojson(episode_dir / "ground_truth_add.geojson")
    map_specs = {
        "prior_map_thumbnail": [
            {"geometry": prior, "color": PALETTE["prior"], "width": LINE_WIDTHS["map"]}
        ],
        "new_map_thumbnail": [
            {"geometry": target, "color": PALETTE["target"], "width": LINE_WIDTHS["map"]}
        ],
        "updated_map_thumbnail": [
            {"geometry": prior, "color": PALETTE["prior"], "width": LINE_WIDTHS["map"]},
            {"geometry": gt_add, "color": PALETTE["committed"], "width": LINE_WIDTHS["edit"]},
        ],
    }
    for name, layers in map_specs.items():
        save_vector_asset(
            output_dir / "maps" / name,
            viewport,
            layers,
            generated,
            output_dir,
            export_svg=export_svg,
            export_png=export_png,
        )
        asset_status[name] = {
            "real": True,
            "schematic": False,
            "source": ["prior.geojson", "target.geojson", "ground_truth_add.geojson"],
        }
    save_vector_asset(
        output_dir / "maps" / "prior_map_thumbnail_transparent",
        viewport,
        map_specs["prior_map_thumbnail"],
        generated,
        output_dir,
        export_svg=export_svg,
        export_png=export_png,
        transparent=True,
    )
    save_vector_asset(
        output_dir / "maps" / "new_map_thumbnail_transparent",
        viewport,
        map_specs["new_map_thumbnail"],
        generated,
        output_dir,
        export_svg=export_svg,
        export_png=export_png,
        transparent=True,
    )
    asset_status["prior_map_thumbnail_transparent"] = {
        "real": True,
        "schematic": False,
        "source": "prior.geojson",
    }
    asset_status["new_map_thumbnail_transparent"] = {
        "real": True,
        "schematic": False,
        "source": "target.geojson",
    }
    save_vector_asset(
        output_dir / "maps" / "prior_plus_visible_evidence",
        viewport,
        map_specs["prior_map_thumbnail"],
        generated,
        output_dir,
        export_svg=export_svg,
        export_png=export_png,
        embedded_image=visible,
    )
    asset_status["prior_plus_visible_evidence"] = {
        "real": True,
        "schematic": False,
        "source": ["prior.geojson", f"evidence_{max(years)}.png"],
    }

    proposed_path = episode_dir / "proposed_edit.geojson"
    if proposed_path.exists():
        proposed = geometry_from_geojson(proposed_path)
        candidate_source = "real_model_output_from_proposed_edit.geojson"
        candidate_real = True
    else:
        proposed = representative_gt_geometry(gt_add)
        candidate_source = "schematic_from_gt_delta"
        candidate_real = False
    proposed_layers = [
        {"geometry": prior, "color": PALETTE["prior"], "width": LINE_WIDTHS["map"]},
        {
            "geometry": proposed,
            "color": PALETTE["candidate"],
            "width": LINE_WIDTHS["edit"],
            "dashed": True,
            "fill_alpha": 0.12 if list(iter_polygons(proposed)) else 0.0,
        },
    ]
    save_vector_asset(
        output_dir / "edits" / "proposed_edit_thumbnail",
        viewport,
        proposed_layers,
        generated,
        output_dir,
        export_svg=export_svg,
        export_png=export_png,
    )
    asset_status["proposed_edit_thumbnail"] = {
        "real": candidate_real,
        "schematic": not candidate_real,
        "source": candidate_source,
        "not_used_for_quantitative_evaluation": True,
    }

    committed_path = episode_dir / "committed_edit.geojson"
    committed = geometry_from_geojson(committed_path) if committed_path.exists() else proposed
    committed_real = committed_path.exists()
    committed_layers = [
        {"geometry": prior, "color": PALETTE["prior"], "width": LINE_WIDTHS["map"]},
        {
            "geometry": committed,
            "color": PALETTE["committed"],
            "width": LINE_WIDTHS["edit"],
            "fill_alpha": 0.10 if list(iter_polygons(committed)) else 0.0,
        },
    ]
    save_vector_asset(
        output_dir / "edits" / "committed_edit_thumbnail",
        viewport,
        committed_layers,
        generated,
        output_dir,
        export_svg=export_svg,
        export_png=export_png,
    )
    asset_status["committed_edit_thumbnail"] = {
        "real": committed_real,
        "schematic": not committed_real,
        "source": "committed_edit.geojson"
        if committed_real
        else "schematic_commit_from_proposed_edit",
    }

    cross_svg = (
        f'<circle cx="{args.size * 0.82:.1f}" cy="{args.size * 0.18:.1f}" '
        f'r="{args.size * 0.07:.1f}" fill="#FFFFFF" '
        f'stroke="{PALETTE["reject"]}" stroke-width="5"/>'
        f'<path d="M {args.size * 0.785:.1f} {args.size * 0.145:.1f} '
        f"L {args.size * 0.855:.1f} {args.size * 0.215:.1f} "
        f"M {args.size * 0.855:.1f} {args.size * 0.145:.1f} "
        f'L {args.size * 0.785:.1f} {args.size * 0.215:.1f}" '
        f'stroke="{PALETTE["reject"]}" stroke-width="6" stroke-linecap="round"/>'
    )

    def cross_png(canvas: Image.Image) -> None:
        draw = ImageDraw.Draw(canvas)
        cx, cy, radius = args.size * 0.82, args.size * 0.18, args.size * 0.07
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill="white",
            outline=PALETTE["reject"],
            width=5,
        )
        draw.line(
            (cx - radius * 0.5, cy - radius * 0.5, cx + radius * 0.5, cy + radius * 0.5),
            fill=PALETTE["reject"],
            width=6,
        )
        draw.line(
            (cx + radius * 0.5, cy - radius * 0.5, cx - radius * 0.5, cy + radius * 0.5),
            fill=PALETTE["reject"],
            width=6,
        )

    rejected_layers = [
        {"geometry": prior, "color": PALETTE["prior"], "width": LINE_WIDTHS["map"]},
        {
            "geometry": proposed,
            "color": PALETTE["reject"],
            "width": LINE_WIDTHS["edit"],
            "dashed": True,
            "opacity": 0.82,
        },
    ]
    save_vector_asset(
        output_dir / "edits" / "rejected_edit_thumbnail",
        viewport,
        rejected_layers,
        generated,
        output_dir,
        export_svg=export_svg,
        export_png=export_png,
        decorations_svg=cross_svg,
        decorations_png=cross_png,
    )
    asset_status["rejected_edit_thumbnail"] = {
        "real": False,
        "schematic": True,
        "source": "safety_branch_from_proposed_edit",
    }

    rollback_svg = (
        f'<path d="M {args.size * 0.82:.1f} {args.size * 0.16:.1f} '
        f"A {args.size * 0.09:.1f} {args.size * 0.09:.1f} 0 1 0 "
        f'{args.size * 0.84:.1f} {args.size * 0.27:.1f}" fill="none" '
        f'stroke="{PALETTE["rollback"]}" stroke-width="6" stroke-linecap="round"/>'
        f'<path d="M {args.size * 0.78:.1f} {args.size * 0.13:.1f} '
        f"L {args.size * 0.82:.1f} {args.size * 0.16:.1f} "
        f'L {args.size * 0.77:.1f} {args.size * 0.18:.1f}" fill="none" '
        f'stroke="{PALETTE["rollback"]}" stroke-width="6" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )

    def rollback_png(canvas: Image.Image) -> None:
        draw = ImageDraw.Draw(canvas)
        box = (args.size * 0.73, args.size * 0.09, args.size * 0.91, args.size * 0.27)
        draw.arc(box, start=210, end=535, fill=PALETTE["rollback"], width=6)
        draw.line(
            (args.size * 0.78, args.size * 0.13, args.size * 0.82, args.size * 0.16),
            fill=PALETTE["rollback"],
            width=6,
        )
        draw.line(
            (args.size * 0.78, args.size * 0.13, args.size * 0.77, args.size * 0.18),
            fill=PALETTE["rollback"],
            width=6,
        )

    rollback_layers = [
        {"geometry": prior, "color": PALETTE["prior"], "width": LINE_WIDTHS["map"]},
        {
            "geometry": proposed,
            "color": PALETTE["rollback"],
            "width": LINE_WIDTHS["edit"],
            "dashed": True,
            "opacity": 0.28,
        },
    ]
    save_vector_asset(
        output_dir / "edits" / "rollback_edit_thumbnail",
        viewport,
        rollback_layers,
        generated,
        output_dir,
        export_svg=export_svg,
        export_png=export_png,
        decorations_svg=rollback_svg,
        decorations_png=rollback_png,
    )
    asset_status["rollback_edit_thumbnail"] = {
        "real": False,
        "schematic": True,
        "source": "safety_branch_from_proposed_edit",
    }

    before, after, belief_source = belief_values(episode_dir)
    if before is not None and after is not None:
        numeric_svg_path = output_dir / "belief" / "belief_update_numeric.svg"
        numeric_png_path = output_dir / "belief" / "belief_update_numeric.png"
        save_svg(
            numeric_svg_path,
            belief_svg(args.size, before, after),
            export_svg,
            generated,
            output_dir,
        )
        save_png(
            numeric_png_path,
            belief_png(args.size, before, after),
            export_png,
            generated,
            output_dir,
        )
        asset_status["belief_update_numeric"] = {
            "real": True,
            "schematic": False,
            "source": belief_source,
            "before": before,
            "after": after,
        }
    abstract_svg_path = output_dir / "belief" / "belief_update_abstract.svg"
    abstract_png_path = output_dir / "belief" / "belief_update_abstract.png"
    save_svg(
        abstract_svg_path, belief_svg(args.size, 0.32, 0.78), export_svg, generated, output_dir
    )
    save_png(
        abstract_png_path, belief_png(args.size, 0.32, 0.78), export_png, generated, output_dir
    )
    asset_status["belief_update_abstract"] = {
        "real": False,
        "schematic": True,
        "source": "abstract_no_numeric_labels",
    }

    if export_png:
        previews = {
            "passive_construction_preview": [
                fixed_path,
                output_dir / "maps" / "new_map_thumbnail.png",
            ],
            "passive_updating_preview": [
                output_dir / "maps" / "prior_plus_visible_evidence.png",
                output_dir / "maps" / "updated_map_thumbnail.png",
            ],
            "active_perception_preview": [
                output_dir / "imagery" / "candidate_evidence_pool_horizontal.png",
                output_dir / "imagery" / "selected_evidence.png",
                output_dir / "maps" / "new_map_thumbnail.png",
            ],
            "active_map_preview": [
                output_dir / "maps" / "prior_map_thumbnail.png",
                output_dir / "edits" / "proposed_edit_thumbnail.png",
                output_dir / "imagery" / "selected_evidence.png",
                output_dir / "belief" / "belief_update_abstract.png",
                output_dir / "edits" / "committed_edit_thumbnail.png",
            ],
        }
        for name, paths in previews.items():
            destination = output_dir / "previews" / f"{name}.png"
            preview_strip(paths, destination, name.replace("_", " ").title())
            generated.append(destination.relative_to(output_dir).as_posix())
            asset_status[name] = {
                "real": False,
                "schematic": True,
                "preview_only": True,
                "source": [path.stem for path in paths],
            }
        if args.contact_sheet:
            contact_paths = [
                output_dir / "imagery" / "visible_evidence.png",
                output_dir / "imagery" / "candidate_evidence_pool_horizontal.png",
                output_dir / "imagery" / "candidate_evidence_pool_stacked.png",
                output_dir / "imagery" / "selected_evidence.png",
                output_dir / "maps" / "prior_map_thumbnail.png",
                output_dir / "maps" / "new_map_thumbnail.png",
                output_dir / "maps" / "updated_map_thumbnail.png",
                output_dir / "maps" / "prior_plus_visible_evidence.png",
                output_dir / "edits" / "proposed_edit_thumbnail.png",
                output_dir / "edits" / "committed_edit_thumbnail.png",
                output_dir / "edits" / "rejected_edit_thumbnail.png",
                output_dir / "edits" / "rollback_edit_thumbnail.png",
                output_dir / "belief" / "belief_update_numeric.png",
                output_dir / "belief" / "belief_update_abstract.png",
            ]
            contact = output_dir / "previews" / "all_assets_contact_sheet.png"
            make_contact_sheet(contact_paths, contact)
            generated.append(contact.relative_to(output_dir).as_posix())
            asset_status["all_assets_contact_sheet"] = {
                "real": False,
                "schematic": True,
                "preview_only": True,
                "source": [path.stem for path in contact_paths if path.exists()],
            }

    source_files = {
        path.name: str(path.resolve()) for path in sorted(episode_dir.iterdir()) if path.is_file()
    }
    ignored_available_files = [
        name
        for name in (
            "current_2019_with_target.png",
            "current_2019_with_ground_truth_add.png",
        )
        if (episode_dir / name).exists()
    ]
    generated_asset_status = {
        relative_path: asset_status.get(
            Path(relative_path).stem,
            {"real": False, "schematic": True, "status_missing": True},
        )
        for relative_path in sorted(generated)
    }
    manifest = {
        "schema_version": "figure1-paradigm-assets-v1",
        "purpose": "conceptual paradigm comparison only; not quantitative evaluation",
        "episode_id": metadata.get("episode_id"),
        "source_files": source_files,
        "crs": metadata.get("crs"),
        "coordinate_space": "source-image pixel coordinates"
        if metadata.get("crs") is None
        else "declared CRS",
        "bbox": list(map(float, bbox)),
        "crop_size": metadata.get("crop_size") or list(evidence[0][2].shape[1::-1]),
        "output_size": [args.size, args.size],
        "content_viewport": {
            "offset_xy": [viewport.offset_x, viewport.offset_y],
            "size": [viewport.content_width, viewport.content_height],
            "letterboxed": True,
        },
        "rgb_normalization": {
            "method": "joint per-channel percentile across all evidence timestamps",
            "lower_percentile": 2,
            "upper_percentile": 98,
            "rgb_low": low.tolist(),
            "rgb_high": high.tolist(),
            "per_timestamp_equalization": False,
        },
        "selected_evidence_time": chosen_year,
        "selected_evidence_source": selection_source,
        "selected_evidence_public_id": selected_public_id,
        "candidate_edit_source": candidate_source,
        "candidate_edit_real_model_output": candidate_real,
        "not_used_for_quantitative_evaluation": True,
        "target_leakage": False,
        "forbidden_inputs_used": [],
        "ignored_available_files": ignored_available_files,
        "palette": PALETTE,
        "line_widths": LINE_WIDTHS,
        "asset_status": asset_status,
        "generated_asset_status": generated_asset_status,
        "generated_file_list": sorted(generated),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "generated_files": len(generated) + 1,
                "selected_evidence_time": chosen_year,
                "candidate_edit_source": candidate_source,
                "committed_edit_schematic": not committed_real,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
