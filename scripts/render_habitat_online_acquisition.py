#!/usr/bin/env python3
"""Render map-native comparisons for Habitat online-acquisition rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

UNKNOWN = np.float32(0.5)
LEGEND = (
    ("Unknown", (166, 174, 184)),
    ("Correct free", (255, 255, 255)),
    ("Correct obstacle", (32, 38, 46)),
    ("False free", (220, 75, 64)),
    ("False obstacle", (232, 145, 50)),
    ("RGB-D acquired", (44, 160, 101)),
    ("RGB-D skipped", (246, 190, 62)),
)


def parse_panel(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("panel must use LABEL=OCCUPANCY_PATH")
    label, raw_path = raw.split("=", 1)
    path = Path(raw_path)
    if not label or not path.is_file():
        raise ValueError(f"invalid panel: {raw}")
    return label, path


def parse_rgb_panel(raw: str) -> tuple[str, Path]:
    return parse_panel(raw)


def parse_summary_panel(raw: str) -> tuple[str, Path]:
    """Parse a per-method rollout summary keyed by its map-panel label."""
    return parse_panel(raw)


def occupancy_rgb(committed: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Render unknown cells and directional occupancy errors against reference."""
    valid = reference != UNKNOWN
    image = np.full((*reference.shape, 3), (246, 246, 246), dtype=np.uint8)
    image[valid & (committed == UNKNOWN)] = (166, 174, 184)
    image[valid & (committed == 0.0) & (reference == 0.0)] = (255, 255, 255)
    image[valid & (committed == 1.0) & (reference == 1.0)] = (32, 38, 46)
    image[valid & (committed == 0.0) & (reference == 1.0)] = (220, 75, 64)
    image[valid & (committed == 1.0) & (reference == 0.0)] = (232, 145, 50)
    return image


def crop_to_reference(reference: np.ndarray, *, padding: int = 4) -> tuple[slice, slice]:
    rows, cols = np.nonzero(reference != UNKNOWN)
    if not len(rows):
        return slice(0, reference.shape[0]), slice(0, reference.shape[1])
    return (
        slice(
            max(0, int(rows.min()) - padding),
            min(reference.shape[0], int(rows.max()) + padding + 1),
        ),
        slice(
            max(0, int(cols.min()) - padding),
            min(reference.shape[1], int(cols.max()) + padding + 1),
        ),
    )


def draw_world_markers(
    draw: ImageDraw.ImageDraw,
    *,
    origin: tuple[int, int],
    panel_size: tuple[int, int],
    crop: tuple[slice, slice],
    reference_shape: tuple[int, int],
    summary: dict[str, object] | None,
) -> None:
    if summary is None:
        return
    grid = summary.get("grid")
    start = summary.get("start_position")
    goal = summary.get("goal_position")
    if not isinstance(grid, dict) or not isinstance(start, list) or not isinstance(goal, list):
        return
    try:
        x_min = float(grid["x_min"])
        z_max = float(grid["z_max"])
        resolution = float(grid["resolution_m"])
        positions = ((start, (25, 172, 214), "S"), (goal, (183, 83, 182), "G"))
        for position, color, label in positions:
            row = (z_max - float(position[2])) / resolution - crop[0].start
            col = (float(position[0]) - x_min) / resolution - crop[1].start
            if not 0 <= row < reference_shape[0] or not 0 <= col < reference_shape[1]:
                continue
            center_x = origin[0] + (col + 0.5) * panel_size[0] / reference_shape[1]
            center_y = origin[1] + (row + 0.5) * panel_size[1] / reference_shape[0]
            radius = 6
            draw.ellipse(
                (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
                fill=color,
                outline=(255, 255, 255),
            )
            draw.text((center_x - 3, center_y - 5), label, fill=(255, 255, 255))
    except (KeyError, TypeError, ValueError, IndexError):
        return


def draw_trajectory(
    draw: ImageDraw.ImageDraw,
    *,
    origin: tuple[int, int],
    panel_size: tuple[int, int],
    crop: tuple[slice, slice],
    reference_shape: tuple[int, int],
    summary: dict[str, object] | None,
) -> None:
    if summary is None:
        return
    grid = summary.get("grid")
    start = summary.get("start_position")
    trace = summary.get("trace")
    if not isinstance(grid, dict) or not isinstance(start, list) or not isinstance(trace, list):
        return
    try:
        x_min = float(grid["x_min"])
        z_max = float(grid["z_max"])
        resolution = float(grid["resolution_m"])
        positions = [start, *[row["position"] for row in trace if isinstance(row, dict)]]
        points = []
        for position in positions:
            row = (z_max - float(position[2])) / resolution - crop[0].start
            col = (float(position[0]) - x_min) / resolution - crop[1].start
            if 0 <= row < reference_shape[0] and 0 <= col < reference_shape[1]:
                points.append(
                    (
                        origin[0] + (col + 0.5) * panel_size[0] / reference_shape[1],
                        origin[1] + (row + 0.5) * panel_size[1] / reference_shape[0],
                    )
                )
        if len(points) > 1:
            draw.line(points, fill=(25, 172, 214), width=2)
    except (KeyError, TypeError, ValueError, IndexError):
        return


def _world_to_panel(
    position: list[object],
    *,
    origin: tuple[int, int],
    panel_size: tuple[int, int],
    crop: tuple[slice, slice],
    reference_shape: tuple[int, int],
    summary: dict[str, object],
) -> tuple[float, float] | None:
    """Convert a logged Habitat world position to a cropped map pixel."""
    grid = summary.get("grid")
    if not isinstance(grid, dict) or len(position) < 3:
        return None
    try:
        x_min = float(grid["x_min"])
        z_max = float(grid["z_max"])
        resolution = float(grid["resolution_m"])
        row = (z_max - float(position[2])) / resolution - crop[0].start
        col = (float(position[0]) - x_min) / resolution - crop[1].start
        if not 0 <= row < reference_shape[0] or not 0 <= col < reference_shape[1]:
            return None
        return (
            origin[0] + (col + 0.5) * panel_size[0] / reference_shape[1],
            origin[1] + (row + 0.5) * panel_size[1] / reference_shape[0],
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def draw_acquisition_decisions(
    draw: ImageDraw.ImageDraw,
    *,
    origin: tuple[int, int],
    panel_size: tuple[int, int],
    crop: tuple[slice, slice],
    reference_shape: tuple[int, int],
    summary: dict[str, object] | None,
) -> None:
    """Overlay each policy's observable acquire/skip decisions on its own map."""
    if summary is None or not isinstance(summary.get("trace"), list):
        return
    for row in summary["trace"]:
        if not isinstance(row, dict) or not isinstance(row.get("position"), list):
            continue
        center = _world_to_panel(
            row["position"],
            origin=origin,
            panel_size=panel_size,
            crop=crop,
            reference_shape=reference_shape,
            summary=summary,
        )
        if center is None:
            continue
        x, y = center
        acquired = bool(row.get("acquired"))
        color = (44, 160, 101) if acquired else (246, 190, 62)
        radius = 3 if acquired else 2
        if acquired:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255))
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(250, 250, 250), outline=color, width=1)


def panel_status(summary: dict[str, object] | None) -> str:
    """Return a compact, per-policy outcome label for the map panel."""
    if summary is None:
        return "reference"
    calls = summary.get("sensor_calls")
    quality = summary.get("final_reference_map_quality")
    fragments = []
    if isinstance(calls, int):
        fragments.append(f"{calls} views")
    if isinstance(quality, (float, int)):
        fragments.append(f"agreement {float(quality):.3f}")
    return " | ".join(fragments) or "rollout"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--panel", action="append", required=True)
    parser.add_argument("--rgb-panel", action="append", default=[])
    parser.add_argument(
        "--reference-summary",
        type=Path,
        help="optional all-acquired rollout summary for the reference map panel",
    )
    parser.add_argument(
        "--panel-summary",
        action="append",
        default=[],
        help="LABEL=SUMMARY_JSON; label must match a --panel label",
    )
    parser.add_argument(
        "--trajectory-summary",
        type=Path,
        help="deprecated fallback summary used for all panels",
    )
    parser.add_argument("--panel-width", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        not args.reference.is_file()
        or args.panel_width < 80
        or (args.trajectory_summary is not None and not args.trajectory_summary.is_file())
        or (args.reference_summary is not None and not args.reference_summary.is_file())
    ):
        raise ValueError("reference must exist and panel_width must be at least 80")

    reference = np.load(args.reference).astype(np.float32)
    fallback_summary = (
        json.loads(args.trajectory_summary.read_text(encoding="utf-8"))
        if args.trajectory_summary is not None
        else None
    )
    reference_summary = (
        json.loads(args.reference_summary.read_text(encoding="utf-8"))
        if args.reference_summary is not None
        else None
    )
    summary_paths = dict(parse_summary_panel(raw) for raw in args.panel_summary)
    panel_summaries: dict[str, dict[str, object]] = {}
    for label, path in summary_paths.items():
        panel_summaries[label] = json.loads(path.read_text(encoding="utf-8"))
    panels = [("All acquired reference", reference, reference_summary)]
    for label, path in (parse_panel(raw) for raw in args.panel):
        occupancy = np.load(path).astype(np.float32)
        if occupancy.shape != reference.shape:
            raise ValueError(f"occupancy shape mismatch: {path}")
        panels.append((label, occupancy, panel_summaries.get(label, fallback_summary)))
    row_slice, col_slice = crop_to_reference(reference)
    reference = reference[row_slice, col_slice]
    panels = [
        (label, occupancy[row_slice, col_slice], summary)
        for label, occupancy, summary in panels
    ]
    panel_height = max(1, round(args.panel_width * reference.shape[0] / reference.shape[1]))
    margin, title_height, legend_height = 12, 52, 28
    rgb_panels = [parse_rgb_panel(raw) for raw in args.rgb_panel]
    rgb_height = 180 if rgb_panels else 0
    rgb_rows = (len(rgb_panels) + len(panels) - 1) // len(panels) if rgb_panels else 0
    canvas = Image.new(
        "RGB",
        (
            len(panels) * (args.panel_width + margin) + margin,
            rgb_rows * (rgb_height + title_height + margin)
            + panel_height
            + title_height
            + legend_height
            + 2 * margin,
        ),
        (250, 250, 250),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, path) in enumerate(rgb_panels):
        image = Image.open(path).convert("RGB")
        image.thumbnail((args.panel_width, rgb_height), Image.Resampling.LANCZOS)
        row, col = divmod(index, len(panels))
        origin_x = margin + col * (args.panel_width + margin)
        origin_y = margin + row * (rgb_height + title_height + margin) + title_height
        canvas.paste(image, (origin_x + (args.panel_width - image.width) // 2, origin_y))
        draw.rectangle(
            (origin_x - 1, origin_y - 1, origin_x + args.panel_width, origin_y + rgb_height),
            outline=(75, 82, 90),
            width=1,
        )
        draw.text((origin_x, origin_y - title_height + 8), label, fill=(25, 31, 38))
    map_top = margin + rgb_rows * (rgb_height + title_height + margin)
    for index, (label, occupancy, summary) in enumerate(panels):
        panel = Image.fromarray(occupancy_rgb(occupancy, reference), mode="RGB").resize(
            (args.panel_width, panel_height), Image.Resampling.NEAREST
        )
        origin = (margin + index * (args.panel_width + margin), map_top + title_height)
        canvas.paste(panel, origin)
        draw.rectangle(
            (origin[0] - 1, origin[1] - 1, origin[0] + args.panel_width, origin[1] + panel_height),
            outline=(75, 82, 90),
            width=1,
        )
        draw.text((origin[0], map_top + 6), label, fill=(25, 31, 38))
        draw.text((origin[0], map_top + 25), panel_status(summary), fill=(86, 94, 102))
        draw_trajectory(
            draw,
            origin=origin,
            panel_size=(args.panel_width, panel_height),
            crop=(row_slice, col_slice),
            reference_shape=reference.shape,
            summary=summary,
        )
        draw_acquisition_decisions(
            draw,
            origin=origin,
            panel_size=(args.panel_width, panel_height),
            crop=(row_slice, col_slice),
            reference_shape=reference.shape,
            summary=summary,
        )
        draw_world_markers(
            draw,
            origin=origin,
            panel_size=(args.panel_width, panel_height),
            crop=(row_slice, col_slice),
            reference_shape=reference.shape,
            summary=summary,
        )
    legend_x = margin
    legend_y = map_top + title_height + panel_height + 5
    for label, color in LEGEND:
        draw.rectangle(
            (legend_x, legend_y, legend_x + 10, legend_y + 10), fill=color, outline=(75, 82, 90)
        )
        draw.text((legend_x + 15, legend_y - 2), label, fill=(25, 31, 38))
        legend_x += int(15 + draw.textlength(label) + 18)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)


if __name__ == "__main__":
    main()
