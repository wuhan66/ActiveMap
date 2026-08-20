"""Dependency-light training curve rendering from epoch records."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

COLORS = ("#1565C0", "#D32F2F", "#2E7D32", "#6A1B9A", "#EF6C00", "#00838F")


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    series: list[tuple[str, list[float]]],
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline="#BDBDBD", width=1)
    draw.text((left + 8, top + 6), title, fill="#212121")
    plot_left, plot_top = left + 52, top + 30
    plot_right, plot_bottom = right - 12, bottom - 34
    values = [value for _, rows in series for value in rows if math.isfinite(value)]
    if not values:
        draw.text((plot_left, plot_top), "no values", fill="#757575")
        return
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        minimum -= 0.5
        maximum += 0.5
    padding = 0.05 * (maximum - minimum)
    minimum -= padding
    maximum += padding
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#616161", width=1)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#616161", width=1)
    draw.text((left + 4, plot_top - 4), f"{maximum:.3g}", fill="#616161")
    draw.text((left + 4, plot_bottom - 8), f"{minimum:.3g}", fill="#616161")
    max_steps = max(len(rows) for _, rows in series)
    for series_index, (label, rows) in enumerate(series):
        color = COLORS[series_index % len(COLORS)]
        points: list[tuple[float, float]] = []
        for index, value in enumerate(rows):
            if not math.isfinite(value):
                continue
            x = plot_left + (plot_right - plot_left) * index / max(max_steps - 1, 1)
            y = plot_bottom - (plot_bottom - plot_top) * (value - minimum) / (maximum - minimum)
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        elif points:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        legend_x = plot_left + (series_index % 3) * 150
        legend_y = plot_bottom + 10 + (series_index // 3) * 13
        draw.line((legend_x, legend_y + 5, legend_x + 16, legend_y + 5), fill=color, width=2)
        draw.text((legend_x + 20, legend_y), label[:20], fill="#424242")
    draw.text((plot_right - 60, plot_bottom + 10), f"epoch {max_steps}", fill="#616161")


def _values(records: list[dict[str, Any]], split: str, metric: str) -> list[float]:
    return [float(record.get(split, {}).get(metric, float("nan"))) for record in records]


def render_training_curves(records: list[dict[str, Any]], output_path: Path) -> None:
    if not records:
        return
    panels = [
        (
            "Total objective",
            [
                ("train/loss", _values(records, "train", "loss")),
                ("val/loss", _values(records, "val", "loss")),
                ("val/segmentation", _values(records, "val", "loss_segmentation")),
            ],
        ),
        (
            "Segmentation components",
            [
                ("BCE", _values(records, "val", "loss_segmentation_bce")),
                ("Dice", _values(records, "val", "loss_segmentation_dice")),
                ("Focal", _values(records, "val", "loss_segmentation_focal")),
                ("clDice", _values(records, "val", "loss_segmentation_cldice")),
            ],
        ),
        (
            "Edit and safety losses",
            [
                ("edit", _values(records, "val", "loss_edit")),
                ("geometry", _values(records, "val", "loss_geometry")),
                ("false edit", _values(records, "val", "loss_false_edit")),
                ("missed edit", _values(records, "val", "loss_missed_edit")),
                ("confidence", _values(records, "val", "loss_confidence")),
            ],
        ),
        (
            "Validation metrics",
            [
                ("IoU", _values(records, "val", "iou")),
                ("edit accuracy", _values(records, "val", "edit_accuracy")),
                ("false edit rate", _values(records, "val", "false_edit_rate")),
                ("missed edit rate", _values(records, "val", "missed_edit_rate")),
            ],
        ),
    ]
    canvas = Image.new("RGB", (1400, 900), "white")
    draw = ImageDraw.Draw(canvas)
    boxes = ((20, 20, 690, 430), (710, 20, 1380, 430), (20, 450, 690, 880), (710, 450, 1380, 880))
    for panel, box in zip(panels, boxes, strict=True):
        _draw_panel(draw, box, panel[0], panel[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    canvas.save(temporary, format="PNG")
    temporary.replace(output_path)


def render_run_comparison(
    runs: dict[str, list[dict[str, Any]]], output_path: Path
) -> None:
    """Render validation trajectories from several updater runs on shared panels."""

    if not runs:
        return
    panel_metrics = (
        ("Validation loss", "loss"),
        ("Validation raster IoU", "iou"),
        ("Validation DELETE recall", "delete_recall"),
        ("Validation false-edit rate", "false_edit_rate"),
    )
    canvas = Image.new("RGB", (1400, 900), "white")
    draw = ImageDraw.Draw(canvas)
    boxes = ((20, 20, 690, 430), (710, 20, 1380, 430), (20, 450, 690, 880), (710, 450, 1380, 880))
    for (title, metric), box in zip(panel_metrics, boxes, strict=True):
        series = [(name, _values(records, "val", metric)) for name, records in runs.items()]
        _draw_panel(draw, box, title, series)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    canvas.save(temporary, format="PNG")
    temporary.replace(output_path)
