#!/usr/bin/env python3
"""Plot cross-region active-selection and Safe Commit boundary heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


BACKEND_ORDER = {"changeformer": 0, "ban": 1}


def region_order(region: str) -> tuple[int, int]:
    if region == "primary":
        return (0, 0)
    if region.startswith("rank") and region[4:].isdigit():
        return (1, int(region[4:]))
    return (2, 0)


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in root.glob("*.json"):
        region, backend = path.stem.rsplit("_", 1)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["protocol"].get("test_assets_read") is not False:
            raise ValueError(f"test-access audit failed: {path}")
        selection = payload["ambiguous_selection"]
        terminal = payload["terminal_all_validation"]
        first = float(selection["first"]["mean_map_iou"])
        learned = float(selection["learned_ranker"]["mean_map_iou"])
        random_expected = float(selection["random_expected"]["mean_map_iou"])
        oracle = float(selection["oracle"]["mean_map_iou"])
        available_gain = oracle - first
        safe = terminal["safe_commit"]
        commit = terminal["always_commit"]
        rows.append({
            "region": region,
            "backend": backend,
            "episodes": int(selection["first"]["episodes"]),
            "learned_minus_first": learned - first,
            "learned_minus_random": learned - random_expected,
            "oracle_gain_recovered": (
                (learned - first) / available_gain if available_gain >= 0.01 else np.nan
            ),
            "safe_iou_gain_vs_commit": float(safe["mean_map_iou"]) - float(commit["mean_map_iou"]),
            "safe_false_edit_reduction": float(commit["false_edit_rate"]) - float(safe["false_edit_rate"]),
            "safe_missed_edit_reduction": float(commit["missed_edit_rate"]) - float(safe["missed_edit_rate"]),
        })
    if not rows or len(rows) % len(BACKEND_ORDER):
        raise ValueError(f"expected complete backend pairs, found {len(rows)} summaries")
    region_backends: dict[str, set[str]] = {}
    for row in rows:
        region_backends.setdefault(row["region"], set()).add(row["backend"])
    incomplete = {
        region: sorted(set(BACKEND_ORDER) - backends)
        for region, backends in region_backends.items()
        if backends != set(BACKEND_ORDER)
    }
    if incomplete:
        raise ValueError(f"incomplete region/backend support: {incomplete}")
    return sorted(rows, key=lambda row: (region_order(row["region"]), BACKEND_ORDER[row["backend"]]))


def render_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    output: Path,
    *,
    percentage_columns: set[int] | None = None,
    normalize_columns: bool = False,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    percentage_columns = percentage_columns or set()
    image_matrix = matrix.copy()
    if normalize_columns:
        for column in range(image_matrix.shape[1]):
            values = image_matrix[:, column]
            finite_column = values[np.isfinite(values)]
            scale = float(np.max(np.abs(finite_column))) if finite_column.size else 1.0
            if scale > 0:
                image_matrix[:, column] = values / scale
    finite = image_matrix[np.isfinite(image_matrix)]
    bound = max(float(np.max(np.abs(finite))), 1e-6)
    fig_height = max(4.2, 0.48 * len(row_labels) + 1.5)
    fig, axis = plt.subplots(figsize=(8.4, fig_height), constrained_layout=True)
    image = axis.imshow(
        image_matrix,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
        aspect="auto",
    )
    axis.set_xticks(np.arange(len(column_labels)), labels=column_labels)
    axis.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    axis.tick_params(axis="x", rotation=12)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if not np.isfinite(value):
                label = "n/a"
            elif column in percentage_columns:
                label = f"{value * 100:+.1f}%"
            else:
                label = f"{value:+.3f}"
            image_value = image_matrix[row, column]
            color = "white" if abs(image_value) > 0.55 * bound else "black"
            axis.text(column, row, label, ha="center", va="center", color=color, fontsize=9)
    for edge in np.arange(1.5, len(row_labels) - 0.5, 2.0):
        axis.axhline(edge, color="white", linewidth=2.2)
    axis.spines[:].set_visible(False)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label(
        "Per-metric normalized direction" if normalize_columns else "Direction-corrected delta"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input)

    fields = list(rows[0])
    with (args.output / "spacenet8_cross_region_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "spacenet8_cross_region_metrics.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )

    row_labels = [
        f"{row['region']} / {'ChangeFormer' if row['backend'] == 'changeformer' else 'BAN-MiT'}"
        for row in rows
    ]
    selection_fields = (
        "learned_minus_first",
        "learned_minus_random",
        "oracle_gain_recovered",
    )
    render_heatmap(
        np.asarray([[row[field] for field in selection_fields] for row in rows]),
        row_labels,
        ["Learned - first", "Learned - random", "Oracle recovery (gap >= .01)"],
        args.output / "spacenet8_active_selection_boundaries",
        percentage_columns={2},
        normalize_columns=True,
    )

    safety_fields = (
        "safe_iou_gain_vs_commit",
        "safe_false_edit_reduction",
        "safe_missed_edit_reduction",
    )
    render_heatmap(
        np.asarray([[row[field] for field in safety_fields] for row in rows]),
        row_labels,
        ["Map IoU gain", "False-edit reduction", "Missed-edit reduction"],
        args.output / "spacenet8_safe_commit_boundaries",
    )


if __name__ == "__main__":
    main()
