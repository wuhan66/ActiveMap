#!/usr/bin/env python3
"""Plot comparable training diagnostics for SN7 concat U-Net seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(raw_path)


def _load(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty history: {path}")
    return rows


def plot(histories: list[tuple[str, Path]], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "savefig.dpi": 400,
        }
    )
    data = {name: _load(path) for name, path in histories}
    specs = (
        ("loss", "Objective", lambda row: row["val"]["loss"]),
        ("iou", "Raster IoU", lambda row: row["val"]["iou"]),
        ("edit_accuracy", "Edit accuracy", lambda row: row["val"]["edit_accuracy"]),
        ("false_edit_rate", "False-edit rate", lambda row: row["val"]["false_edit_rate"]),
        ("delete_recall", "DELETE recall", lambda row: row["val"]["delete_recall"]),
        ("learning_rate", "Learning rate", lambda row: row["learning_rate"]),
    )
    colors = ("#008F83", "#4472A8", "#D95F4A")
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 4.7))
    axes_flat = list(axes.flat)
    for axis, (key, title, accessor) in zip(axes_flat, specs, strict=False):
        for index, (name, rows) in enumerate(data.items()):
            axis.plot(
                [int(row["epoch"]) for row in rows],
                [float(accessor(row)) for row in rows],
                color=colors[index % len(colors)],
                linewidth=1.5,
                marker="o",
                markersize=2.5,
                label=name,
            )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Epoch")
        axis.grid(axis="y", color="#D9DDE0", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if key == "loss":
            axis.legend(frameon=False, fontsize=7.5)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.1, top=0.95, wspace=0.3, hspace=0.38)
    outputs = []
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"sn7_concat_unet_training.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(fig)
    payload = {
        "schema_version": "sn7-concat-unet-training-figure-v2",
        "split": "val",
        "test_assets_read": False,
        "histories": {name: str(path.resolve()) for name, path in histories},
        "outputs": [path.name for path in outputs],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--history", action="append", type=_named_path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.history, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
