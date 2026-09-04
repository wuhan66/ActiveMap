#!/usr/bin/env python3
"""Plot compact paper-ready training diagnostics for SN7 updater runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise argparse.ArgumentTypeError(f"invalid run name: {name}")
    return name, Path(raw_path)


def _read_history(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty history: {path}")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, len(rows) + 1)):
        raise ValueError(f"non-contiguous epochs: {path}")
    return rows


def plot_histories(
    histories: list[tuple[str, Path]], output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = {name: _read_history(path) for name, path in histories}
    colors = ("#1769aa", "#d1495b", "#2a9d63", "#7b61a8", "#e08e0b")
    plots = (
        ("train_loss", "Training loss", lambda row: row["train_loss"]),
        (
            "committed_map_iou",
            "Validation committed map IoU",
            lambda row: row["val"]["committed_map_iou"],
        ),
        (
            "map_iou_delta",
            "Validation map IoU gain",
            lambda row: row["val"]["map_iou_delta"],
        ),
        (
            "change_iou",
            "Validation change IoU",
            lambda row: row["val"]["change_iou"],
        ),
    )
    output_files = []
    for key, ylabel, accessor in plots:
        figure, axis = plt.subplots(figsize=(5.6, 3.5), constrained_layout=True)
        for index, (name, rows) in enumerate(values.items()):
            axis.plot(
                [int(row["epoch"]) for row in rows],
                [float(accessor(row)) for row in rows],
                label=name,
                color=colors[index % len(colors)],
                linewidth=2.0,
                marker="o",
                markersize=3.0,
            )
        if key == "map_iou_delta":
            axis.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
        path = output_dir / f"{key}.png"
        figure.savefig(path, dpi=300, facecolor="white")
        plt.close(figure)
        output_files.append(path.name)
    summary = {
        "schema_version": "sn7-updater-training-plots-v1",
        "histories": {name: str(path) for name, path in histories},
        "outputs": output_files,
        "test_assets_read": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--history",
        action="append",
        type=_parse_named_path,
        required=True,
        help="Repeatable NAME=PATH history JSONL.",
    )
    args = parser.parse_args()
    result = plot_histories(args.history, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
