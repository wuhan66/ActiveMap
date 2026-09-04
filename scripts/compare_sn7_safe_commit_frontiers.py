#!/usr/bin/env python3
"""Compare matched quality-safety operating points across frozen backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FALSE_EDIT_CAPS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30)
MISSED_EDIT_CAPS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def read_frontier(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("test_assets_read") is not False:
        raise ValueError(f"frontier is not test-free: {path}")
    if not payload.get("points"):
        raise ValueError(f"frontier is empty: {path}")
    return payload


def flatten(point: dict[str, Any]) -> dict[str, float]:
    return {
        "threshold": float(point["threshold"]),
        **{
            name: float(values["mean"])
            for name, values in point["metrics"].items()
        },
    }


def best_under(
    points: list[dict[str, float]], metric: str, cap: float
) -> dict[str, float] | None:
    eligible = [point for point in points if point[metric] <= cap + 1e-12]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda point: (
            point["map_iou_delta"],
            -point["false_edit_rate"],
            -point["missed_edit_rate"],
            point["accepted_commit_rate"],
        ),
    )


def pareto(points: list[dict[str, float]]) -> list[dict[str, float]]:
    result = []
    for candidate in points:
        dominated = any(
            other["map_iou_delta"] >= candidate["map_iou_delta"]
            and other["false_edit_rate"] <= candidate["false_edit_rate"]
            and other["missed_edit_rate"] <= candidate["missed_edit_rate"]
            and (
                other["map_iou_delta"] > candidate["map_iou_delta"]
                or other["false_edit_rate"] < candidate["false_edit_rate"]
                or other["missed_edit_rate"] < candidate["missed_edit_rate"]
            )
            for other in points
        )
        if not dominated:
            result.append(candidate)
    return sorted(result, key=lambda point: point["threshold"])


def compare(frontiers: dict[str, Path]) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for name, path in frontiers.items():
        payload = read_frontier(path)
        points = [flatten(point) for point in payload["points"]]
        methods[name] = {
            "source": str(path),
            "selected_thresholds": payload["selected_thresholds"],
            "best_map_iou_point": max(
                points, key=lambda point: point["map_iou_delta"]
            ),
            "matched_false_edit": {
                str(cap): best_under(points, "false_edit_rate", cap)
                for cap in FALSE_EDIT_CAPS
            },
            "matched_missed_edit": {
                str(cap): best_under(points, "missed_edit_rate", cap)
                for cap in MISSED_EDIT_CAPS
            },
            "pareto_points": pareto(points),
            "points": points,
        }
    return {
        "schema_version": "sn7-safe-commit-frontier-comparison-v1",
        "selection_note": (
            "The 101-point validation grid is diagnostic only; primary operating "
            "points remain source-train OOF calibrated."
        ),
        "false_edit_caps": FALSE_EDIT_CAPS,
        "missed_edit_caps": MISSED_EDIT_CAPS,
        "methods": methods,
        "test_assets_read": False,
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "| Method | Best map-IoU gain | False edit | Missed edit | Threshold |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, method in result["methods"].items():
        point = method["best_map_iou_point"]
        lines.append(
            f"| {name} | {point['map_iou_delta']:+.6f} | "
            f"{point['false_edit_rate']:.6f} | {point['missed_edit_rate']:.6f} | "
            f"{point['threshold']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Matched false-edit cap = 0.01:",
            "",
            "| Method | Map-IoU gain | False edit | Missed edit | Threshold |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, method in result["methods"].items():
        point = method["matched_false_edit"]["0.01"]
        if point is None:
            lines.append(f"| {name} | n/a | n/a | n/a | n/a |")
        else:
            lines.append(
                f"| {name} | {point['map_iou_delta']:+.6f} | "
                f"{point['false_edit_rate']:.6f} | "
                f"{point['missed_edit_rate']:.6f} | {point['threshold']:.2f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    display_names = {
        "changemamba_self": "ChangeMamba (self-cal.)",
        "ban_self": "BAN (self-cal.)",
        "changemamba_to_ban": "ChangeMamba gate -> BAN",
        "ban_to_changemamba": "BAN gate -> ChangeMamba",
    }
    for name, method in result["methods"].items():
        points = method["points"]
        axes[0].plot(
            [point["false_edit_rate"] for point in points],
            [point["map_iou_delta"] for point in points],
            marker=".",
            markersize=3,
            label=display_names.get(name, name),
        )
        axes[1].plot(
            [point["missed_edit_rate"] for point in points],
            [point["map_iou_delta"] for point in points],
            marker=".",
            markersize=3,
            label=display_names.get(name, name),
        )
    axes[0].set_xlabel("False-edit rate")
    axes[1].set_xlabel("Missed-edit rate")
    for axis in axes:
        axis.set_ylabel("Map-IoU gain")
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--frontier", nargs=2, action="append", metavar=("NAME", "PATH"))
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--output-figure", type=Path)
    args = parser.parse_args()
    if not args.frontier or len(args.frontier) < 2:
        raise ValueError("at least two named frontiers are required")
    frontiers = {name: Path(path) for name, path in args.frontier}
    if len(frontiers) != len(args.frontier):
        raise ValueError("frontier names must be unique")
    result = compare(frontiers)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if args.output_markdown is not None:
        write_markdown(result, args.output_markdown)
    if args.output_figure is not None:
        plot(result, args.output_figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
