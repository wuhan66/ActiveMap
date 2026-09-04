#!/usr/bin/env python3
"""Build a protocol-aligned MUNO21 controller and executable-writeback table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "method",
    "budget",
    "sample_count",
    "terminal_accuracy",
    "false_edit_rate",
    "missed_edit_rate",
    "mean_acquisitions",
    "mean_cost",
    "mean_quality_cost_utility",
    "mean_raster_iou",
    "mean_prior_raster_iou",
    "mean_raster_iou_gain",
    "mean_episode_utility_v2_balanced",
    "mean_episode_utility_v2_safety",
    "mean_episode_utility_v2_cost_aware",
)


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected METHOD=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected METHOD=PATH")
    return name, Path(raw_path)


def _assert_validation_only(data: dict[str, Any], path: Path) -> None:
    protocol = data.get("protocol", {})
    test_assets_read = data.get("test_assets_read", protocol.get("test_assets_read"))
    if test_assets_read is not False:
        raise ValueError(f"missing validation-only provenance in {path}")
    split = data.get("split", protocol.get("split"))
    if split is not None and split != "val":
        raise ValueError(f"expected validation split in {path}, got {split!r}")


def _read_rollouts(paths: list[Path]) -> tuple[dict[tuple[str, float], dict[str, Any]], list[dict[str, str]]]:
    rows: dict[tuple[str, float], dict[str, Any]] = {}
    sources: list[dict[str, str]] = []
    for path in paths:
        data = _load(path)
        _assert_validation_only(data, path)
        sources.append({"path": str(path), "sha256": _sha256(path)})
        for result in data.get("results", []):
            key = (str(result["method"]), float(result["budget"]))
            if key in rows:
                raise ValueError(f"duplicate rollout row {key} from {path}")
            rows[key] = dict(result)
    return rows, sources


def _read_closed_loop(
    specifications: list[tuple[str, Path]],
    budgets: list[float],
) -> tuple[dict[tuple[str, float], dict[str, Any]], list[dict[str, str]]]:
    rows: dict[tuple[str, float], dict[str, Any]] = {}
    sources: list[dict[str, str]] = []
    for method, path in specifications:
        data = _load(path)
        _assert_validation_only(data, path)
        metrics = data["metrics"]
        sources.append({"path": str(path), "sha256": _sha256(path)})
        for budget in budgets:
            rows[(method, budget)] = {
                "method": method,
                "budget": budget,
                "sample_count": int(data["sample_count"]) // len(budgets),
                **metrics,
            }
    return rows, sources


def _read_writebacks(
    specifications: list[tuple[str, Path]],
) -> tuple[dict[tuple[str, float], dict[str, Any]], list[dict[str, str]]]:
    rows: dict[tuple[str, float], dict[str, Any]] = {}
    sources: list[dict[str, str]] = []
    for method, path in specifications:
        data = _load(path)
        _assert_validation_only(data, path)
        sources.append({"path": str(path), "sha256": _sha256(path)})
        for result in data["budgets"]:
            key = (method, float(result["budget"]))
            if key in rows:
                raise ValueError(f"duplicate writeback row {key} from {path}")
            rows[key] = dict(result)
    return rows, sources


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in FIELDS} for row in rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = (
        "Method",
        "Budget",
        "Acc.",
        "False edit",
        "Missed edit",
        "Acq.",
        "Cost",
        "Q-C utility",
        "Map IoU",
        "Map IoU gain",
        "Utility-v2",
    )
    lines = [
        "# MUNO21 Validation Baselines",
        "",
        "All rows use the frozen validation split and executable vector-map writeback.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for row in rows:
        values = (
            row["method"],
            _fmt(row["budget"], 1),
            _fmt(row.get("terminal_accuracy")),
            _fmt(row.get("false_edit_rate")),
            _fmt(row.get("missed_edit_rate")),
            _fmt(row.get("mean_acquisitions")),
            _fmt(row.get("mean_cost")),
            _fmt(row.get("mean_quality_cost_utility")),
            _fmt(row.get("mean_raster_iou")),
            _fmt(row.get("mean_raster_iou_gain")),
            _fmt(row.get("mean_episode_utility_v2_balanced")),
        )
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Controller metrics are merged with the matched executable-writeback metrics.",
            "Utility-v2 is the predeclared balanced profile; safety and cost-aware profiles",
            "remain available in the CSV and JSON exports.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r"Method & Budget & Acc. & FE & ME & Acq. & Cost & Q-C util. & Map IoU & $\Delta$IoU & Util.-v2 \\",
        r"\midrule",
    ]
    for row in rows:
        method = str(row["method"]).replace("_", r"\_")
        values = [
            method,
            _fmt(row["budget"], 1),
            _fmt(row.get("terminal_accuracy")),
            _fmt(row.get("false_edit_rate")),
            _fmt(row.get("missed_edit_rate")),
            _fmt(row.get("mean_acquisitions")),
            _fmt(row.get("mean_cost")),
            _fmt(row.get("mean_quality_cost_utility")),
            _fmt(row.get("mean_raster_iou")),
            _fmt(row.get("mean_raster_iou_gain")),
            _fmt(row.get("mean_episode_utility_v2_balanced")),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--rollout-summary", type=Path, action="append", default=[])
    parser.add_argument("--writeback", type=_named_path, action="append", default=[])
    parser.add_argument("--closed-loop", type=_named_path, action="append", default=[])
    parser.add_argument("--budget", type=float, action="append", default=[])
    args = parser.parse_args()

    budgets = sorted(set(args.budget or [1.5, 3.0, 4.5]))
    controller_rows, rollout_sources = _read_rollouts(args.rollout_summary)
    closed_loop_rows, closed_loop_sources = _read_closed_loop(args.closed_loop, budgets)
    for key, value in closed_loop_rows.items():
        if key in controller_rows:
            raise ValueError(f"duplicate controller row {key}")
        controller_rows[key] = value
    writeback_rows, writeback_sources = _read_writebacks(args.writeback)

    missing = sorted(set(controller_rows) ^ set(writeback_rows))
    if missing:
        raise ValueError(f"controller/writeback key mismatch: {missing}")

    rows: list[dict[str, Any]] = []
    for key in sorted(controller_rows, key=lambda item: (item[0], item[1])):
        controller = controller_rows[key]
        writeback = writeback_rows[key]
        rows.append(
            {
                "method": key[0],
                "budget": key[1],
                "sample_count": int(writeback["sample_count"]),
                "terminal_accuracy": controller["terminal_accuracy"],
                "false_edit_rate": controller["false_edit_rate"],
                "missed_edit_rate": controller["missed_edit_rate"],
                "mean_acquisitions": controller["mean_acquisitions"],
                "mean_cost": controller["mean_cost"],
                "mean_quality_cost_utility": controller["mean_quality_cost_utility"],
                "mean_raster_iou": writeback["mean_raster_iou"],
                "mean_prior_raster_iou": writeback["mean_prior_raster_iou"],
                "mean_raster_iou_gain": writeback["mean_raster_iou_gain"],
                "mean_episode_utility_v2_balanced": writeback[
                    "mean_episode_utility_v2_balanced"
                ],
                "mean_episode_utility_v2_safety": writeback[
                    "mean_episode_utility_v2_safety"
                ],
                "mean_episode_utility_v2_cost_aware": writeback[
                    "mean_episode_utility_v2_cost_aware"
                ],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(args.output_dir / "table.csv", rows)
    _write_markdown(args.output_dir / "table.md", rows)
    _write_latex(args.output_dir / "table.tex", rows)
    bundle = {
        "schema_version": "muno21-baseline-paper-table-v1",
        "split": "val",
        "budgets": budgets,
        "method_count": len({row["method"] for row in rows}),
        "row_count": len(rows),
        "rows": rows,
        "sources": {
            "rollout_summaries": rollout_sources,
            "closed_loop_summaries": closed_loop_sources,
            "writeback_summaries": writeback_sources,
        },
        "test_assets_read": False,
    }
    (args.output_dir / "table.json").write_text(
        json.dumps(bundle, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
