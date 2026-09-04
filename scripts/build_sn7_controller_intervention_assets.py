#!/usr/bin/env python3
"""Build tables, claim gates, and plots for controller robustness interventions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from statistics import fmean
from typing import Any


METRICS = (
    ("raster_iou_auc", "IoU"),
    ("false_edit_auc", "False edit"),
    ("missed_edit_auc", "Missed edit"),
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    test_assets_read = payload.get(
        "test_assets_read",
        payload.get("protocol", {}).get("test_assets_read"),
    )
    if test_assets_read is not False:
        raise ValueError(f"expected validation-only artifact: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _condition(value: str) -> tuple[str, Path]:
    label, separator, root = value.partition("=")
    if not separator or not label or not root:
        raise argparse.ArgumentTypeError("condition must use LABEL=ROOT")
    return label, Path(root)


def _slug(value: str) -> str:
    result = "".join(character if character.isalnum() or character in "-_." else "_" for character in value)
    if not result or result in {".", ".."}:
        raise ValueError(f"unsafe condition label: {value}")
    return result


def build(conditions: list[tuple[str, Path]]) -> dict[str, Any]:
    if len(conditions) < 2:
        raise ValueError("at least two intervention conditions are required")
    labels = [label for label, _ in conditions]
    if len(set(labels)) != len(labels):
        raise ValueError("intervention labels must be unique")

    rows = []
    for label, root in conditions:
        notool_path = root / "benefit_vs_notool.json"
        forced_path = root / "benefit_vs_forced.json"
        controller_path = root / "controller_summary.json"
        notool = _load(notool_path)
        forced = _load(forced_path)
        controller = _load(controller_path)
        benefit_rows = [
            row for row in controller["per_seed"] if row["variant"] == "benefit"
        ]
        if not benefit_rows:
            raise ValueError(f"missing benefit controller rows: {controller_path}")
        row: dict[str, Any] = {
            "condition": label,
            "tool_call_episode_rate": fmean(
                float(item["tool_call_episode_rate"]) for item in benefit_rows
            ),
            "mean_tool_calls": fmean(
                float(item["mean_tool_calls"]) for item in benefit_rows
            ),
            "inputs": {
                "benefit_vs_notool": {
                    "path": str(notool_path.resolve()),
                    "sha256": _sha256(notool_path),
                },
                "benefit_vs_forced": {
                    "path": str(forced_path.resolve()),
                    "sha256": _sha256(forced_path),
                },
                "controller": {
                    "path": str(controller_path.resolve()),
                    "sha256": _sha256(controller_path),
                },
            },
        }
        for metric, _ in METRICS:
            row[f"{metric}_vs_notool"] = notool["paired_delta"][metric]
        row["spent_cost_auc_vs_forced"] = forced["paired_delta"]["spent_cost_auc"]
        quality = row["raster_iou_auc_vs_notool"]
        false_edit = row["false_edit_auc_vs_notool"]
        cost = row["spent_cost_auc_vs_forced"]
        row["checks"] = {
            "quality_ci_positive": float(quality["ci95_low"]) > 0.0,
            "false_edit_ci_negative": float(false_edit["ci95_high"]) < 0.0,
            "cost_vs_forced_ci_negative": float(cost["ci95_high"]) < 0.0,
        }
        row["checks"]["strict_gate_passed"] = all(row["checks"].values())
        rows.append(row)

    return {
        "schema_version": "sn7-controller-intervention-assets-v1",
        "rows": rows,
        "all_conditions_strict": all(row["checks"]["strict_gate_passed"] for row in rows),
        "protocol": {
            "split": "val",
            "frozen_controller": True,
            "severity_specific_recalibration": False,
            "test_assets_read": False,
        },
    }


def _write_csv(payload: dict[str, Any], output: Path) -> None:
    fields = (
        "condition",
        "tool_call_episode_rate",
        "mean_tool_calls",
        "iou_delta",
        "iou_ci95_low",
        "iou_ci95_high",
        "false_edit_delta",
        "false_edit_ci95_low",
        "false_edit_ci95_high",
        "cost_vs_forced_delta",
        "cost_vs_forced_ci95_low",
        "cost_vs_forced_ci95_high",
        "strict_gate_passed",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["rows"]:
            quality = row["raster_iou_auc_vs_notool"]
            false_edit = row["false_edit_auc_vs_notool"]
            cost = row["spent_cost_auc_vs_forced"]
            writer.writerow(
                {
                    "condition": row["condition"],
                    "tool_call_episode_rate": row["tool_call_episode_rate"],
                    "mean_tool_calls": row["mean_tool_calls"],
                    "iou_delta": quality["delta"],
                    "iou_ci95_low": quality["ci95_low"],
                    "iou_ci95_high": quality["ci95_high"],
                    "false_edit_delta": false_edit["delta"],
                    "false_edit_ci95_low": false_edit["ci95_low"],
                    "false_edit_ci95_high": false_edit["ci95_high"],
                    "cost_vs_forced_delta": cost["delta"],
                    "cost_vs_forced_ci95_low": cost["ci95_low"],
                    "cost_vs_forced_ci95_high": cost["ci95_high"],
                    "strict_gate_passed": row["checks"]["strict_gate_passed"],
                }
            )


def _interval(item: dict[str, float]) -> str:
    return (
        f"{float(item['delta']):+.6f} "
        f"[{float(item['ci95_low']):+.6f},{float(item['ci95_high']):+.6f}]"
    )


def _write_markdown(payload: dict[str, Any], output: Path) -> None:
    lines = [
        "| Condition | Call rate | IoU vs no-tool (95% CI) | "
        "False edit vs no-tool (95% CI) | Cost vs forced (95% CI) | Strict gate |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['condition']} | {row['tool_call_episode_rate']:.4f} | "
            f"{_interval(row['raster_iou_auc_vs_notool'])} | "
            f"{_interval(row['false_edit_auc_vs_notool'])} | "
            f"{_interval(row['spent_cost_auc_vs_forced'])} | "
            f"{'PASS' if row['checks']['strict_gate_passed'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "Validation only. Strict passage requires positive IoU CI, negative "
            "false-edit CI, and negative cost-vs-forced CI for the same condition.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def _write_latex(payload: dict[str, Any], output: Path) -> None:
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Condition & Call rate & IoU $\\Delta$ & False-edit $\\Delta$ & "
        "Cost vs. forced $\\Delta$ \\\\",
        "\\midrule",
    ]
    for row in payload["rows"]:
        lines.append(
            f"{row['condition']} & {row['tool_call_episode_rate']:.4f} & "
            f"{float(row['raster_iou_auc_vs_notool']['delta']):+.6f} & "
            f"{float(row['false_edit_auc_vs_notool']['delta']):+.6f} & "
            f"{float(row['spent_cost_auc_vs_forced']['delta']):+.6f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def _plot(payload: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["condition"] for row in payload["rows"]]
    specs = (
        ("raster_iou_auc_vs_notool", "IoU delta vs no-tool", "#2A9D8F"),
        ("false_edit_auc_vs_notool", "False-edit delta vs no-tool", "#E45756"),
        ("spent_cost_auc_vs_forced", "Cost delta vs forced", "#4C78A8"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.75))
    for axis, (key, title, color) in zip(axes, specs):
        items = [row[key] for row in payload["rows"]]
        values = [float(item["delta"]) for item in items]
        errors = [
            [value - float(item["ci95_low"]) for value, item in zip(values, items)],
            [float(item["ci95_high"]) - value for value, item in zip(values, items)],
        ]
        axis.bar(labels, values, color=color, width=0.58)
        axis.errorbar(
            range(len(labels)),
            values,
            yerr=errors,
            fmt="none",
            color="#222222",
            capsize=3,
            linewidth=1.0,
        )
        axis.axhline(0.0, color="#222222", linewidth=0.8)
        axis.set_title(title, fontsize=9)
        axis.grid(axis="y", color="#D9DDE0", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=400, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--condition", action="append", type=_condition, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    work_dir = args.output_dir.with_name(f"{args.output_dir.name}.tmp")
    if work_dir.exists():
        if work_dir.parent != args.output_dir.parent or not work_dir.name.endswith(".tmp"):
            raise RuntimeError(f"unsafe temporary output path: {work_dir}")
        shutil.rmtree(work_dir)
    payload = build(args.condition)
    work_dir.mkdir(parents=True)
    used_slugs: set[str] = set()
    for row in payload["rows"]:
        slug = _slug(str(row["condition"]))
        if slug in used_slugs:
            raise ValueError(f"condition labels collide after path normalization: {slug}")
        used_slugs.add(slug)
        input_dir = work_dir / "inputs" / slug
        input_dir.mkdir(parents=True)
        for name, metadata in row["inputs"].items():
            source = Path(metadata["path"])
            destination = input_dir / f"{name}.json"
            shutil.copy2(source, destination)
            metadata["bundle_path"] = destination.relative_to(work_dir).as_posix()
    (work_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (work_dir / "claim_gate.json").write_text(
        json.dumps(
            {
                "schema_version": "sn7-controller-intervention-claim-gate-v1",
                "checks": [
                    {"condition": row["condition"], **row["checks"]}
                    for row in payload["rows"]
                ],
                "all_conditions_strict": payload["all_conditions_strict"],
                "test_assets_read": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(payload, work_dir / "table.csv")
    _write_markdown(payload, work_dir / "table.md")
    _write_latex(payload, work_dir / "table.tex")
    _plot(payload, work_dir / "intervention_summary.png")
    files = sorted(path for path in work_dir.rglob("*") if path.is_file())
    (work_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "sn7-controller-intervention-manifest-v1",
                "files": {
                    path.relative_to(work_dir).as_posix(): {
                        "sha256": _sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in files
                },
                "test_assets_read": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    work_dir.replace(args.output_dir)


if __name__ == "__main__":
    main()
