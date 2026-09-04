#!/usr/bin/env python3
"""Export submission-facing SN7 Step-0 validation tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


CONTROLLER_METRICS = (
    ("terminal_accuracy", "Terminal Acc."),
    ("false_edit_rate", "False Edit"),
    ("missed_edit_rate", "Missed Edit"),
    ("mean_quality_gain", "Quality Gain"),
    ("mean_quality_cost_utility", "Q-C Utility"),
    ("mean_tool_calls", "Tool Calls"),
    ("tool_call_episode_rate", "Call Rate"),
    ("mean_tool_belief_l1_delta", "Belief Delta"),
)

WRITEBACK_METRICS = (
    ("raster_iou_auc", "Raster IoU AUC"),
    ("raster_iou_gain_auc", "Raster Gain AUC"),
    ("false_edit_auc", "False Edit AUC"),
    ("missed_edit_auc", "Missed Edit AUC"),
    ("spent_cost_auc", "Cost AUC"),
    ("episode_utility_v2_balanced_auc", "Balanced Utility"),
    ("episode_utility_v2_safety_auc", "Safety Utility"),
    ("episode_utility_v2_cost_aware_auc", "Cost-aware Utility"),
    ("vector_replay_iou_auc", "Replay IoU"),
    ("vector_delta_topology_valid_auc", "Topology Valid"),
)

DISPLAY_NAMES = {
    "notool": "No tool",
    "forced": "Forced tools",
    "benefit": "ActiveMap (benefit-aware)",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_validation(payload: dict[str, Any], name: str) -> None:
    # Older aggregate receipts stored these fields under `protocol` while
    # retaining top-level null placeholders. Prefer a non-null top-level value
    # and fall back to the protocol record without weakening test-access checks.
    split = payload.get("split") or payload.get("protocol", {}).get("split")
    test_read = payload.get("test_assets_read")
    if test_read is None:
        test_read = payload.get("protocol", {}).get("test_assets_read")
    validation_claim = str(payload.get("claim_boundary", "")).startswith(
        "Validation-only"
    )
    if (split != "val" and not (split is None and validation_claim)) or test_read is not False:
        raise ValueError(f"{name} must be validation-only with test_assets_read=false")


def _seed_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
    }


def _controller_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("per_seed")
    if not isinstance(rows, list) or not rows:
        raise ValueError("controller summary has no per-seed rows")
    seeds = sorted(int(seed) for seed in summary["seeds"])
    output = []
    for variant in ("notool", "forced", "benefit"):
        selected = [row for row in rows if row.get("variant") == variant]
        if sorted(int(row["seed"]) for row in selected) != seeds:
            raise ValueError(f"{variant} does not have the frozen seed support")
        result: dict[str, Any] = {
            "method": DISPLAY_NAMES[variant],
            "variant": variant,
            "seed_count": len(selected),
        }
        for metric, _ in CONTROLLER_METRICS:
            result[metric] = _seed_stats([float(row[metric]) for row in selected])
        output.append(result)
    return output


def _same_number(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)


def _writeback_rows(
    versus_notool: dict[str, Any], versus_forced: dict[str, Any]
) -> list[dict[str, Any]]:
    benefit_a = versus_notool["candidate"]
    benefit_b = versus_forced["candidate"]
    for metric, _ in WRITEBACK_METRICS:
        if not _same_number(benefit_a[metric], benefit_b[metric]):
            raise ValueError(f"benefit aggregate differs across comparisons: {metric}")
    return [
        {
            "method": DISPLAY_NAMES["notool"],
            "variant": "notool",
            **{metric: versus_notool["baseline"][metric] for metric, _ in WRITEBACK_METRICS},
        },
        {
            "method": DISPLAY_NAMES["forced"],
            "variant": "forced",
            **{metric: versus_forced["baseline"][metric] for metric, _ in WRITEBACK_METRICS},
        },
        {
            "method": DISPLAY_NAMES["benefit"],
            "variant": "benefit",
            **{metric: benefit_a[metric] for metric, _ in WRITEBACK_METRICS},
        },
    ]


def _format(value: float) -> str:
    return f"{float(value):.6f}"


def _format_seed_stat(value: dict[str, float]) -> str:
    return f"{value['mean']:.6f} +/- {value['std']:.6f}"


def _markdown(
    title: str,
    rows: list[dict[str, Any]],
    metrics: tuple[tuple[str, str], ...],
    *,
    seed_stats: bool,
) -> str:
    labels = [label for _, label in metrics]
    lines = [
        f"# {title}",
        "",
        "| Method | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join("---:" for _ in labels) + " |",
    ]
    formatter = _format_seed_stat if seed_stats else _format
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [str(row["method"]), *(formatter(row[key]) for key, _ in metrics)]
            )
            + " |"
        )
    lines.extend(["", "Validation only; test assets were not read.", ""])
    return "\n".join(lines)


def _latex(
    rows: list[dict[str, Any]],
    metrics: tuple[tuple[str, str], ...],
    *,
    seed_stats: bool,
) -> str:
    formatter = _format_seed_stat if seed_stats else _format
    lines = [
        "\\begin{tabular}{l" + "r" * len(metrics) + "}",
        "\\toprule",
        "Method & " + " & ".join(label for _, label in metrics) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        method = str(row["method"]).replace("&", "\\&")
        lines.append(
            method
            + " & "
            + " & ".join(formatter(row[key]) for key, _ in metrics)
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    metrics: tuple[tuple[str, str], ...],
    *,
    seed_stats: bool,
) -> None:
    fields = ["method", "variant"]
    if seed_stats:
        fields.extend(
            field
            for metric, _ in metrics
            for field in (f"{metric}_mean", f"{metric}_std")
        )
    else:
        fields.extend(metric for metric, _ in metrics)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {"method": row["method"], "variant": row["variant"]}
            for metric, _ in metrics:
                if seed_stats:
                    flat[f"{metric}_mean"] = row[metric]["mean"]
                    flat[f"{metric}_std"] = row[metric]["std"]
                else:
                    flat[metric] = row[metric]
            writer.writerow(flat)


def export(
    summary_path: Path,
    versus_notool_path: Path,
    versus_forced_path: Path,
    promotion_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    inputs = {
        "controller_summary": summary_path,
        "benefit_vs_notool": versus_notool_path,
        "benefit_vs_forced": versus_forced_path,
        "promotion": promotion_path,
    }
    payloads = {name: _load(path) for name, path in inputs.items()}
    for name, payload in payloads.items():
        _assert_validation(payload, name)
    if payloads["promotion"].get("promote") is not True:
        raise ValueError("writeback promotion gate did not pass")

    expected_seeds = sorted(int(seed) for seed in payloads["controller_summary"]["seeds"])
    for name in ("benefit_vs_notool", "benefit_vs_forced"):
        seeds = sorted(int(seed) for seed in payloads[name]["model_seeds"])
        if seeds != expected_seeds:
            raise ValueError(f"{name} seed support differs from controller summary")

    controller = _controller_rows(payloads["controller_summary"])
    writeback = _writeback_rows(
        payloads["benefit_vs_notool"], payloads["benefit_vs_forced"]
    )
    paired = {
        "controller": payloads["controller_summary"]["comparisons"],
        "writeback": {
            "benefit_vs_notool": payloads["benefit_vs_notool"]["paired_delta"],
            "benefit_vs_forced": payloads["benefit_vs_forced"]["paired_delta"],
        },
    }

    output_dir.mkdir(parents=True)
    _write_csv(
        output_dir / "controller_table.csv",
        controller,
        CONTROLLER_METRICS,
        seed_stats=True,
    )
    _write_csv(
        output_dir / "writeback_table.csv",
        writeback,
        WRITEBACK_METRICS,
        seed_stats=False,
    )
    (output_dir / "controller_table.md").write_text(
        _markdown(
            "SN7 Step-0 Controller Results",
            controller,
            CONTROLLER_METRICS,
            seed_stats=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "writeback_table.md").write_text(
        _markdown(
            "SN7 Executable Writeback Results",
            writeback,
            WRITEBACK_METRICS,
            seed_stats=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "controller_table.tex").write_text(
        _latex(controller, CONTROLLER_METRICS, seed_stats=True), encoding="utf-8"
    )
    (output_dir / "writeback_table.tex").write_text(
        _latex(writeback, WRITEBACK_METRICS, seed_stats=False), encoding="utf-8"
    )
    (output_dir / "paired_intervals.json").write_text(
        json.dumps(paired, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "sn7-step0-paper-tables-v1",
        "split": "val",
        "test_assets_read": False,
        "seeds": expected_seeds,
        "promotion_passed": True,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "outputs": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("controller_summary", type=Path)
    parser.add_argument("benefit_vs_notool", type=Path)
    parser.add_argument("benefit_vs_forced", type=Path)
    parser.add_argument("promotion", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                args.controller_summary,
                args.benefit_vs_notool,
                args.benefit_vs_forced,
                args.promotion,
                args.output_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
