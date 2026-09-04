#!/usr/bin/env python3
"""Aggregate MUNO21 capacity and direct-VLM backbone controls."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("input must use GROUP=PATH")
    return label, Path(raw)


def read_metrics(path: Path) -> tuple[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "operation_metrics" in payload:
        return "direct_vlm", {
            "schema_valid_rate": float(payload["schema_valid_rate"]),
            "accuracy": float(payload["operation_metrics"]["accuracy"]),
            "macro_f1": float(payload["operation_metrics"]["macro_f1"]),
            "false_edit_rate": float(payload["false_edit_rate"]),
            "missed_edit_rate": float(payload["missed_edit_rate"]),
        }
    acquire = payload["per_action"].get("ACQUIRE", {})
    use_tool = payload["per_action"].get("USE_TOOL", {})
    return "structured_agent", {
        "schema_valid_rate": float(payload["schema_valid_rate"]),
        "accuracy": float(payload["exact_action_accuracy"]),
        "macro_f1": float(payload["macro_f1"]),
        "acquire_recall": float(acquire.get("recall", 0.0)),
        "use_tool_recall": float(use_tool.get("recall", 0.0)),
        "false_call_rate": float(payload["tool_metrics"]["false_call_rate"]),
    }


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--input", action="append", type=labeled_path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    protocols: dict[str, str] = {}
    for group, path in args.input:
        protocol, metrics = read_metrics(path)
        if group in protocols and protocols[group] != protocol:
            raise ValueError(f"mixed protocols in {group}")
        protocols[group] = protocol
        groups[group].append({"path": str(path.resolve()), "metrics": metrics})

    result_groups = {}
    for group, rows in groups.items():
        metric_names = rows[0]["metrics"].keys()
        result_groups[group] = {
            "protocol": protocols[group],
            "seed_count": len(rows),
            "runs": rows,
            "aggregate": {
                name: summarize([float(row["metrics"][name]) for row in rows])
                for name in metric_names
            },
        }
    payload = {
        "schema_version": "muno21-capacity-backbone-controls-v1",
        "split": "val",
        "test_assets_read": False,
        "groups": result_groups,
        "comparison_boundary": (
            "Compare groups only when protocol fields match; structured-agent and "
            "direct-VLM panels are separate controls."
        ),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = ["# MUNO21 Capacity and Backbone Controls", ""]
    for protocol in ("structured_agent", "direct_vlm"):
        lines.extend([f"## {protocol.replace('_', ' ').title()}", ""])
        matching = [(name, value) for name, value in result_groups.items() if value["protocol"] == protocol]
        if not matching:
            continue
        metrics = list(matching[0][1]["aggregate"])
        lines.append("| Group | Seeds | " + " | ".join(metrics) + " |")
        lines.append("| --- | ---: | " + " | ".join(["---:"] * len(metrics)) + " |")
        for name, value in matching:
            cells = [f"{value['aggregate'][metric]['mean']:.4f} +/- {value['aggregate'][metric]['standard_deviation']:.4f}" for metric in metrics]
            lines.append(f"| {name} | {value['seed_count']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend([
        "Structured-agent and direct-VLM rows use different action protocols and are not ranked against each other.",
        "All inputs are validation-only and report `test_assets_read=false`.",
    ])
    (args.output_dir / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
