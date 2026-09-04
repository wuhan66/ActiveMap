#!/usr/bin/env python3
"""Aggregate compatible ArgoTweak native-adaptation official evaluations.

The official ArgoTweak evaluator writes human-readable tables instead of a
machine-readable metric file.  This utility parses the fixed Map Generation
table, verifies proposal-export protocol compatibility, and emits a compact
receipt suitable for the experiment ledger.  It deliberately does not read
test assets or alter model outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any


MAP_ROW = re.compile(r"^\|\s*mAP\s*\|(?P<cells>.+)\|\s*$")
METRIC_NAMES = (
    "all_map",
    "unchanged_map",
    "changed_map",
    "inserted_map",
    "deleted_map",
    "other_map",
)


def parse_map_metrics(path: Path) -> dict[str, float]:
    """Read the final Map Generation mAP row from an official evaluator log."""
    matches: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = MAP_ROW.match(line.strip())
        if match is None:
            continue
        cells = [cell.strip() for cell in match.group("cells").split("|")]
        values: list[float] = []
        for cell in cells[: len(METRIC_NAMES)]:
            try:
                values.append(float(cell))
            except ValueError as error:
                raise ValueError(f"non-numeric mAP cell in {path}: {cell!r}") from error
        if len(values) != len(METRIC_NAMES):
            raise ValueError(f"incomplete mAP row in {path}")
        matches.append(dict(zip(METRIC_NAMES, values)))
    if not matches:
        raise ValueError(f"no Map Generation mAP row found in {path}")
    return matches[-1]


def parse_seed(value: str) -> tuple[str, Path, Path]:
    try:
        name, log, proposal = value.split("=", 2)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--seed requires NAME=EVAL_LOG=PROPOSAL_SUMMARY") from error
    return name, Path(log), Path(proposal)


def _summary_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty metric")
    result: dict[str, float | int] = {"mean": mean(values), "n": len(values)}
    result["sample_std"] = stdev(values) if len(values) > 1 else 0.0
    result["min"] = min(values)
    result["max"] = max(values)
    return result


def _load_export_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "frames",
        "object_threshold",
        "object_assignment_policy",
        "object_match_distance",
        "test_assets_read",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"proposal summary {path} lacks required protocol fields: {missing}")
    if value["test_assets_read"]:
        raise ValueError(f"proposal summary must be validation-only: {path}")
    return value


def _protocol_signature(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": summary["schema_version"],
        "frames": summary["frames"],
        "object_threshold": summary["object_threshold"],
        "object_assignment_policy": summary["object_assignment_policy"],
        "object_match_distance": summary["object_match_distance"],
        "test_assets_read": summary["test_assets_read"],
    }


def _markdown(receipt: dict[str, Any]) -> str:
    lines = [
        "# ArgoTweak Native-Adaptation Aggregate",
        "",
        "This receipt aggregates three independently trained native-domain "
        "adaptations under the same official validation protocol. It is "
        "conditional transfer evidence, not a replacement for the frozen SN7 claim.",
        "",
        "## Protocol",
        "",
        f"- Seeds: {', '.join(receipt['seeds'])}",
        f"- Official validation frames per seed: {receipt['proposal_protocol']['frames']}",
        f"- Proposal object threshold: {receipt['proposal_protocol']['object_threshold']}",
        f"- Assignment: {receipt['proposal_protocol']['object_assignment_policy']} "
        f"(distance {receipt['proposal_protocol']['object_match_distance']})",
        "- Test assets read: false",
        "",
        "## Official Map Generation",
        "",
        "| Metric | Mean | Sample std | Min | Max | n |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, stats in receipt["aggregate"].items():
        lines.append(
            f"| {key} | {stats['mean']:.6f} | {stats['sample_std']:.6f} | "
            f"{stats['min']:.6f} | {stats['max']:.6f} | {stats['n']} |"
        )
    baseline = receipt.get("baseline")
    if baseline:
        lines.extend(
            [
                "",
                "## Reference Checkpoint",
                "",
                f"Official baseline all mAP: {baseline['all_map']:.6f}.",
                "This number is descriptive only because it is a single checkpoint, not a matched three-seed distribution.",
            ]
        )
    lines.extend(["", "## Per Seed", "", "| Seed | All mAP | Changed mAP | Inserted mAP | Deleted mAP |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in receipt["per_seed"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['seed']} | {metrics['all_map']:.6f} | {metrics['changed_map']:.6f} | "
            f"{metrics['inserted_map']:.6f} | {metrics['deleted_map']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        action="append",
        type=parse_seed,
        required=True,
        help="NAME=EVAL_LOG=PROPOSAL_SUMMARY; pass once per compatible seed",
    )
    parser.add_argument("--baseline-log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if len(args.seed) < 2:
        raise ValueError("at least two independently trained seeds are required")

    names = [name for name, _, _ in args.seed]
    if len(set(names)) != len(names):
        raise ValueError("seed names must be unique")
    per_seed: list[dict[str, Any]] = []
    signatures: list[dict[str, Any]] = []
    for name, log_path, proposal_path in args.seed:
        proposal = _load_export_summary(proposal_path)
        signatures.append(_protocol_signature(proposal))
        per_seed.append(
            {
                "seed": name,
                "evaluation_log": str(log_path),
                "proposal_summary": str(proposal_path),
                "metrics": parse_map_metrics(log_path),
            }
        )
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("proposal exports are not protocol-compatible")

    aggregate = {
        metric: _summary_stats([float(row["metrics"][metric]) for row in per_seed])
        for metric in METRIC_NAMES
    }
    receipt: dict[str, Any] = {
        "schema_version": "activemap-argotweak-native-adaptation-aggregate-v1",
        "scope": "official_validation_only_conditional_transfer_evidence",
        "test_assets_read": False,
        "seeds": names,
        "proposal_protocol": signatures[0],
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    if args.baseline_log is not None:
        receipt["baseline"] = {
            "evaluation_log": str(args.baseline_log),
            **parse_map_metrics(args.baseline_log),
        }
        receipt["baseline_delta_all_map"] = aggregate["all_map"]["mean"] - receipt["baseline"]["all_map"]

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "aggregate.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "AGGREGATE.md").write_text(_markdown(receipt), encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "all_map": aggregate["all_map"]}, indent=2))


if __name__ == "__main__":
    main()
