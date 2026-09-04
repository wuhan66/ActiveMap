#!/usr/bin/env python3
"""Build the final matched executable controller table from writeback records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.compare_agent_writebacks import _load, _metrics, compare


DISPLAY_METRICS = (
    ("raster_iou_auc", "Raster IoU"),
    ("raster_iou_gain_auc", "Raster gain"),
    ("episode_utility_v2_balanced_auc", "Balanced utility"),
    ("episode_utility_v2_safety_auc", "Safety utility"),
    ("false_edit_auc", "False edit"),
    ("missed_edit_auc", "Missed edit"),
    ("spent_cost_auc", "Cost"),
    ("vector_replay_iou_auc", "Replay IoU"),
    ("vector_delta_topology_valid_auc", "Topology"),
)


def _parse_method(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("method must be label=/path/writeback.jsonl")
    return label, Path(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_table(
    method_paths: dict[str, Path],
    *,
    reference: str,
    candidate: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if reference not in method_paths or candidate not in method_paths:
        raise ValueError("reference and candidate must be registered methods")
    loaded = {label: _load(path) for label, path in method_paths.items()}
    support = set(loaded[reference])
    for label, rows in loaded.items():
        if set(rows) != support:
            raise ValueError(f"{label} lacks identical executable task-budget support")
        for key in support:
            if (
                rows[key].get("target") != loaded[reference][key].get("target")
                or rows[key].get("aoi_id") != loaded[reference][key].get("aoi_id")
            ):
                raise ValueError(f"{label} metadata differs on {key}")
            if rows[key].get("split", "val") != "val" or rows[key].get(
                "test_assets_read", False
            ):
                raise ValueError(f"{label} includes non-validation evidence")
    absolute = {
        label: _metrics([rows[key] for key in sorted(rows)])
        for label, rows in loaded.items()
    }
    versus_reference = {
        label: compare(
            method_paths[reference],
            path,
            bootstrap=repetitions,
            seed=seed,
            group_key="aoi_id",
        )["paired_delta"]
        for label, path in method_paths.items()
        if label != reference
    }
    candidate_vs_all = {
        label: compare(
            path,
            method_paths[candidate],
            bootstrap=repetitions,
            seed=seed,
            group_key="aoi_id",
        )["paired_delta"]
        for label, path in method_paths.items()
        if label != candidate
    }
    return {
        "schema_version": "sn7-executable-controller-table-v1",
        "split": "val",
        "test_assets_read": False,
        "reference": reference,
        "candidate": candidate,
        "record_count": len(support),
        "aoi_count": len(
            {str(loaded[reference][key]["aoi_id"]) for key in support}
        ),
        "budgets": sorted({float(key[1]) for key in support}),
        "methods": absolute,
        "paired_vs_reference": versus_reference,
        "candidate_paired_vs_all": candidate_vs_all,
        "bootstrap": {
            "unit": "AOI across matched tasks and budgets",
            "repetitions": repetitions,
            "seed": seed,
        },
        "sources": {
            label: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for label, path in method_paths.items()
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SN7 Executable Controller Table",
        "",
        (
            f"Validation-only, {payload['record_count']} matched task-budget rows, "
            f"{payload['aoi_count']} AOIs. Candidate: `{payload['candidate']}`."
        ),
        "",
        "| Controller | " + " | ".join(label for _, label in DISPLAY_METRICS) + " |",
        "| --- | " + " | ".join("---:" for _ in DISPLAY_METRICS) + " |",
    ]
    for label, values in payload["methods"].items():
        lines.append(
            "| "
            + " | ".join(
                [label, *(f"{float(values.get(name, 0.0)):.6f}" for name, _ in DISPLAY_METRICS)]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "All rows use the same executable updater, budgets, targets, and AOI support.",
            "Paired AOI confidence intervals are stored in the JSON artifact.",
            "Test assets were not read.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--method", action="append", type=_parse_method, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    methods = dict(args.method)
    if len(methods) != len(args.method):
        raise ValueError("duplicate method label")
    payload = build_table(
        methods,
        reference=args.reference,
        candidate=args.candidate,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "controller_table.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "controller_table.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
