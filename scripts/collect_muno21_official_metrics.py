#!/usr/bin/env python3
"""Collect official MUNO21 graph outputs into provenance-ready JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.export_muno21_writeback_graphs import _episode_index_lookup


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score_map(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, float] = {}
    for row in payload:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"invalid official score row in {path}: {row!r}")
        index = int(row[0])
        value = float(row[1])
        if index in result or not math.isfinite(value):
            raise ValueError(f"invalid or duplicate official score for annotation {index}")
        result[index] = value
    return result


def _budget(path: Path) -> float:
    prefix = "budget-"
    if not path.name.startswith(prefix):
        raise ValueError(f"invalid budget directory: {path}")
    return float(path.name[len(prefix) :].replace("p", "."))


def collect(
    annotations_path: Path,
    regions_path: Path,
    episodes_path: Path,
    graphs_root: Path,
    *,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if split not in {"val", "test"}:
        raise ValueError("official metric split must be val or test")
    test_assets_read = split == "test"
    if test_assets_read:
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()

    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    regions = set(map(str, json.loads(regions_path.read_text(encoding="utf-8"))))
    task_lookup = _episode_index_lookup(episodes_path, expected_split=split)
    tasks_by_annotation: dict[int, list[str]] = defaultdict(list)
    for task_id, (annotation_index, _) in task_lookup.items():
        tasks_by_annotation[annotation_index].append(task_id)

    selected_indices = {
        index
        for index, annotation in enumerate(annotations)
        if str(annotation["Cluster"]["Region"]) in regions
    }
    change_indices = {
        index
        for index in selected_indices
        if "nochange" not in {str(tag).lower() for tag in annotations[index].get("Tags", [])}
    }
    nochange_indices = selected_indices - change_indices
    missing_tasks = sorted(selected_indices - set(tasks_by_annotation))
    if missing_tasks:
        raise ValueError(f"official annotations lack ActiveMap tasks: {missing_tasks[:20]}")
    if not change_indices or not nochange_indices:
        raise ValueError("official metric collection requires change and no-change scenarios")

    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    budgets = []
    for budget_dir in sorted(path for path in graphs_root.glob("budget-*") if path.is_dir()):
        budget = _budget(budget_dir)
        budgets.append(budget)
        apls_path = budget_dir / "scores.json"
        geo_path = budget_dir / "geo.json"
        error_path = budget_dir / "error.json"
        for path in (apls_path, geo_path, error_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing official metric output: {path}")
            sources.append(
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "budget": budget,
                }
            )
        apls = _score_map(apls_path)
        geo = _score_map(geo_path)
        if set(apls) != change_indices or set(geo) != change_indices:
            raise ValueError(
                f"official change support mismatch at budget {budget}: "
                f"apls={len(apls)}, geo={len(geo)}, expected={len(change_indices)}"
            )
        for annotation_index in sorted(change_indices):
            task_id = sorted(tasks_by_annotation[annotation_index])[0]
            rows.append(
                {
                    "task_id": task_id,
                    "annotation_index": annotation_index,
                    "budget": budget,
                    "apls_improvement": apls[annotation_index],
                    "pixel_f1_improvement": geo[annotation_index],
                    "official_unit": "change_scenario",
                }
            )
        error_rate = float(json.loads(error_path.read_text(encoding="utf-8")))
        if not math.isfinite(error_rate) or not 0.0 <= error_rate <= 1.0:
            raise ValueError(f"invalid official error rate at budget {budget}: {error_rate}")
        rows.append(
            {
                "task_id": "__aggregate__",
                "budget": budget,
                "no_change_error_rate": error_rate,
                "support_count": len(nochange_indices),
                "official_unit": "official_nochange_aggregate",
            }
        )

    if not budgets:
        raise ValueError(f"no official budget outputs in {graphs_root}")
    summary = {
        "schema_version": "muno21-official-metrics-structured-v1",
        "split": split,
        "budgets": budgets,
        "change_scenario_count": len(change_indices),
        "nochange_scenario_count": len(nochange_indices),
        "apls_unit": "official_change_scenario",
        "pixel_f1_unit": "official_change_scenario",
        "error_rate_unit": "official_aggregate_per_model_seed",
        "sources": sources,
        "test_assets_read": test_assets_read,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    parser.add_argument("regions", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("graphs_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--frozen-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.frozen_test:
        raise PermissionError("test official metrics require --frozen-test")
    if args.split != "test" and args.frozen_test:
        raise ValueError("--frozen-test is valid only with --split test")
    if args.output.exists() or args.output.with_suffix(".summary.json").exists():
        raise FileExistsError(f"refusing to overwrite official metrics: {args.output}")
    rows, summary = collect(
        args.annotations,
        args.regions,
        args.episodes,
        args.graphs_root,
        split=args.split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
