#!/usr/bin/env python3
"""Build a deterministic, stratified qualitative selection registry.

The input casebooks must already be exhaustive validation exports.  This
script records selection strata and candidate counts; it does not inspect test
assets and does not rank records by the magnitude of a positive result.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sn7_stratum(row: dict[str, Any]) -> tuple[str, str]:
    delta = float(row["paired"]["committed_map_iou_delta"])
    direction = "left_better" if delta > 1e-9 else "right_better" if delta < -1e-9 else "matched"
    return str(row["target_edit"]), direction


def _muno_stratum(row: dict[str, Any]) -> tuple[str, str]:
    direct, active = row["direct_metrics"], row["activemap_metrics"]
    if direct["false_edit"] and not active["false_edit"]:
        outcome = "false_write_prevented"
    elif direct["missed_edit"] and not active["missed_edit"]:
        outcome = "missed_edit_recovered"
    elif active["false_edit"] and not direct["false_edit"]:
        outcome = "false_write_introduced"
    elif active["missed_edit"] and not direct["missed_edit"]:
        outcome = "missed_edit_introduced"
    elif row["paired_raster_iou_gain_delta"] > 1e-9:
        outcome = "quality_improved"
    elif row["paired_raster_iou_gain_delta"] < -1e-9:
        outcome = "quality_degraded"
    else:
        outcome = "matched"
    return str(row["target_edit"]), outcome


def _sn8_stratum(row: dict[str, Any]) -> tuple[str, str]:
    delta = float(row["selection"]["learned_minus_first"])
    direction = "selected_better" if delta > 1e-9 else "selected_worse" if delta < -1e-9 else "matched"
    return direction, "committed" if row["safe_commit"]["commit"] else "deferred"


def _visibility(row: dict[str, Any]) -> float:
    if row["dataset"] == "SpaceNet 7":
        values = []
        for method in row["methods"].values():
            values.extend(
                [
                    float(method.get("target_change_fraction", 0.0)),
                    float(method.get("predicted_change_fraction", 0.0)),
                    float(method.get("prior_foreground_fraction", 0.0)),
                ]
            )
        return max(values, default=0.0)
    # MUNO21 and SpaceNet8 receipts do not contain a neutral visual-footprint
    # field.  Preserve their complete outcome strata and choose deterministically
    # by geography/case id instead of treating an outcome metric as a visual score.
    return 0.0


def _select(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    strata: Callable[[dict[str, Any]], tuple[str, str]],
    max_per_stratum: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[strata(row)].append(row)
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for key in sorted(grouped):
        candidates = grouped[key]
        counts["/".join(key)] = len(candidates)
        # Stable visual triage: SN7 uses map footprint; other domains use AOI
        # and case id.  Positive/negative outcomes are strata, never scores.
        ordered = sorted(
            candidates,
            key=lambda row: (-_visibility(row), str(row.get("aoi_id") or ""), str(row["case_id"])),
        )
        used_aoi: set[str] = set()
        for row in ordered:
            aoi = str(row.get("aoi_id") or "")
            if aoi and aoi in used_aoi and len(used_aoi) < max_per_stratum:
                continue
            used_aoi.add(aoi)
            selected.append(
                {
                    "dataset": dataset,
                    "case_id": row["case_id"],
                    "folder": row["folder"],
                    "stratum": list(key),
                    "aoi_id": row.get("aoi_id"),
                    "visibility_proxy": _visibility(row),
                    "selection_rule": "stratified outcome coverage; visibility then AOI then case id",
                    "source_record": row,
                }
            )
            if sum(item["dataset"] == dataset and item["stratum"] == list(key) for item in selected) >= max_per_stratum:
                break
    return selected, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sn7_index", type=Path)
    parser.add_argument("muno21_index", type=Path)
    parser.add_argument("spacenet8_index", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-per-stratum", type=int, default=1)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.max_per_stratum < 1:
        raise ValueError("max-per-stratum must be positive")
    inputs = {
        "SpaceNet 7": _read_jsonl(args.sn7_index),
        "MUNO21": _read_jsonl(args.muno21_index),
        "SpaceNet 8": _read_jsonl(args.spacenet8_index),
    }
    for dataset, rows in inputs.items():
        if not rows or any(row.get("split") != "val" or row.get("test_assets_read") is not False for row in rows):
            raise ValueError(f"{dataset} input is not a non-empty validation-only index")
    selected, counts = [], {}
    for dataset, rows, function in (
        ("SpaceNet 7", inputs["SpaceNet 7"], _sn7_stratum),
        ("MUNO21", inputs["MUNO21"], _muno_stratum),
        ("SpaceNet 8", inputs["SpaceNet 8"], _sn8_stratum),
    ):
        records, dataset_counts = _select(
            rows, dataset=dataset, strata=function, max_per_stratum=args.max_per_stratum
        )
        selected.extend(records)
        counts[dataset] = dataset_counts
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "qualitative_selection_registry.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "schema_version": "qualitative-selection-registry-v1",
        "split": "val",
        "test_assets_read": False,
        "input_counts": {dataset: len(rows) for dataset, rows in inputs.items()},
        "candidate_counts_by_stratum": counts,
        "selected_count": len(selected),
        "max_per_stratum": args.max_per_stratum,
        "selection_rule": "stratified outcome coverage; SN7 map footprint then AOI/case id, otherwise AOI/case id; no score-magnitude ranking within a stratum",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
