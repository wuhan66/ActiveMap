#!/usr/bin/env python3
"""Build a frozen validation-only diagnostic split for sparse tool use."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from activemap.agent.tool_sft import select_sparse_tool_stage


def build_tool_opportunity_strata(
    detail_rows: list[dict[str, Any]],
    *,
    split: str = "validation",
    positive_margin: float = 1e-6,
    harmful_margin: float = -0.2,
) -> dict[str, Any]:
    """Stratify episodes by the best achievable tool-prefix utility margin.

    The split is diagnostic: labels use validation ground truth and must never be
    exposed to a policy or used for checkpoint selection.
    """

    if split == "test":
        raise ValueError("test data are forbidden when constructing opportunity strata")
    if harmful_margin >= positive_margin:
        raise ValueError("harmful_margin must be smaller than positive_margin")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        grouped[str(row["sequence_id"])].append(row)
    if not grouped:
        raise ValueError("no detail rows supplied")

    records: list[dict[str, Any]] = []
    episode_ids: set[str] = set()
    for sequence_id in sorted(grouped):
        rows = grouped[sequence_id]
        selected_stage, utilities, predictions = select_sparse_tool_stage(rows)
        ordered = sorted(rows, key=lambda row: int(row["step"]))
        episode_id = str(ordered[0]["episode_id"])
        if episode_id in episode_ids:
            raise ValueError(f"duplicate episode_id across sequences: {episode_id}")
        episode_ids.add(episode_id)

        tool_margins = [utility - utilities[0] for utility in utilities[1:]]
        best_tool_stage = max(
            range(1, 4), key=lambda stage: (utilities[stage], -stage)
        )
        best_tool_margin = tool_margins[best_tool_stage - 1]
        if best_tool_margin > positive_margin:
            stratum = "positive"
        elif best_tool_margin < harmful_margin:
            stratum = "harmful"
        else:
            stratum = "near_boundary"

        records.append(
            {
                "sequence_id": sequence_id,
                "episode_id": episode_id,
                "stratum": stratum,
                "target": str(ordered[0]["target"]),
                "baseline_prediction": str(ordered[0]["baseline"]),
                "baseline_utility": utilities[0],
                "stage_utilities": utilities[1:],
                "stage_predictions": [prediction.value for prediction in predictions[1:]],
                "best_tool_stage": best_tool_stage,
                "oracle_selected_stage": selected_stage,
                "best_tool_margin": best_tool_margin,
                "forced_full_margin": utilities[3] - utilities[0],
            }
        )

    counts = Counter(record["stratum"] for record in records)
    canonical_ids = "\n".join(record["episode_id"] for record in records)
    return {
        "schema_version": "tool-opportunity-strata-v1",
        "split": split,
        "scope": "diagnostic_only",
        "uses_ground_truth": True,
        "allowed_for_checkpoint_selection": False,
        "allowed_as_primary_result": False,
        "thresholds": {
            "positive_margin_strictly_greater_than": positive_margin,
            "harmful_margin_strictly_less_than": harmful_margin,
        },
        "sequence_count": len(records),
        "stratum_counts": dict(sorted(counts.items())),
        "episode_id_sha256": hashlib.sha256(canonical_ids.encode("utf-8")).hexdigest(),
        "records": records,
        "test_assets_read": False,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"row {line_number} in {path} is not an object")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("details", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", default="validation", choices=("train", "validation"))
    parser.add_argument("--positive-margin", type=float, default=1e-6)
    parser.add_argument("--harmful-margin", type=float, default=-0.2)
    args = parser.parse_args()

    result = build_tool_opportunity_strata(
        _read_jsonl(args.details),
        split=args.split,
        positive_margin=args.positive_margin,
        harmful_margin=args.harmful_margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
