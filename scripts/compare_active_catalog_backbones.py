#!/usr/bin/env python3
"""Paired AOI comparison for two active-catalog VLM backbones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.compare_active_catalog_sampling_ablation import (
    _load_jsonl,
    _validate_summary,
    compare_traces,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_trace", type=Path)
    parser.add_argument("reference_trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--reference-name", required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    candidate_summary = _validate_summary(args.candidate_trace)
    reference_summary = _validate_summary(args.reference_trace)
    for key in ("validation", "evaluation_index"):
        if candidate_summary["sources"][key]["sha256"] != reference_summary["sources"][key]["sha256"]:
            raise ValueError(f"backbone comparison has mismatched {key}")
    paired = compare_traces(
        _load_jsonl(args.candidate_trace),
        _load_jsonl(args.reference_trace),
        repetitions=args.repetitions,
        seed=args.seed,
    )
    intervals = paired["paired_aoi_bootstrap"]["intervals"]
    for value in intervals.values():
        value["observed_candidate_improvement"] = value.pop("observed_weighted_improvement")
    report = {
        "schema_version": "active-catalog-backbone-paired-aoi-v1",
        "candidate": {
            "name": args.candidate_name,
            "trace": str(args.candidate_trace.resolve()),
            "metrics": paired["weighted_metrics"],
        },
        "reference": {
            "name": args.reference_name,
            "trace": str(args.reference_trace.resolve()),
            "metrics": paired["unweighted_metrics"],
        },
        "sample_count": paired["sample_count"],
        "aoi_count": paired["aoi_count"],
        "paired_aoi_bootstrap": paired["paired_aoi_bootstrap"],
        "claim_boundary": "Single-seed validation-only backbone robustness comparison.",
        "test_assets_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
