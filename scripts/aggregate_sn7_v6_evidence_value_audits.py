#!/usr/bin/env python3
"""Aggregate immutable three-seed V6 Evidence Value policy audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

METRICS = (
    "acquire_rate",
    "oracle_acquire_rate",
    "acquire_recall",
    "false_call_rate",
    "harmful_call_fraction",
    "false_or_wrong_call_fraction",
    "missed_edit_call_fraction",
    "exact_oracle_candidate_rate",
    "mean_chosen_utility",
    "mean_oracle_utility",
    "mean_regret",
    "mean_selected_final_raster_iou",
)


def parse_record(value: str) -> tuple[int, Path]:
    raw_seed, separator, raw_path = value.partition("=")
    if not separator or not raw_seed.isdigit() or not raw_path:
        raise argparse.ArgumentTypeError("record must be SEED=SUMMARY_JSON")
    return int(raw_seed), Path(raw_path)


def load_record(seed: int, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "sn7-v6-evidence-value-policy-audit-v1":
        raise ValueError(f"{path}: unexpected V6 policy-audit schema")
    if payload.get("split") != "val" or payload.get("test_assets_read") is not False:
        raise ValueError(f"{path}: only validation-only V6 audits are accepted")
    checkpoint = Path(str(payload.get("checkpoint", "")))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{path}: checkpoint is missing")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha256 != payload.get("checkpoint_sha256"):
        raise ValueError(f"{path}: audited checkpoint hash does not match current file")
    overall = payload.get("overall")
    if not isinstance(overall, dict) or set(METRICS) - overall.keys():
        raise ValueError(f"{path}: incomplete overall metrics")
    return {
        "seed": seed,
        "summary": str(path.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "state_count": overall.get("state_count"),
        "overall": {metric: float(overall[metric]) for metric in METRICS},
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 3:
        raise ValueError("V6 policy-audit aggregation requires exactly three seeds")
    seeds = [int(record["seed"]) for record in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate V6 policy-audit seed")
    metric_summary = {}
    for metric in METRICS:
        values = np.asarray([record["overall"][metric] for record in records])
        metric_summary[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "per_seed": {
                str(record["seed"]): float(record["overall"][metric])
                for record in records
            },
        }
    return {
        "schema_version": "sn7-v6-evidence-value-policy-audit-three-seed-v1",
        "seeds": sorted(seeds),
        "records": sorted(records, key=lambda record: int(record["seed"])),
        "metrics": metric_summary,
        "interpretation": (
            "Cached candidate-outcome selection audit only; not an executable "
            "map-writeback result."
        ),
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--record", action="append", type=parse_record, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite V6 audit aggregation: {args.output}")
    raw_records = dict(args.record)
    if len(raw_records) != len(args.record):
        raise ValueError("duplicate V6 policy-audit record")
    result = aggregate(
        [load_record(seed, path) for seed, path in sorted(raw_records.items())]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
