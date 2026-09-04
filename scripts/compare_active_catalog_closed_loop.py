#!/usr/bin/env python3
"""Paired AOI-bootstrap comparison for model and closed-loop baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluate_active_catalog_closed_loop import metrics

COMPARISON_FIELDS = (
    ("terminal_accuracy", "terminal_correct"),
    ("false_edit_rate", "false_edit"),
    ("missed_edit_rate", "missed_edit"),
    ("mean_cost", "spent_cost"),
    ("mean_quality_gain", "quality_gain"),
    ("mean_quality_cost_utility", "quality_cost_utility"),
    (
        "mean_episode_utility_v2_proxy_balanced",
        "episode_utility_v2_proxy_balanced",
    ),
    (
        "mean_episode_utility_v2_proxy_safety",
        "episode_utility_v2_proxy_safety",
    ),
    (
        "mean_episode_utility_v2_proxy_cost_aware",
        "episode_utility_v2_proxy_cost_aware",
    ),
)
COMPARISON_METRICS = tuple(name for name, _ in COMPARISON_FIELDS)


def parse_record(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("record must be label=/path/to/traces.jsonl")
    return label, Path(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_validation_manifest(path: Path) -> tuple[set[tuple[str, float]], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "val" or payload.get("test_assets_read") is not False:
        raise ValueError("comparison manifest must be validation-only")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("comparison manifest has no records")
    try:
        keys = {
            (str(row["source_episode"]), float(row["budget"]))
            for row in records
        }
    except (KeyError, TypeError) as error:
        raise ValueError("comparison manifest has malformed records") from error
    if len(keys) != len(records):
        raise ValueError("comparison manifest has duplicate episode-budget identities")
    return keys, payload


def load_rows(
    path: Path, *, split: str = "val", frozen_test: bool = False
) -> dict[tuple[str, float], dict[str, Any]]:
    test_assets_read = split == "test"
    if test_assets_read:
        if not frozen_test:
            raise PermissionError("test comparison requires --frozen-test")
        from activemap.frozen_test import assert_frozen_test_access

        assert_frozen_test_access()
    elif frozen_test:
        raise ValueError("--frozen-test is valid only for the test split")
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != split or row.get("test_assets_read") is not test_assets_read:
                raise ValueError(f"invalid {split} trace at {path}:{line_number}")
            key = (str(row["source_episode"]), float(row["budget"]))
            if key in result:
                raise ValueError(f"duplicate trace identity in {path}: {key}")
            result[key] = row
    if not result:
        raise ValueError(f"empty trace: {path}")
    return result


def metric_delta(
    candidate: list[dict[str, Any]], reference: list[dict[str, Any]]
) -> dict[str, float]:
    candidate_metrics = metrics(candidate)
    reference_metrics = metrics(reference)
    return {
        name: candidate_metrics[name] - reference_metrics[name]
        for name in COMPARISON_METRICS
    }


def paired_aoi_bootstrap(
    candidate: dict[tuple[str, float], dict[str, Any]],
    reference: dict[tuple[str, float], dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if candidate.keys() != reference.keys():
        raise ValueError("paired comparison requires identical episode-budget keys")
    groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for key, row in candidate.items():
        if str(row["aoi_id"]) != str(reference[key]["aoi_id"]):
            raise ValueError(f"AOI mismatch for {key}")
        groups[str(row["aoi_id"])].append(key)
    if len(groups) < 2 or repetitions <= 0:
        raise ValueError("paired AOI bootstrap requires multiple AOIs and repetitions")
    group_ids = sorted(groups)
    ordered_keys = sorted(candidate)
    observed = metric_delta(
        [candidate[key] for key in ordered_keys],
        [reference[key] for key in ordered_keys],
    )
    group_counts = np.asarray(
        [len(groups[group_id]) for group_id in group_ids], dtype=np.float64
    )
    group_delta_sums = np.zeros(
        (len(group_ids), len(COMPARISON_METRICS)), dtype=np.float64
    )
    for group_index, group_id in enumerate(group_ids):
        for key in groups[group_id]:
            for metric_index, (_, field) in enumerate(COMPARISON_FIELDS):
                group_delta_sums[group_index, metric_index] += float(
                    candidate[key][field]
                ) - float(reference[key][field])
    rng = np.random.default_rng(seed)
    draw_matrix = np.empty((repetitions, len(COMPARISON_METRICS)), dtype=np.float64)
    # Accumulate each AOI-resample explicitly. This is algebraically identical
    # to the former one-hot matrix product but avoids a platform-specific MKL
    # abort observed for tiny test matrices on Windows.
    for draw_index in range(repetitions):
        sampled_indices = rng.integers(0, len(group_ids), size=len(group_ids))
        total_rows = 0.0
        total_delta = np.zeros(len(COMPARISON_METRICS), dtype=np.float64)
        for group_index in sampled_indices:
            total_rows += group_counts[group_index]
            total_delta += group_delta_sums[group_index]
        draw_matrix[draw_index] = total_delta / total_rows
    intervals = {
        name: {
            "observed_delta": observed[name],
            "ci95_low": float(np.quantile(draw_matrix[:, index], 0.025)),
            "ci95_high": float(np.quantile(draw_matrix[:, index], 0.975)),
        }
        for index, name in enumerate(COMPARISON_METRICS)
    }
    return {
        "group_key": "aoi_id",
        "group_count": len(group_ids),
        "repetitions": repetitions,
        "intervals": intervals,
        "non_dominated_observed": (
            observed["mean_quality_cost_utility"] > 0
            and observed["terminal_accuracy"] >= 0
            and observed["false_edit_rate"] <= 0
        ),
        "strict_utility_gain_ci95": (
            intervals["mean_quality_cost_utility"]["ci95_low"] > 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--records", action="append", type=parse_record, required=True)
    parser.add_argument("--reference", default="always_stop")
    parser.add_argument("--candidate")
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--frozen-test", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Validation-only manifest that every input trace must exactly cover.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    paths = dict(args.records)
    if len(paths) != len(args.records):
        raise ValueError("duplicate record labels")
    if args.candidate is not None and args.candidate not in paths:
        raise ValueError(f"missing candidate policy: {args.candidate}")
    if args.candidate is None and args.reference not in paths:
        raise ValueError(f"missing reference policy: {args.reference}")
    rows = {
        label: load_rows(path, split=args.split, frozen_test=args.frozen_test)
        for label, path in paths.items()
    }
    manifest_receipt = None
    if args.manifest is not None:
        if args.split != "val":
            raise ValueError("--manifest is supported only for validation comparisons")
        manifest_keys, manifest_payload = load_validation_manifest(args.manifest)
        for label, policy_rows in rows.items():
            if policy_rows.keys() != manifest_keys:
                raise ValueError(
                    f"trace {label} does not exactly cover the comparison manifest"
                )
        manifest_receipt = {
            "path": str(args.manifest.resolve()),
            "sha256": _sha256(args.manifest),
            "record_count": len(manifest_keys),
            "selection_contract": manifest_payload.get("selection_contract"),
        }
    if args.candidate is not None:
        candidate = rows[args.candidate]
        comparisons = {
            f"{args.candidate}_minus_{label}": paired_aoi_bootstrap(
                candidate,
                value,
                repetitions=args.repetitions,
                seed=args.seed,
            )
            for label, value in rows.items()
            if label != args.candidate
        }
        orientation = {
            "candidate": args.candidate,
            "references": sorted(label for label in rows if label != args.candidate),
        }
        record_count = len(candidate)
    else:
        reference = rows[args.reference]
        comparisons = {
            f"{label}_minus_{args.reference}": paired_aoi_bootstrap(
                value,
                reference,
                repetitions=args.repetitions,
                seed=args.seed,
            )
            for label, value in rows.items()
            if label != args.reference
        }
        orientation = {"reference": args.reference}
        record_count = len(reference)
    summary = {
        "schema_version": "active-catalog-closed-loop-paired-comparison-v1",
        "comparison_orientation": orientation,
        "record_count": record_count,
        "metrics": {
            label: metrics(list(value.values())) for label, value in rows.items()
        },
        "traces": {
            label: {"path": str(paths[label].resolve()), "sha256": _sha256(paths[label])}
            for label in sorted(paths)
        },
        "comparison_manifest": manifest_receipt,
        "paired_aoi_comparisons": comparisons,
        "interpretation": {
            "quality_gain_semantics": "frozen_teacher_proxy",
            "negative_false_edit_delta_is_better": True,
            "negative_cost_delta_is_better": True,
            "positive_other_deltas_are_better": True,
        },
        "split": args.split,
        "test_assets_read": args.split == "test",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
