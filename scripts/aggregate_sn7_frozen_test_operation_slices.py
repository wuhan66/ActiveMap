#!/usr/bin/env python3
"""Aggregate post-hoc SN7 frozen-test deltas by ground-truth edit type."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


OPERATIONS = ("KEEP", "ADD", "DELETE", "RESHAPE")
METRICS = (
    "map_quality_after",
    "map_quality_gain",
    "false_edit",
    "missed_edit",
    "wrong_edit",
    "spent_cost",
    "episode_utility_v2_balanced",
    "episode_utility_v2_safety",
    "episode_utility_v2_cost_aware",
    "vector_delta_topology_valid",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair(value: str) -> tuple[int, Path, Path]:
    raw_seed, separator, paths = value.partition("=")
    reference, comma, candidate = paths.partition(",")
    if not separator or not comma:
        raise argparse.ArgumentTypeError("pair must be SEED=REFERENCE,CANDIDATE")
    return int(raw_seed), Path(reference), Path(candidate)


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                if row.get("split") != "test" or row.get("test_assets_read") is not True:
                    raise ValueError(f"non-test row in {path}:{line_number}")
                yield row


def _operation(target: Any) -> str:
    text = str(target or "KEEP").upper().replace("-", "_")
    if text in {"REJECT", "STOP", "KEEP", "COMMIT:KEEP"}:
        return "KEEP"
    for operation in OPERATIONS[1:]:
        if text == operation or text.endswith(f":{operation}"):
            return operation
    raise ValueError(f"unsupported target operation: {target!r}")


def _values(row: dict[str, Any]) -> dict[str, float]:
    before = float(row["map_quality_before"])
    after = float(row["map_quality_after"])
    values = {
        "map_quality_after": after,
        "map_quality_gain": after - before,
        "false_edit": float(bool(row["false_edit"])),
        "missed_edit": float(bool(row["missed_edit"])),
        "wrong_edit": float(bool(row["wrong_edit"])),
        "spent_cost": float(row["spent_cost"]),
        "episode_utility_v2_balanced": float(row["episode_utility_v2_balanced"]),
        "episode_utility_v2_safety": float(row["episode_utility_v2_safety"]),
        "episode_utility_v2_cost_aware": float(row["episode_utility_v2_cost_aware"]),
        "vector_delta_topology_valid": float(bool(row["vector_delta_topology_valid"])),
    }
    return values


def _empty_stats() -> dict[str, Any]:
    return {"count": 0, "sums": {metric: 0.0 for metric in METRICS}}


def _add(stats: dict[str, Any], values: dict[str, float]) -> None:
    stats["count"] += 1
    for metric in METRICS:
        stats["sums"][metric] += values[metric]


def _means(stats: dict[str, Any]) -> dict[str, float]:
    count = int(stats["count"])
    if count <= 0:
        raise ValueError("cannot summarize an empty operation slice")
    return {metric: float(stats["sums"][metric]) / count for metric in METRICS}


def _read_pair(
    reference_path: Path,
    candidate_path: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "reference": _empty_stats(),
                "candidate": _empty_stats(),
                "delta": _empty_stats(),
            }
        )
    )
    reference_rows = _rows(reference_path)
    candidate_rows = _rows(candidate_path)
    sentinel = object()
    while True:
        reference = next(reference_rows, sentinel)
        candidate = next(candidate_rows, sentinel)
        if reference is sentinel or candidate is sentinel:
            if reference is not candidate:
                raise ValueError("paired writeback files have different lengths")
            break
        key_fields = ("task_id", "aoi_id", "budget", "target")
        if any(reference.get(field) != candidate.get(field) for field in key_fields):
            raise ValueError("paired writeback rows are not aligned")
        operation = _operation(reference["target"])
        aoi = str(reference["aoi_id"])
        reference_values = _values(reference)
        candidate_values = _values(candidate)
        _add(grouped[operation][aoi]["reference"], reference_values)
        _add(grouped[operation][aoi]["candidate"], candidate_values)
        _add(
            grouped[operation][aoi]["delta"],
            {
                metric: candidate_values[metric] - reference_values[metric]
                for metric in METRICS
            },
        )
    return grouped


def _combine(
    grouped: dict[str, dict[str, dict[str, Any]]], operation: str, side: str
) -> dict[str, Any]:
    result = _empty_stats()
    for record in grouped.get(operation, {}).values():
        result["count"] += record[side]["count"]
        for metric in METRICS:
            result["sums"][metric] += record[side]["sums"][metric]
    return result


def _bootstrap(
    per_seed: dict[int, dict[str, dict[str, dict[str, Any]]]],
    operation: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float]] | None:
    aois = sorted(
        set.intersection(
            *(set(grouped.get(operation, {})) for grouped in per_seed.values())
        )
    )
    if len(aois) < 2 or repetitions <= 0:
        return None
    rng = random.Random(seed)
    draws = {metric: [] for metric in METRICS}
    for _ in range(repetitions):
        sampled = rng.choices(aois, k=len(aois))
        seed_means = {metric: [] for metric in METRICS}
        for grouped in per_seed.values():
            count = sum(grouped[operation][aoi]["delta"]["count"] for aoi in sampled)
            if count <= 0:
                continue
            for metric in METRICS:
                total = sum(
                    grouped[operation][aoi]["delta"]["sums"][metric]
                    for aoi in sampled
                )
                seed_means[metric].append(total / count)
        for metric in METRICS:
            draws[metric].append(statistics.fmean(seed_means[metric]))
    return {
        metric: {
            "ci95_low": sorted(values)[int(0.025 * (len(values) - 1))],
            "ci95_high": sorted(values)[int(0.975 * (len(values) - 1))],
        }
        for metric, values in draws.items()
    }


def aggregate(
    records: list[tuple[int, Path, Path]],
    *,
    reference_label: str,
    candidate_label: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if len(records) != 3 or len({item[0] for item in records}) != 3:
        raise ValueError("exactly three unique model-seed pairs are required")
    per_seed = {
        model_seed: _read_pair(reference, candidate)
        for model_seed, reference, candidate in records
    }
    operations = {}
    for operation in OPERATIONS:
        support = [_combine(grouped, operation, "reference")["count"] for grouped in per_seed.values()]
        if not all(support) or len(set(support)) != 1:
            raise ValueError(f"inconsistent or empty support for {operation}: {support}")
        seed_rows = {}
        for model_seed, grouped in per_seed.items():
            seed_rows[str(model_seed)] = {
                "support": support[0],
                "reference": _means(_combine(grouped, operation, "reference")),
                "candidate": _means(_combine(grouped, operation, "candidate")),
                "candidate_minus_reference": _means(_combine(grouped, operation, "delta")),
            }
        operations[operation] = {
            "support_per_seed": support[0],
            "aoi_count": len(next(iter(per_seed.values()))[operation]),
            "per_seed": seed_rows,
            "candidate_minus_reference_mean": {
                metric: statistics.fmean(
                    row["candidate_minus_reference"][metric]
                    for row in seed_rows.values()
                )
                for metric in METRICS
            },
            "candidate_minus_reference_aoi_bootstrap": _bootstrap(
                per_seed, operation, repetitions=repetitions, seed=seed
            ),
        }
    return {
        "schema_version": "sn7-frozen-test-operation-slices-v1",
        "split": "test",
        "test_assets_read": True,
        "analysis_role": "posthoc_descriptive_only",
        "promotion_decision_used": False,
        "reference": reference_label,
        "candidate": candidate_label,
        "model_seeds": sorted(per_seed),
        "metrics": list(METRICS),
        "operations": operations,
        "bootstrap": {
            "unit": "aoi_id",
            "repetitions": repetitions,
            "seed": seed,
            "shared_across_model_seeds": True,
        },
        "sources": {
            str(model_seed): {
                "reference": str(reference.resolve()),
                "reference_sha256": _sha256(reference),
                "candidate": str(candidate.resolve()),
                "candidate_sha256": _sha256(candidate),
            }
            for model_seed, reference, candidate in records
        },
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# SN7 Frozen-Test Operation Slices",
        "",
        f"Comparison: `{payload['candidate']}` minus `{payload['reference']}`.",
        "This is a post-hoc descriptive analysis and cannot alter promotion.",
        "",
        "| Target edit | Support/seed | AOIs | Map quality | Quality gain | False edit | Missed edit | Cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for operation in OPERATIONS:
        row = payload["operations"][operation]
        delta = row["candidate_minus_reference_mean"]
        lines.append(
            f"| {operation} | {row['support_per_seed']} | {row['aoi_count']} | "
            f"{delta['map_quality_after']:+.6f} | {delta['map_quality_gain']:+.6f} | "
            f"{delta['false_edit']:+.6f} | {delta['missed_edit']:+.6f} | "
            f"{delta['spent_cost']:+.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pair", action="append", type=_pair, required=True)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    payload = aggregate(
        args.pair,
        reference_label=args.reference_label,
        candidate_label=args.candidate_label,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "operation_slices.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "operation_slices.md").write_text(
        render(payload), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
