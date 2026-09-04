#!/usr/bin/env python3
"""Build aligned controller robustness slices without reopening test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.compare_active_catalog_closed_loop import (
    load_rows,
    metric_delta,
    paired_aoi_bootstrap,
)
from scripts.evaluate_active_catalog_closed_loop import metrics

CORE_METRICS = (
    "terminal_accuracy",
    "false_edit_rate",
    "missed_edit_rate",
    "mean_quality_gain",
    "mean_cost",
    "mean_quality_cost_utility",
)


def _method(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("method must be LABEL=TRACE")
    return label, Path(raw_path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ring_area(ring: list[list[float]]) -> float:
    if len(ring) < 3:
        return 0.0
    return abs(
        math.fsum(
            float(ring[index][0]) * float(ring[(index + 1) % len(ring)][1])
            - float(ring[(index + 1) % len(ring)][0]) * float(ring[index][1])
            for index in range(len(ring))
        )
    ) / 2.0


def geometry_area(geometry: dict[str, Any] | None) -> float:
    if not geometry:
        return 0.0
    coordinates = geometry.get("coordinates", [])
    if geometry.get("type") == "Polygon":
        return max(0.0, _ring_area(coordinates[0]) - math.fsum(
            _ring_area(ring) for ring in coordinates[1:]
        ))
    if geometry.get("type") == "MultiPolygon":
        return math.fsum(
            max(0.0, _ring_area(polygon[0]) - math.fsum(
                _ring_area(ring) for ring in polygon[1:]
            ))
            for polygon in coordinates
        )
    return 0.0


def _episodes(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "val":
                continue
            result[str(row["episode_id"])] = row
    if not result:
        raise ValueError("no validation episodes")
    return result


def _initial_state(row: dict[str, Any]) -> dict[str, Any]:
    for event in row.get("events", []):
        state = event.get("observable_state")
        if isinstance(state, dict):
            return state
    raise ValueError(f"trace lacks observable initial state: {row['sample_id']}")


def _continuous_metadata(
    row: dict[str, Any], episode: dict[str, Any]
) -> dict[str, float]:
    state = _initial_state(row)
    candidates = state.get("candidate_evidence", [])
    clear = [
        float(item["clear_fraction"])
        for item in candidates
        if item.get("clear_fraction") is not None
    ]
    geometry = episode.get("prior_geometry") or episode.get("target_geometry")
    return {
        "object_area": geometry_area(geometry),
        "image_quality": float(np.mean(clear)) if clear else 0.0,
        "prior_confidence": float(state["direct_draft"]["confidence"]),
        "prior_uncertainty": float(state["belief"]["uncertainty"]),
    }


def _quantile_labels(
    values: dict[tuple[str, float], float],
) -> tuple[dict[tuple[str, float], str], dict[str, Any]]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    low, high = (float(value) for value in np.quantile(array, [1 / 3, 2 / 3]))
    if math.isclose(low, high, abs_tol=1e-12):
        return (
            {key: "all" for key in values},
            {"collapsed": True, "edges": [low, high]},
        )
    labels = {
        key: ("low" if value <= low else "mid" if value <= high else "high")
        for key, value in values.items()
    }
    return labels, {"collapsed": False, "edges": [low, high]}


def build_slices(
    method_paths: dict[str, Path],
    episodes_path: Path,
    *,
    reference: str,
    candidate: str,
    expected_records: int,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if reference not in method_paths or candidate not in method_paths:
        raise ValueError("reference and candidate must be present")
    rows = {label: load_rows(path) for label, path in method_paths.items()}
    support = set(rows[reference])
    if len(support) != expected_records:
        raise ValueError(f"expected {expected_records} aligned records")
    if any(set(values) != support for values in rows.values()):
        raise ValueError("method traces do not share episode-budget support")
    episodes = _episodes(episodes_path)
    reference_rows = rows[reference]
    continuous: dict[str, dict[tuple[str, float], float]] = {
        name: {} for name in (
            "object_area",
            "image_quality",
            "prior_confidence",
            "prior_uncertainty",
        )
    }
    categorical: dict[str, dict[tuple[str, float], str]] = {
        "edit_type": {},
        "budget": {},
        "aoi": {},
    }
    for key, row in reference_rows.items():
        episode = episodes[str(row["source_episode"])]
        values = _continuous_metadata(row, episode)
        for name, value in values.items():
            continuous[name][key] = value
        categorical["edit_type"][key] = str(row["target_edit"])
        categorical["budget"][key] = f"{float(row['budget']):g}"
        categorical["aoi"][key] = str(row["aoi_id"])

    quantile_protocol = {}
    for name, values in continuous.items():
        categorical[name], quantile_protocol[name] = _quantile_labels(values)

    report: dict[str, Any] = {}
    for dimension, labels in categorical.items():
        dimension_rows = {}
        for value in sorted(set(labels.values())):
            keys = [key for key in support if labels[key] == value]
            method_metrics = {
                label: metrics([method_rows[key] for key in keys])
                for label, method_rows in rows.items()
            }
            candidate_rows = {key: rows[candidate][key] for key in keys}
            reference_slice = {key: rows[reference][key] for key in keys}
            observed = metric_delta(
                list(candidate_rows.values()), list(reference_slice.values())
            )
            aoi_count = len({str(reference_rows[key]["aoi_id"]) for key in keys})
            paired = (
                paired_aoi_bootstrap(
                    candidate_rows,
                    reference_slice,
                    repetitions=repetitions,
                    seed=seed,
                )
                if aoi_count >= 2 and repetitions > 0
                else None
            )
            dimension_rows[value] = {
                "record_count": len(keys),
                "aoi_count": aoi_count,
                "methods": method_metrics,
                "candidate_minus_reference_observed": {
                    name: observed[name] for name in CORE_METRICS
                },
                "candidate_minus_reference_aoi_bootstrap": paired,
            }
        report[dimension] = dimension_rows

    return {
        "schema_version": "active-catalog-robustness-slices-v1",
        "split": "val",
        "record_count": len(support),
        "reference": reference,
        "candidate": candidate,
        "dimensions": report,
        "quantile_protocol": quantile_protocol,
        "bootstrap": {
            "unit": "aoi_id",
            "repetitions": repetitions,
            "seed": seed,
            "omitted_for_single_aoi_slices": True,
        },
        "sources": {
            "episodes": {
                "path": str(episodes_path.resolve()),
                "sha256": _sha256(episodes_path),
            },
            "methods": {
                label: {"path": str(path.resolve()), "sha256": _sha256(path)}
                for label, path in method_paths.items()
            },
        },
        "claim_boundary": (
            "observed validation strata; prior corruption and geometry jitter "
            "require separate intervention experiments"
        ),
        "test_assets_read": False,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ActiveMap Robustness Slices",
        "",
        (
            f"Candidate `{payload['candidate']}` minus reference "
            f"`{payload['reference']}` on aligned validation support."
        ),
        "",
        "| Dimension | Slice | N | AOIs | Accuracy | False edit | Missed edit | Quality | Cost | Q-cost utility |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dimension, slices in payload["dimensions"].items():
        for label, row in slices.items():
            delta = row["candidate_minus_reference_observed"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        dimension,
                        label,
                        str(row["record_count"]),
                        str(row["aoi_count"]),
                        *(f"{delta[name]:.6f}" for name in CORE_METRICS),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Continuous slices use quantile edges frozen on the aligned reference support.",
            "AOI bootstrap intervals are stored in JSON and omitted for single-AOI slices.",
            "These are observed strata, not synthetic corruption experiments.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--method", action="append", type=_method, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--expected-records", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    methods = dict(args.method)
    if len(methods) != len(args.method):
        raise ValueError("duplicate method labels")
    payload = build_slices(
        methods,
        args.episodes,
        reference=args.reference,
        candidate=args.candidate,
        expected_records=args.expected_records,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "robustness_slices.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "robustness_slices.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
