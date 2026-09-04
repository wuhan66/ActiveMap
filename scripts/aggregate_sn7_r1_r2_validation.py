#!/usr/bin/env python3
"""Fail-closed three-seed aggregation for the validation-only R1/R2 queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

FIELDS = (
    ("terminal_accuracy", "terminal_correct"),
    ("false_edit_rate", "false_edit"),
    ("missed_edit_rate", "missed_edit"),
    ("mean_cost", "spent_cost"),
    ("mean_quality_gain", "quality_gain"),
    ("mean_quality_cost_utility", "quality_cost_utility"),
    ("balanced_utility", "episode_utility_v2_proxy_balanced"),
    ("safety_utility", "episode_utility_v2_proxy_safety"),
    ("cost_aware_utility", "episode_utility_v2_proxy_cost_aware"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_trace(path: Path) -> dict[tuple[str, float], dict[str, Any]]:
    rows: dict[tuple[str, float], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != "val" or row.get("test_assets_read") is not False:
            raise ValueError(f"non-validation trace row at {path}:{line_number}")
        key = (str(row["source_episode"]), float(row["budget"]))
        if key in rows:
            raise ValueError(f"duplicate trace identity in {path}: {key}")
        rows[key] = row
    if not rows:
        raise ValueError(f"empty trace: {path}")
    return rows


def _paired_deltas(
    candidate: dict[tuple[str, float], dict[str, Any]],
    reference: dict[tuple[str, float], dict[str, Any]],
) -> list[dict[str, Any]]:
    if candidate.keys() != reference.keys():
        raise ValueError("paired policies do not have identical validation support")
    rows = []
    for key in sorted(candidate):
        left, right = candidate[key], reference[key]
        if left["aoi_id"] != right["aoi_id"]:
            raise ValueError(f"AOI mismatch for {key}")
        rows.append(
            {
                "aoi_id": str(left["aoi_id"]),
                "delta": {
                    name: float(left[field]) - float(right[field])
                    for name, field in FIELDS
                },
            }
        )
    return rows


def _hierarchical_bootstrap(
    paired: dict[int, list[dict[str, Any]]], *, repetitions: int, seed: int
) -> dict[str, Any]:
    if len(paired) != 3:
        raise ValueError("R1/R2 aggregation requires exactly three controller seeds")
    rng = np.random.default_rng(seed)
    per_seed: dict[int, dict[str, list[float]]] = {}
    for controller_seed, rows in paired.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[row["aoi_id"]].append(row)
        if len(groups) < 2:
            raise ValueError("each seed requires multiple AOIs")
        per_seed[controller_seed] = {
            name: [
                sum(float(row["delta"][name]) for row in groups[aoi]) / len(groups[aoi])
                for aoi in sorted(groups)
            ]
            for name, _ in FIELDS
        }
    seeds = sorted(per_seed)
    draws = {name: np.empty(repetitions, dtype=np.float64) for name, _ in FIELDS}
    for index in range(repetitions):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        for name, _ in FIELDS:
            seed_means = []
            for controller_seed in sampled_seeds:
                values = per_seed[int(controller_seed)][name]
                sampled = rng.integers(0, len(values), size=len(values))
                seed_means.append(float(np.mean(np.asarray(values)[sampled])))
            draws[name][index] = float(np.mean(seed_means))
    observed = {
        name: float(
            np.mean([row["delta"][name] for rows in paired.values() for row in rows])
        )
        for name, _ in FIELDS
    }
    return {
        "hierarchical_bootstrap_unit": "controller_seed_then_aoi",
        "controller_seed_count": len(seeds),
        "bootstrap_repetitions": repetitions,
        "intervals": {
            name: {
                "observed_delta": observed[name],
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
            }
            for name, values in draws.items()
        },
    }


def _load_receipt(root: Path, seed: int) -> tuple[dict[str, Any], Path]:
    path = root / f"seed{seed}" / "COMPLETE.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "val" or payload.get("test_assets_read") is not False:
        raise ValueError(f"seed {seed} is not a validation-only receipt")
    if int(payload.get("controller_seed", -1)) != seed:
        raise ValueError(f"receipt seed mismatch at {path}")
    return payload, path


def aggregate(root: Path, *, repetitions: int, seed: int) -> dict[str, Any]:
    seeds = (20260730, 20260731, 20260801)
    receipts = {value: _load_receipt(root, value) for value in seeds}
    input_keys = ("train_states", "train_episodes", "val_states", "val_episodes")
    for key in input_keys:
        hashes = {payload["inputs"][key]["sha256"] for payload, _ in receipts.values()}
        if len(hashes) != 1:
            raise ValueError(f"controller seeds disagree on {key} input hash")
    manifest_hashes = {
        payload["inputs"]["edit_manifest"]["sha256"] for payload, _ in receipts.values()
    }
    if len(manifest_hashes) != 1:
        raise ValueError("controller seeds disagree on the generated R2 manifest")
    variants = sorted(next(iter(receipts.values()))[0]["natural_validation_traces"])
    if any(
        sorted(receipt["natural_validation_traces"]) != variants
        for receipt, _ in receipts.values()
    ):
        raise ValueError("controller seeds disagree on natural-validation variants")
    if any(
        sorted(receipt["r2_filtered_traces"]) != variants
        for receipt, _ in receipts.values()
    ):
        raise ValueError("controller seeds disagree on R2 variants")

    results: dict[str, Any] = {}
    analyses = (
        ("r1_natural", "natural_validation_traces"),
        ("r2_edit_only", "r2_filtered_traces"),
    )
    for analysis, key in analyses:
        comparisons = {}
        for reference in variants:
            if reference == "activemap":
                continue
            paired = {}
            for controller_seed, (receipt, _) in receipts.items():
                candidate_path = Path(receipt[key]["activemap"]["path"])
                reference_path = Path(receipt[key][reference]["path"])
                if _sha256(candidate_path) != receipt[key]["activemap"]["sha256"]:
                    raise ValueError(f"candidate trace hash mismatch for seed {controller_seed}")
                if _sha256(reference_path) != receipt[key][reference]["sha256"]:
                    raise ValueError(f"reference trace hash mismatch for seed {controller_seed}")
                paired[controller_seed] = _paired_deltas(
                    _read_trace(candidate_path), _read_trace(reference_path)
                )
            comparisons[f"activemap_minus_{reference}"] = _hierarchical_bootstrap(
                paired, repetitions=repetitions, seed=seed
            )
        results[analysis] = comparisons
    return {
        "schema_version": "activemap-r1-r2-validation-three-seed-v1",
        "split": "val",
        "test_assets_read": False,
        "seeds": list(seeds),
        "input_hashes": {
            key: next(
                iter({payload["inputs"][key]["sha256"] for payload, _ in receipts.values()})
            )
            for key in input_keys
        },
        "r2_manifest_sha256": next(iter(manifest_hashes)),
        "variants": variants,
        "results": results,
        "receipts": {
            str(controller_seed): {"path": str(path.resolve()), "sha256": _sha256(path)}
            for controller_seed, (_, path) in receipts.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.bootstrap_repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    result = aggregate(
        args.run_root,
        repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
