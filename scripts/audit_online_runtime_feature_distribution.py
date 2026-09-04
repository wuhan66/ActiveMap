#!/usr/bin/env python3
"""Compare frozen-selector training support with exact online runtime inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from activemap.selector_records import SelectorSample


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reservoir_training_samples(
    path: Path,
    *,
    split: str,
    limit: int,
    seed: int,
) -> tuple[list[SelectorSample], int]:
    rng = random.Random(seed)
    reservoir: list[SelectorSample] = []
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                sample = SelectorSample.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid training record at line {line_number}") from exc
            if sample.split != split:
                continue
            count += 1
            if len(reservoir) < limit:
                reservoir.append(sample)
            else:
                replacement = rng.randrange(count)
                if replacement < limit:
                    reservoir[replacement] = sample
    if not reservoir:
        raise ValueError(f"no {split} records in {path}")
    return reservoir, count


def load_runtime_rows(
    path: Path,
    *,
    policy: str | None,
    step: int | None,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid runtime record at line {line_number}") from exc
            if row.get("test_assets_read") is not False:
                raise ValueError("runtime diagnostics must explicitly prohibit test access")
            if policy is not None and row.get("policy") != policy:
                continue
            if step is not None and int(row.get("step", -1)) != step:
                continue
            total += 1
            if row.get("selector_invoked"):
                rows.append(row)
    if not rows:
        raise ValueError("no selector-invoked runtime records match the requested policy")
    return rows, total


def scalar_summary(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty array")
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def feature_block(
    training: np.ndarray,
    runtime: np.ndarray,
) -> dict[str, Any]:
    training = np.asarray(training, dtype=np.float64)
    runtime = np.asarray(runtime, dtype=np.float64)
    if training.ndim != 2 or runtime.ndim != 2 or training.shape[1] != runtime.shape[1]:
        raise ValueError("training and runtime feature matrices must share a feature dimension")
    train_mean = training.mean(axis=0)
    train_std = np.maximum(training.std(axis=0), 1e-6)
    runtime_mean = runtime.mean(axis=0)
    low = np.quantile(training, 0.01, axis=0)
    high = np.quantile(training, 0.99, axis=0)
    standardized_shift = (runtime_mean - train_mean) / train_std
    outside = np.mean((runtime < low) | (runtime > high), axis=0)
    dimensions = [
        {
            "index": index,
            "training_mean": float(train_mean[index]),
            "training_std": float(train_std[index]),
            "runtime_mean": float(runtime_mean[index]),
            "standardized_mean_shift": float(standardized_shift[index]),
            "runtime_outside_training_p01_p99_rate": float(outside[index]),
        }
        for index in range(training.shape[1])
    ]
    return {
        "training_row_count": int(training.shape[0]),
        "runtime_row_count": int(runtime.shape[0]),
        "dimensions": dimensions,
        "largest_absolute_shifts": sorted(
            dimensions,
            key=lambda row: abs(row["standardized_mean_shift"]),
            reverse=True,
        )[:5],
    }


def sample_total_evidence_count(sample: SelectorSample) -> int:
    predictions = sample.metadata.get("evidence_predictions")
    prediction_ids = set(predictions) if isinstance(predictions, dict) else set()
    return len(
        set(sample.evidence_ids)
        | set(sample.metadata.get("selected_evidence_ids", []))
        | prediction_ids
    )


def audit(
    training_manifest: Path,
    runtime_inputs: Path,
    *,
    policy: str | None,
    training_split: str,
    training_sample_limit: int,
    seed: int,
    runtime_step: int | None = None,
) -> dict[str, Any]:
    training, training_population = reservoir_training_samples(
        training_manifest,
        split=training_split,
        limit=training_sample_limit,
        seed=seed,
    )
    runtime, runtime_population = load_runtime_rows(
        runtime_inputs, policy=policy, step=runtime_step
    )

    training_hypothesis = np.asarray(
        [sample.hypothesis_features for sample in training], dtype=np.float64
    )
    runtime_hypothesis = np.asarray(
        [row["hypothesis_features"] for row in runtime], dtype=np.float64
    )
    training_state = np.asarray(
        [sample.state_features for sample in training], dtype=np.float64
    )
    runtime_state = np.asarray(
        [row["state_features"] for row in runtime], dtype=np.float64
    )
    training_evidence = np.concatenate(
        [np.asarray(sample.evidence_features, dtype=np.float64) for sample in training], axis=0
    )
    runtime_evidence = np.concatenate(
        [np.asarray(row["evidence_features"], dtype=np.float64) for row in runtime], axis=0
    )

    training_candidate_counts = [len(sample.evidence_ids) for sample in training]
    runtime_candidate_counts = [int(row["candidate_count"]) for row in runtime]
    training_selected_counts = [
        len(sample.metadata.get("selected_evidence_ids", [])) for sample in training
    ]
    runtime_selected_counts = [len(row["selected_evidence_ids"]) for row in runtime]
    runtime_gap = [
        float(row["candidate_minus_stop"])
        for row in runtime
        if row.get("candidate_minus_stop") is not None
    ]
    eligible = np.asarray(
        [float(row["runtime_grid_eligible_count"]) for row in runtime], dtype=np.float64
    )
    excluded = np.asarray(
        [float(row["runtime_grid_excluded_count"]) for row in runtime], dtype=np.float64
    )
    support = {
        "training_candidate_count": scalar_summary(training_candidate_counts),
        "runtime_candidate_count": scalar_summary(runtime_candidate_counts),
        "training_selected_count": scalar_summary(training_selected_counts),
        "runtime_selected_count": scalar_summary(runtime_selected_counts),
        "training_total_evidence_count": scalar_summary(
            [sample_total_evidence_count(sample) for sample in training]
        ),
        "runtime_total_evidence_count": scalar_summary(
            [float(row["total_evidence_count"]) for row in runtime]
        ),
        "runtime_source_catalog_candidate_count": scalar_summary(
            [float(row["source_catalog_candidate_count"]) for row in runtime]
        ),
        "runtime_grid_eligible_fraction": scalar_summary(
            eligible / np.maximum(eligible + excluded, 1.0)
        ),
        "runtime_affordable_fraction_of_grid_eligible": scalar_summary(
            np.asarray(runtime_candidate_counts, dtype=np.float64) / np.maximum(eligible, 1.0)
        ),
        "training_evidence_cost": scalar_summary(
            np.concatenate(
                [np.asarray(sample.evidence_costs, dtype=np.float64) for sample in training]
            )
        ),
        "runtime_evidence_cost": scalar_summary(
            np.concatenate(
                [np.asarray(row["evidence_costs"], dtype=np.float64) for row in runtime]
            )
        ),
        "runtime_candidate_minus_stop": scalar_summary(runtime_gap),
        "runtime_positive_candidate_margin_rate": float(
            np.mean(np.asarray(runtime_gap) > 0.0)
        ),
    }
    return {
        "schema_version": "activemap-online-runtime-feature-audit-v1",
        "training_manifest": {
            "path": str(training_manifest.resolve()),
            "sha256": sha256_path(training_manifest),
            "split": training_split,
            "population_count": training_population,
            "sampled_count": len(training),
        },
        "runtime_inputs": {
            "path": str(runtime_inputs.resolve()),
            "sha256": sha256_path(runtime_inputs),
            "policy": policy,
            "step": runtime_step,
            "matching_record_count": runtime_population,
            "selector_invoked_record_count": len(runtime),
        },
        "feature_distribution": {
            "hypothesis": feature_block(training_hypothesis, runtime_hypothesis),
            "state": feature_block(training_state, runtime_state),
            "evidence": feature_block(training_evidence, runtime_evidence),
        },
        "support": support,
        "test_assets_read": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_manifest", type=Path)
    parser.add_argument("runtime_inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--policy", default="active_selective_safe")
    parser.add_argument("--training-split", choices=("train", "val"), default="train")
    parser.add_argument("--training-sample-limit", type=int, default=5000)
    parser.add_argument("--runtime-step", type=int)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.training_sample_limit <= 0:
        raise ValueError("--training-sample-limit must be positive")
    result = audit(
        args.training_manifest,
        args.runtime_inputs,
        policy=args.policy,
        training_split=args.training_split,
        training_sample_limit=args.training_sample_limit,
        seed=args.seed,
        runtime_step=args.runtime_step,
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    compact = {
        "schema_version": result["schema_version"],
        "training_manifest": result["training_manifest"],
        "runtime_inputs": result["runtime_inputs"],
        "support": result["support"],
        "largest_feature_shifts": {
            name: payload["largest_absolute_shifts"]
            for name, payload in result["feature_distribution"].items()
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
