#!/usr/bin/env python3
"""Audit whether a candidate bank can support safe non-KEEP edit recovery.

The audit runs before selector training. It uses executable candidate outcomes
only as an oracle upper bound: if no candidate safely improves an ADD, DELETE,
or RESHAPE episode over the fixed direct anchor, no selection policy can recover
that edit with the frozen perception/candidate interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from activemap.selector_records import SelectorSample


OPERATIONS = ("ADD", "DELETE", "RESHAPE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_samples(path: Path, *, split: str) -> list[SelectorSample]:
    samples: list[SelectorSample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                sample = SelectorSample.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid selector state at line {line_number}") from exc
            if sample.split == split:
                samples.append(sample)
    if not samples:
        raise ValueError(f"no selector states found for split={split!r}")
    return samples


def operation(value: Any) -> str:
    text = str(value or "KEEP").upper().replace("-", "_").replace(" ", "")
    text = text.removeprefix("COMMIT:")
    if text in {"", "STOP", "REJECT", "KEEP"}:
        return "KEEP"
    if text in OPERATIONS:
        return text
    raise ValueError(f"unsupported operation: {value!r}")


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def outcome(sample: Any, evidence_id: str) -> dict[str, Any]:
    outcomes = sample.metadata.get("executable_outcomes")
    if not isinstance(outcomes, dict):
        raise ValueError(f"{sample.sample_id}: missing executable outcomes")
    result = outcomes.get(evidence_id)
    if not isinstance(result, dict):
        raise ValueError(f"{sample.sample_id}: missing outcome for {evidence_id}")
    required = {"final_raster_iou", "false_edit", "missed_edit"}
    missing = required - set(result)
    if missing:
        raise ValueError(
            f"{sample.sample_id}: outcome for {evidence_id} lacks {sorted(missing)}"
        )
    return result


def predicted_operation(sample: Any, evidence_id: str) -> str:
    predictions = sample.metadata.get("evidence_predictions", {})
    if not isinstance(predictions, dict):
        return "UNKNOWN"
    prediction = predictions.get(evidence_id, {})
    if not isinstance(prediction, dict):
        return "UNKNOWN"
    return operation(prediction.get("gated_edit", "KEEP"))


def _safe(candidate: dict[str, Any], anchor: dict[str, Any]) -> bool:
    return (
        bool_value(candidate["false_edit"]) <= bool_value(anchor["false_edit"])
        and bool_value(candidate["missed_edit"]) <= bool_value(anchor["missed_edit"])
    )


def record(sample: Any, *, headroom_epsilon: float) -> dict[str, Any] | None:
    if sample.split == "test" or sample.metadata.get("test_assets_read") is True:
        raise ValueError(f"{sample.sample_id}: headroom audit forbids test provenance")
    target = operation(sample.metadata.get("gt_edit"))
    if target == "KEEP":
        return None
    anchor_id = sample.metadata.get("initial_evidence_id")
    outcomes = sample.metadata.get("executable_outcomes")
    if not isinstance(outcomes, dict):
        raise ValueError(f"{sample.sample_id}: missing executable outcomes")
    if not isinstance(anchor_id, str) or anchor_id not in outcomes:
        raise ValueError(f"{sample.sample_id}: missing valid initial evidence anchor")
    anchor = outcome(sample, anchor_id)
    candidates = []
    # Selector training states expose only currently affordable actions. The
    # immutable outcome registry retains the complete candidate bank required
    # for this pre-policy oracle feasibility bound.
    for evidence_id in sorted(outcomes):
        candidate = outcome(sample, evidence_id)
        candidates.append(
            {
                "evidence_id": evidence_id,
                "final_raster_iou": float(candidate["final_raster_iou"]),
                "false_edit": bool_value(candidate["false_edit"]),
                "missed_edit": bool_value(candidate["missed_edit"]),
                "predicted_operation": predicted_operation(sample, evidence_id),
            }
        )
    anchor_row = next(item for item in candidates if item["evidence_id"] == anchor_id)
    safe_candidates = [item for item in candidates if _safe(item, anchor)]
    # The direct anchor is always safe, so an empty list indicates malformed input.
    if not safe_candidates:
        raise ValueError(f"{sample.sample_id}: direct anchor is absent from candidate bank")
    oracle = max(
        safe_candidates,
        key=lambda item: (item["final_raster_iou"], -item["missed_edit"], item["evidence_id"]),
    )
    headroom = oracle["final_raster_iou"] - anchor_row["final_raster_iou"]
    outcome_signatures = {
        (
            item["predicted_operation"],
            round(item["final_raster_iou"], 8),
            item["false_edit"],
            item["missed_edit"],
        )
        for item in candidates
    }
    return {
        "sample_id": str(sample.sample_id),
        "source_episode": str(sample.metadata.get("source_episode", sample.sample_id)),
        "aoi_id": str(sample.metadata.get("aoi_id", "unknown")),
        "target_operation": target,
        "candidate_count": len(candidates),
        "safe_candidate_count": len(safe_candidates),
        "distinct_outcome_count": len(outcome_signatures),
        "anchor": anchor_row,
        "safe_oracle": oracle,
        "safe_map_headroom": headroom,
        "safe_map_recovery_possible": headroom > headroom_epsilon,
        "missed_edit_recovery_possible": (
            anchor_row["missed_edit"] and not oracle["missed_edit"]
        ),
        "test_assets_read": False,
    }


def deduplicate_source_episodes(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Collapse budget/step replicas while rejecting inconsistent outcome banks."""

    unique: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for row in rows:
        source_episode = str(row["source_episode"])
        previous = unique.get(source_episode)
        if previous is None:
            unique[source_episode] = row
            continue
        duplicate_count += 1
        comparable = (
            "aoi_id",
            "target_operation",
            "candidate_count",
            "safe_candidate_count",
            "distinct_outcome_count",
            "anchor",
            "safe_oracle",
            "safe_map_headroom",
            "safe_map_recovery_possible",
            "missed_edit_recovery_possible",
        )
        if any(previous[key] != row[key] for key in comparable):
            raise ValueError(
                f"inconsistent candidate-bank outcome replicas for {source_episode}"
            )
    return [unique[key] for key in sorted(unique)], duplicate_count


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def aoi_bootstrap(
    rows: list[dict[str, Any]], *, metric: str, draws: int, seed: int
) -> tuple[float, float]:
    by_aoi: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_aoi[str(row["aoi_id"])].append(float(row[metric]))
    names = sorted(by_aoi)
    if not names:
        raise ValueError("empty AOI support")
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        selected = [rng.choice(names) for _ in names]
        samples.append(
            statistics.fmean(
                statistics.fmean(by_aoi[name]) for name in selected
            )
        )
    return quantile(samples, 0.025), quantile(samples, 0.975)


def summarize(
    rows: list[dict[str, Any]],
    *,
    minimum_rows: int,
    minimum_aois: int,
    headroom_epsilon: float,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("no non-KEEP records")
    by_operation: dict[str, list[dict[str, Any]]] = {
        operation_name: [
            row for row in rows if row["target_operation"] == operation_name
        ]
        for operation_name in OPERATIONS
    }
    slices: dict[str, dict[str, Any]] = {}
    for index, (operation_name, operation_rows) in enumerate(by_operation.items()):
        aoi_count = len({str(row["aoi_id"]) for row in operation_rows})
        if operation_rows:
            headroom_ci = aoi_bootstrap(
                operation_rows,
                metric="safe_map_headroom",
                draws=bootstrap_draws,
                seed=bootstrap_seed + index,
            )
            summary = {
                "row_count": len(operation_rows),
                "aoi_count": aoi_count,
                "mean_safe_map_headroom": statistics.fmean(
                    float(row["safe_map_headroom"]) for row in operation_rows
                ),
                "safe_map_headroom_ci95": list(headroom_ci),
                "safe_map_recovery_fraction": statistics.fmean(
                    float(bool(row["safe_map_recovery_possible"]))
                    for row in operation_rows
                ),
                "missed_edit_recovery_fraction": statistics.fmean(
                    float(bool(row["missed_edit_recovery_possible"]))
                    for row in operation_rows
                ),
                "candidate_outcome_diversity_fraction": statistics.fmean(
                    float(int(row["distinct_outcome_count"]) > 1)
                    for row in operation_rows
                ),
                "mean_safe_candidate_count": statistics.fmean(
                    float(row["safe_candidate_count"]) for row in operation_rows
                ),
            }
        else:
            summary = {
                "row_count": 0,
                "aoi_count": 0,
                "mean_safe_map_headroom": 0.0,
                "safe_map_headroom_ci95": [0.0, 0.0],
                "safe_map_recovery_fraction": 0.0,
                "missed_edit_recovery_fraction": 0.0,
                "candidate_outcome_diversity_fraction": 0.0,
                "mean_safe_candidate_count": 0.0,
            }
        summary["passes_preflight"] = (
            summary["row_count"] >= minimum_rows
            and summary["aoi_count"] >= minimum_aois
            and summary["candidate_outcome_diversity_fraction"] > 0.0
            and summary["safe_map_headroom_ci95"][0] > headroom_epsilon
        )
        slices[operation_name] = summary
    overall_ci = aoi_bootstrap(
        rows,
        metric="safe_map_headroom",
        draws=bootstrap_draws,
        seed=bootstrap_seed + len(OPERATIONS),
    )
    gate = {
        "minimum_rows_per_operation": minimum_rows,
        "minimum_aois_per_operation": minimum_aois,
        "headroom_epsilon": headroom_epsilon,
        "passes_by_operation": {
            operation_name: slices[operation_name]["passes_preflight"]
            for operation_name in OPERATIONS
        },
    }
    gate["passes_candidate_recovery_preflight"] = all(
        gate["passes_by_operation"].values()
    )
    return {
        "schema_version": "sn7-nonkeep-candidate-headroom-v1",
        "row_count": len(rows),
        "aoi_count": len({str(row["aoi_id"]) for row in rows}),
        "mean_safe_map_headroom": statistics.fmean(
            float(row["safe_map_headroom"]) for row in rows
        ),
        "safe_map_headroom_ci95": list(overall_ci),
        "slices": slices,
        "gate": gate,
        "interpretation": (
            "This is an oracle candidate-bank feasibility audit, not a policy result. "
            "A failure means the current frozen candidate interface cannot justify "
            "new selector/controller training for the failed operation."
        ),
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("states", type=Path, help="train/validation selector state JSONL")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--minimum-rows", type=int, default=20)
    parser.add_argument("--minimum-aois", type=int, default=4)
    parser.add_argument("--headroom-epsilon", type=float, default=1e-6)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.minimum_rows < 1 or args.minimum_aois < 1:
        raise ValueError("minimum support must be positive")
    if args.headroom_epsilon < 0.0:
        raise ValueError("headroom epsilon must be non-negative")
    if args.bootstrap_draws < 100:
        raise ValueError("bootstrap draws must be at least 100")

    samples = load_samples(args.states, split=args.split)
    raw_rows = [
        item
        for sample in samples
        if (item := record(sample, headroom_epsilon=args.headroom_epsilon)) is not None
    ]
    rows, duplicate_state_count = deduplicate_source_episodes(raw_rows)
    summary = summarize(
        rows,
        minimum_rows=args.minimum_rows,
        minimum_aois=args.minimum_aois,
        headroom_epsilon=args.headroom_epsilon,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary.update(
        {
            "states": str(args.states.resolve()),
            "states_sha256": sha256(args.states),
            "split": args.split,
            "input_state_count": len(samples),
            "nonkeep_input_state_count": len(raw_rows),
            "duplicate_state_count": duplicate_state_count,
            "unique_source_episode_count": len(rows),
            "bootstrap_draws": args.bootstrap_draws,
            "bootstrap_seed": args.bootstrap_seed,
        }
    )
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "per_episode.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
