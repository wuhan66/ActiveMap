#!/usr/bin/env python3
"""Fit a train-only decision stump that safely arbitrates terminal edits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in rows:
                raise ValueError(f"duplicate sample ID: {sample_id}")
            rows[sample_id] = row
    if not rows:
        raise ValueError(f"empty rollout trace: {path}")
    return rows


def observable_features(row: dict[str, Any]) -> dict[str, float]:
    events = row.get("events") or []
    if not events:
        raise ValueError(f"trace has no observable events: {row['sample_id']}")
    state = events[-1]["observable_state"]
    direct = state["direct_draft"]
    belief = state["belief"]
    probabilities = sorted(
        (float(value) for value in belief["edit_probabilities"]), reverse=True
    )
    prediction = str(row["predicted_edit"])
    confidence = float(direct["confidence"])
    return {
        "direct_confidence": confidence,
        "belief_uncertainty": float(belief["uncertainty"]),
        "belief_probability_max": probabilities[0],
        "belief_probability_margin": probabilities[0] - probabilities[1],
        "spent_cost": float(row["spent_cost"]),
        "tool_calls": float(row["tool_calls"]),
        "acquisitions": float(row["acquisitions"]),
        "is_add": float(prediction == "ADD"),
        "is_delete": float(prediction == "DELETE"),
        "is_reshape": float(prediction == "RESHAPE"),
        "delete_confidence": confidence if prediction == "DELETE" else 0.0,
        "nonkeep_confidence": confidence if prediction != "KEEP" else 0.0,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    count = len(rows)
    return {
        "episodes": float(count),
        "terminal_accuracy": sum(bool(row["terminal_correct"]) for row in rows)
        / count,
        "false_edit_rate": sum(bool(row["false_edit"]) for row in rows) / count,
        "missed_edit_rate": sum(bool(row["missed_edit"]) for row in rows) / count,
        "wrong_edit_rate": sum(bool(row["wrong_edit"]) for row in rows) / count,
        "mean_cost": sum(float(row["spent_cost"]) for row in rows) / count,
        "mean_tool_calls": sum(float(row["tool_calls"]) for row in rows) / count,
        "balanced_utility": sum(
            float(row["episode_utility_v2_proxy_balanced"]) for row in rows
        )
        / count,
        "safety_utility": sum(
            float(row["episode_utility_v2_proxy_safety"]) for row in rows
        )
        / count,
    }


def paired_rows(
    baseline: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if baseline.keys() != candidate.keys():
        raise ValueError("baseline and candidate traces have different sample IDs")
    return [(baseline[key], candidate[key]) for key in sorted(baseline)]


def apply_gate(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    feature: str,
    direction: str,
    threshold: float,
) -> tuple[list[dict[str, Any]], int, int]:
    selected = []
    disagreements = 0
    accepted = 0
    for baseline, candidate in pairs:
        disagreement = baseline["predicted_edit"] != candidate["predicted_edit"]
        disagreements += int(disagreement)
        value = observable_features(candidate)[feature]
        passed = value >= threshold if direction == "ge" else value <= threshold
        # Arbitration is needed only when terminal edits disagree. Preserve the
        # candidate's lower-cost trajectory whenever both policies agree.
        use_candidate = not disagreement or passed
        accepted += int(disagreement and passed)
        source = candidate if use_candidate else baseline
        selected.append(
            {
                **source,
                "terminal_arbiter": {
                    "feature": feature,
                    "direction": direction,
                    "threshold": threshold,
                    "observed_value": value,
                    "candidate_accepted": use_candidate,
                    "uses_ground_truth_at_inference": False,
                },
            }
        )
    return selected, disagreements, accepted


def candidate_thresholds(values: list[float]) -> list[float]:
    ordered = sorted(set(values))
    if not ordered:
        return [0.0]
    epsilon = 1e-9
    return [ordered[0] - epsilon, *ordered, ordered[-1] + epsilon]


def fit_gate(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]]
) -> tuple[dict[str, Any], dict[str, float]]:
    baseline_summary = summarize([baseline for baseline, _ in pairs])
    best = None
    for feature in observable_features(pairs[0][1]):
        values = [
            observable_features(candidate)[feature]
            for baseline, candidate in pairs
            if baseline["predicted_edit"] != candidate["predicted_edit"]
        ]
        for threshold in candidate_thresholds(values):
            for direction in ("ge", "le"):
                rows, disagreements, accepted = apply_gate(
                    pairs,
                    feature=feature,
                    direction=direction,
                    threshold=threshold,
                )
                metrics = summarize(rows)
                safe = (
                    metrics["false_edit_rate"]
                    <= baseline_summary["false_edit_rate"] + 1e-12
                )
                key = (
                    safe,
                    metrics["balanced_utility"],
                    metrics["terminal_accuracy"],
                    -metrics["missed_edit_rate"],
                    -accepted,
                )
                if best is None or key > best[0]:
                    best = (
                        key,
                        {
                            "schema_version": "active-catalog-terminal-arbiter-v1",
                            "fit_split": "train",
                            "feature": feature,
                            "direction": direction,
                            "threshold": threshold,
                            "train_disagreements": disagreements,
                            "train_accepted": accepted,
                            "uses_ground_truth_at_inference": False,
                        },
                        metrics,
                    )
    assert best is not None
    return best[1], best[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_train", type=Path)
    parser.add_argument("candidate_train", type=Path)
    parser.add_argument("baseline_val", type=Path)
    parser.add_argument("candidate_val", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    train_pairs = paired_rows(
        load_rows(args.baseline_train), load_rows(args.candidate_train)
    )
    gate, train_metrics = fit_gate(train_pairs)
    val_pairs = paired_rows(load_rows(args.baseline_val), load_rows(args.candidate_val))
    val_rows, val_disagreements, val_accepted = apply_gate(
        val_pairs,
        feature=gate["feature"],
        direction=gate["direction"],
        threshold=float(gate["threshold"]),
    )
    baseline_val = summarize([baseline for baseline, _ in val_pairs])
    candidate_val = summarize([candidate for _, candidate in val_pairs])
    hybrid_val = summarize(val_rows)
    summary = {
        "schema_version": "active-catalog-terminal-arbitration-result-v1",
        "gate": gate,
        "train": {
            "baseline": summarize([baseline for baseline, _ in train_pairs]),
            "candidate": summarize([candidate for _, candidate in train_pairs]),
            "hybrid": train_metrics,
        },
        "val": {
            "disagreements": val_disagreements,
            "accepted": val_accepted,
            "baseline": baseline_val,
            "candidate": candidate_val,
            "hybrid": hybrid_val,
            "hybrid_minus_baseline": {
                key: hybrid_val[key] - baseline_val[key] for key in baseline_val
            },
        },
        "selection_protocol": "train-only threshold fit; frozen validation application",
        "test_assets_read": False,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in val_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
