#!/usr/bin/env python3
"""Evaluate candidate ranking and VLM/ranker safety compositions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from activemap.agent.active_catalog_candidate_ranker import CandidateUtilityRankerPredictor

try:
    from scripts.evaluate_active_catalog_selector import active_catalog_metrics, grouped_bootstrap
    from scripts.train_active_catalog_candidate_ranker import load_examples
except ModuleNotFoundError:
    from evaluate_active_catalog_selector import active_catalog_metrics, grouped_bootstrap
    from train_active_catalog_candidate_ranker import load_examples


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _decision(
    policy: str,
    *,
    scores: dict[str, float],
    margin: float,
    vlm: dict[str, Any],
) -> tuple[str, str | None]:
    best_id = max(scores, key=lambda evidence_id: (scores[evidence_id], evidence_id))
    ranker_call = scores[best_id] > margin
    vlm_call = vlm["predicted_selection"] == "ACQUIRE"
    vlm_id = vlm.get("predicted_evidence_id")
    if policy == "ranker_only":
        return ("ACQUIRE", best_id) if ranker_call else ("STOP", None)
    if policy == "vlm_gate_ranker_candidate":
        return ("ACQUIRE", best_id) if vlm_call else ("STOP", None)
    if policy == "vlm_gate_ranker_candidate_vetoed":
        return ("ACQUIRE", best_id) if vlm_call and ranker_call else ("STOP", None)
    if policy == "ranker_guarded_vlm_candidate":
        accepted = vlm_call and vlm_id in scores and scores[str(vlm_id)] > margin
        return ("ACQUIRE", str(vlm_id)) if accepted else ("STOP", None)
    if policy == "vlm_original":
        return str(vlm["predicted_selection"]), vlm_id
    raise ValueError(f"unknown policy: {policy}")


def build_policy_traces(
    examples: list[dict[str, Any]],
    predictor: CandidateUtilityRankerPredictor,
    vlm_rows: list[dict[str, Any]],
    policy: str,
) -> list[dict[str, Any]]:
    vlm_by_id = {str(row["example_id"]): row for row in vlm_rows}
    if len(vlm_by_id) != len(vlm_rows):
        raise ValueError("duplicate VLM trace example_id")
    if {str(row["example_id"]) for row in examples} != set(vlm_by_id):
        raise ValueError("ranker validation and VLM traces do not have identical examples")
    traces = []
    for example in examples:
        example_id = str(example["example_id"])
        vlm = vlm_by_id[example_id]
        scores = predictor.score_state(example["state"])
        selection, evidence_id = _decision(
            policy,
            scores=scores,
            margin=predictor.safety_margin,
            vlm=vlm,
        )
        utility_by_id = {
            candidate_id: float(utility)
            for candidate_id, utility in zip(
                example["candidate_ids"], example["utilities"], strict=True
            )
        }
        cost_by_id = {
            candidate_id: float(cost)
            for candidate_id, cost in zip(
                example["candidate_ids"], example["costs"], strict=True
            )
        }
        stop = float(example["stop_utility"])
        best_id = max(
            example["candidate_ids"],
            key=lambda candidate_id: (utility_by_id[candidate_id], candidate_id),
        )
        target_call = utility_by_id[best_id] > stop
        policy_utility = utility_by_id[str(evidence_id)] if selection == "ACQUIRE" else stop
        traces.append(
            {
                "example_id": example_id,
                "aoi_id": example["aoi_id"],
                "source_episode": example["source_episode"],
                "candidate_count": len(example["candidate_ids"]),
                "target_selection": "ACQUIRE" if target_call else "STOP",
                "target_evidence_id": best_id if target_call else None,
                "predicted_selection": selection,
                "predicted_evidence_id": evidence_id,
                "stop_utility": stop,
                "oracle_utility": max(stop, utility_by_id[best_id]),
                "policy_utility": policy_utility,
                "policy_cost": cost_by_id[str(evidence_id)] if selection == "ACQUIRE" else 0.0,
                "regret": max(stop, utility_by_id[best_id]) - policy_utility,
                "ranker_scores": scores,
                "ranker_margin": predictor.safety_margin,
            }
        )
    return traces


def _promotion(
    metrics: dict[str, float], bootstrap: dict[str, Any]
) -> dict[str, bool]:
    checks = {
        "nonzero_calls": metrics["predicted_call_rate"] > 0.0,
        "macro_f1_above_always_stop": metrics[
            "selection_macro_f1_delta_vs_always_stop"
        ]
        > 0.0,
        "utility_positive": metrics["realized_utility_mean"] > 0.0,
        "utility_aoi_bootstrap_ci_above_zero": bootstrap["intervals"][
            "realized_utility_mean"
        ]["ci95_low"]
        > 0.0,
        "false_call_rate_at_most_0_10": metrics["false_call_rate"] <= 0.10,
        "exact_recall_above_random": metrics["exact_evidence_recall"]
        > metrics["random_exact_evidence_recall"],
    }
    return {**checks, "passed": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("val_jsonl", type=Path)
    parser.add_argument("val_evaluation_index", type=Path)
    parser.add_argument("vlm_traces", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    examples = load_examples(args.val_jsonl, args.val_evaluation_index, "val")
    predictor = CandidateUtilityRankerPredictor(str(args.checkpoint), args.device)
    vlm_rows = _load_jsonl(args.vlm_traces)
    policies = (
        "vlm_original",
        "ranker_only",
        "vlm_gate_ranker_candidate",
        "vlm_gate_ranker_candidate_vetoed",
        "ranker_guarded_vlm_candidate",
    )
    args.output_dir.mkdir(parents=True)
    results = {}
    for policy in policies:
        traces = build_policy_traces(examples, predictor, vlm_rows, policy)
        metrics = active_catalog_metrics(traces)
        bootstrap = grouped_bootstrap(
            traces,
            group_key="aoi_id",
            repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        )
        trace_path = args.output_dir / f"{policy}_traces.jsonl"
        with trace_path.open("w", encoding="utf-8") as handle:
            for row in traces:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        results[policy] = {
            "metrics": metrics,
            "aoi_bootstrap": bootstrap,
            "promotion_gate": _promotion(metrics, bootstrap),
        }
    summary = {
        "schema_version": "active-catalog-ranker-guard-evaluation-v1",
        "ranker_checkpoint": str(args.checkpoint.resolve()),
        "ranker_safety_margin": predictor.safety_margin,
        "policies": results,
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
