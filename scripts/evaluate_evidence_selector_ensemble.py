#!/usr/bin/env python3
"""Validation-only evaluation for a calibrated evidence-selector ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from activemap.evaluation.selector import (
    evaluate_score_policy,
    initial_states_for_budget,
    metrics_by_edit_type,
)
from activemap.inference import SelectorEnsemblePredictor
from activemap.policy.baselines import BaselineName, baseline_scores
from activemap.policy.rollout import evaluate_rollouts
from activemap.training.data import load_selector_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--budgets", default="1.5,3,4.5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rollout-output", type=Path)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    samples = load_selector_samples(args.states, split=args.split)
    budgets = [float(value) for value in args.budgets.split(",")]
    predictor = SelectorEnsemblePredictor(args.checkpoint, device=args.device)
    rows = []
    details = {}
    for budget in budgets:
        budget_samples = initial_states_for_budget(samples, budget)
        metric = evaluate_score_policy(
            budget_samples,
            method="learned_ensemble",
            budget=budget,
            score_fn=predictor.action_scores,
        )
        rows.append(metric.as_dict())
        details[f"learned_ensemble@{budget:g}"] = {
            key: value.as_dict()
            for key, value in metrics_by_edit_type(
                budget_samples,
                method="learned_ensemble",
                budget=budget,
                score_fn=predictor.action_scores,
            ).items()
        }
    payload = {
        "protocol": {
            "split": args.split,
            "budgets": budgets,
            "checkpoints": [str(path) for path in args.checkpoint],
            "member_count": len(args.checkpoint),
            "test_evaluation": False,
        },
        "overall": rows,
        "by_edit_type": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    if args.rollout_output is not None:
        methods = {
            baseline.value: (
                lambda sample, baseline=baseline: baseline_scores(
                    sample, baseline, seed=args.seed
                )
            )
            for baseline in BaselineName
        }
        methods["learned_ensemble"] = predictor.action_scores
        rollout_rows = []
        args.rollout_output.mkdir(parents=True, exist_ok=True)
        with (args.rollout_output / "traces.jsonl").open("w", encoding="utf-8") as handle:
            for method, score_fn in methods.items():
                for budget in budgets:
                    budget_samples = initial_states_for_budget(samples, budget)
                    summary, traces = evaluate_rollouts(
                        budget_samples,
                        method=method,
                        budget=budget,
                        score_fn=score_fn,
                    )
                    rollout_rows.append(summary)
                    for trace in traces:
                        handle.write(
                            json.dumps(
                                {"method": method, "budget": budget, **trace.as_dict()}
                            )
                            + "\n"
                        )
        pd.DataFrame(rollout_rows).to_csv(
            args.rollout_output / "summary.csv", index=False
        )
        (args.rollout_output / "summary.json").write_text(
            json.dumps(
                {
                    "protocol": payload["protocol"],
                    "dynamic_utility_accounting": True,
                    "results": rollout_rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload["overall"], indent=2))


if __name__ == "__main__":
    main()
