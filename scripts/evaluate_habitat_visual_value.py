#!/usr/bin/env python3
"""Evaluate a trained causal RGB-D value policy against map-only controls."""

from __future__ import annotations

import argparse
from pathlib import Path

from activemap.evaluation.navigation_rollout import (
    COMMIT_RULES,
    POLICIES,
    load_navigation_episodes,
    rollout_navigation_episode,
    summarize_navigation_results,
    write_navigation_rollout_results,
)
from activemap.nn.navigation_visual_value import (
    NavigationPriorAnchoredVisualValuePolicy,
    NavigationVisualValuePolicy,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--commit-rule", choices=tuple(COMMIT_RULES), default="always")
    parser.add_argument(
        "--prior-override-margin",
        action="append",
        type=float,
        default=[],
        help="add an unknown-coverage-anchored visual policy at this margin",
    )
    parser.add_argument(
        "--controls",
        default="stop,nearest,frontier_nearest,unknown_coverage,unknown_per_cost",
    )
    args = parser.parse_args()
    if args.split == "test":
        raise PermissionError("test split is locked for visual-value evaluation")
    controls = tuple(name.strip() for name in args.controls.split(",") if name.strip())
    invalid = sorted(set(controls) - set(POLICIES))
    if invalid:
        raise ValueError(f"unknown controls: {invalid}")
    if any(margin < 0.0 for margin in args.prior_override_margin):
        raise ValueError("prior override margins must be non-negative")
    if len(set(args.prior_override_margin)) != len(args.prior_override_margin):
        raise ValueError("prior override margins must be unique")
    episodes = load_navigation_episodes(args.index, split=args.split)
    predictor = NavigationVisualValuePolicy(args.checkpoint, device=args.device)
    results = []
    policies = [(name, POLICIES[name]) for name in controls]
    policies.append(("visual_value", predictor))
    for margin in args.prior_override_margin:
        label = f"prior_visual_m{margin:g}".replace(".", "p")
        policies.append(
            (
                label,
                NavigationPriorAnchoredVisualValuePolicy(
                    args.checkpoint, device=args.device, override_margin=margin
                ),
            )
        )
    for policy_name, policy in policies:
        for episode in episodes:
            results.append(
                rollout_navigation_episode(
                    episode,
                    policy=policy,
                    policy_name=policy_name,
                    commit_rule=COMMIT_RULES[args.commit_rule],
                    max_steps=args.max_steps,
                    budget_override=args.budget,
                )
            )
    write_navigation_rollout_results(
        args.output,
        summaries=summarize_navigation_results(results),
        results=results,
        index_path=args.index,
        split=args.split,
        max_steps=args.max_steps,
        budget_override=args.budget,
        commit_rule_name=args.commit_rule,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
