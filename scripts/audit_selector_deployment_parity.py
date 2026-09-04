#!/usr/bin/env python3
"""Audit frozen selector decisions before and after environment reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from activemap.inference import SelectorPredictor
from activemap.training.data import load_selector_samples
from scripts.evaluate_active_catalog_closed_loop_baselines import (
    deterministic_sample,
    make_environment,
)


def max_delta(left: list[float], right: list[float]) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.float64)
                - np.asarray(right, dtype=np.float64)
            )
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--sample-seed", type=int, default=20260729)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = [
        sample
        for sample in load_selector_samples(args.states)
        if sample.split == "val"
    ]
    samples = deterministic_sample(samples, args.limit, args.sample_seed)
    predictor = SelectorPredictor(args.checkpoint, device=args.device)
    rows = []
    hypothesis_deltas = []
    state_deltas = []
    for sample in samples:
        environment = make_environment(
            sample,
            max_candidates=16,
            score_fn=predictor.action_scores,
        )
        reconstructed = environment.current_sample()
        direct_scores = predictor.action_scores(sample)
        deployed_scores = predictor.action_scores(reconstructed)
        direct_action = int(np.argmax(direct_scores))
        deployed_action = int(np.argmax(deployed_scores))
        hypothesis_delta = np.abs(
            np.asarray(sample.hypothesis_features, dtype=np.float64)
            - np.asarray(reconstructed.hypothesis_features, dtype=np.float64)
        )
        state_delta = np.abs(
            np.asarray(sample.state_features, dtype=np.float64)
            - np.asarray(reconstructed.state_features, dtype=np.float64)
        )
        hypothesis_deltas.append(hypothesis_delta)
        state_deltas.append(state_delta)
        direct_stop = direct_action == len(sample.evidence_ids)
        deployed_stop = deployed_action == len(reconstructed.evidence_ids)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "source_episode": str(sample.metadata["source_episode"]),
                "budget": float(sample.metadata["budget"]),
                "hypothesis_max_abs_delta": max_delta(
                    sample.hypothesis_features,
                    reconstructed.hypothesis_features,
                ),
                "state_max_abs_delta": max_delta(
                    sample.state_features,
                    reconstructed.state_features,
                ),
                "direct_state_features": sample.state_features,
                "deployed_state_features": reconstructed.state_features,
                "score_max_abs_delta": max_delta(
                    direct_scores.tolist(),
                    deployed_scores.tolist(),
                ),
                "direct_stop": direct_stop,
                "deployed_stop": deployed_stop,
                "action_match": direct_action == deployed_action,
                "direct_margin": float(
                    np.max(direct_scores[:-1]) - direct_scores[-1]
                ),
                "deployed_margin": float(
                    np.max(deployed_scores[:-1]) - deployed_scores[-1]
                ),
            }
        )
    count = len(rows)
    summary = {
        "schema_version": "selector-deployment-parity-v1",
        "sample_count": count,
        "direct_call_rate": sum(not row["direct_stop"] for row in rows) / count,
        "deployed_call_rate": sum(not row["deployed_stop"] for row in rows) / count,
        "action_match_rate": sum(row["action_match"] for row in rows) / count,
        "max_hypothesis_abs_delta": max(
            row["hypothesis_max_abs_delta"] for row in rows
        ),
        "max_state_abs_delta": max(row["state_max_abs_delta"] for row in rows),
        "max_score_abs_delta": max(row["score_max_abs_delta"] for row in rows),
        "hypothesis_max_abs_delta_by_dimension": np.max(
            np.stack(hypothesis_deltas), axis=0
        ).tolist(),
        "state_max_abs_delta_by_dimension": np.max(
            np.stack(state_deltas), axis=0
        ).tolist(),
        "mismatches": [row for row in rows if not row["action_match"]],
        "test_assets_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    compact = {
        key: value for key, value in summary.items() if key != "mismatches"
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
