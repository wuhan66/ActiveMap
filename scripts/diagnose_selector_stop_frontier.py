#!/usr/bin/env python3
"""Audit the train-only STOP trade-off of a frozen evidence selector.

The selector emits a candidate score and a STOP score for each state.  This
script never refits either model: it evaluates the operating frontier induced
by different STOP margins on a declared, non-test split.  It is intended to
detect a calibration infeasibility before an expensive map-writeback run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MarginObservations:
    """Per-state values needed to evaluate a STOP threshold."""

    margins: np.ndarray
    chosen_candidate_utilities: np.ndarray
    stop_utilities: np.ndarray
    oracle_utilities: np.ndarray
    target_acquire: np.ndarray
    exact_candidate: np.ndarray

    def __post_init__(self) -> None:
        size = len(self.margins)
        values = (
            self.chosen_candidate_utilities,
            self.stop_utilities,
            self.oracle_utilities,
            self.target_acquire,
            self.exact_candidate,
        )
        if size == 0 or any(len(value) != size for value in values):
            raise ValueError("STOP-frontier observations must be aligned and non-empty")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def margin_observations(
    predictor: Any,
    samples: list[Any],
    *,
    device: str,
    batch_size: int,
) -> MarginObservations:
    """Run batched raw-score inference without applying the saved STOP margin."""

    import torch
    from torch.utils.data import DataLoader

    from activemap.training.data import SelectorDataset, collate_selector_batch

    dataset = SelectorDataset(
        samples,
        predictor.ablation,
        normalizer=predictor.feature_normalizer,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_selector_batch,
    )
    target_device = torch.device(device)
    predictor.model.to(target_device).eval()
    margins: list[np.ndarray] = []
    candidate_utilities: list[np.ndarray] = []
    stop_utilities: list[np.ndarray] = []
    oracle_utilities: list[np.ndarray] = []
    target_acquire: list[np.ndarray] = []
    exact_candidate: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            evidence = batch["evidence"].to(target_device)
            hypothesis = batch["hypothesis"].to(target_device)
            state = batch["state"].to(target_device)
            mask = batch["mask"].to(target_device)
            utilities = batch["utilities"].to(target_device)
            scores = predictor.model(evidence, hypothesis, state, mask)
            candidate_scores = scores[:, :-1]
            chosen_indices = torch.argmax(candidate_scores, dim=-1)
            chosen_scores = torch.gather(
                candidate_scores, 1, chosen_indices[:, None]
            ).squeeze(1)
            chosen_utilities = torch.gather(
                utilities[:, :-1], 1, chosen_indices[:, None]
            ).squeeze(1)
            stop = utilities[:, -1]
            best_candidate_utility, target_indices = utilities[:, :-1].max(dim=-1)
            target = torch.argmax(utilities, dim=-1)
            margins.append((chosen_scores - scores[:, -1]).cpu().numpy())
            candidate_utilities.append(chosen_utilities.cpu().numpy())
            stop_utilities.append(stop.cpu().numpy())
            oracle_utilities.append(torch.maximum(best_candidate_utility, stop).cpu().numpy())
            target_acquire.append((best_candidate_utility > stop).cpu().numpy())
            exact_candidate.append((chosen_indices == target_indices).cpu().numpy())
            if torch.any(target >= utilities.shape[1]):
                raise RuntimeError("invalid selector target index")
    return MarginObservations(
        margins=np.concatenate(margins),
        chosen_candidate_utilities=np.concatenate(candidate_utilities),
        stop_utilities=np.concatenate(stop_utilities),
        oracle_utilities=np.concatenate(oracle_utilities),
        target_acquire=np.concatenate(target_acquire),
        exact_candidate=np.concatenate(exact_candidate),
    )


def point(
    observations: MarginObservations,
    *,
    stop_margin: float,
) -> dict[str, float]:
    """Evaluate a threshold where a candidate is acquired iff margin is larger."""

    acquire = observations.margins > stop_margin
    target_acquire = observations.target_acquire.astype(bool)
    chosen = np.where(
        acquire,
        observations.chosen_candidate_utilities,
        observations.stop_utilities,
    )
    acquire_count = int(acquire.sum())
    true_calls = acquire & target_acquire
    harmful_calls = acquire & (
        observations.chosen_candidate_utilities < observations.stop_utilities
    )
    return {
        "stop_margin": float(stop_margin),
        "acquire_rate": float(np.mean(acquire)),
        "false_call_rate": float(np.mean(acquire & ~target_acquire)),
        "harmful_call_fraction": float(harmful_calls.sum() / max(acquire_count, 1)),
        "acquire_recall": float(true_calls.sum() / max(int(target_acquire.sum()), 1)),
        "exact_acquire_rate": float(
            np.mean(acquire & target_acquire & observations.exact_candidate)
        ),
        "mean_chosen_utility": float(np.mean(chosen)),
        "mean_oracle_utility": float(np.mean(observations.oracle_utilities)),
        "mean_regret": float(np.mean(observations.oracle_utilities - chosen)),
    }


def is_feasible(
    value: dict[str, float],
    *,
    max_false_call_rate: float | None,
    max_harmful_call_fraction: float | None,
    min_acquire_recall: float,
) -> bool:
    return (
        (max_false_call_rate is None or value["false_call_rate"] <= max_false_call_rate)
        and (
            max_harmful_call_fraction is None
            or value["harmful_call_fraction"] <= max_harmful_call_fraction
        )
        and value["acquire_recall"] >= min_acquire_recall
    )


def constraint_distance(
    value: dict[str, float],
    *,
    max_false_call_rate: float | None,
    max_harmful_call_fraction: float | None,
    min_acquire_recall: float,
) -> tuple[float, float, float, float]:
    """Use the historical calibrator's safety-first fallback ordering."""

    false_violation = (
        max(value["false_call_rate"] - max_false_call_rate, 0.0)
        if max_false_call_rate is not None
        else 0.0
    )
    harmful_violation = (
        max(value["harmful_call_fraction"] - max_harmful_call_fraction, 0.0)
        if max_harmful_call_fraction is not None
        else 0.0
    )
    recall_violation = max(min_acquire_recall - value["acquire_recall"], 0.0)
    return (
        false_violation + harmful_violation,
        recall_violation,
        -value["mean_chosen_utility"],
        value["acquire_rate"],
    )


def summarize_frontier(
    observations: MarginObservations,
    *,
    current_stop_margin: float,
    points: int,
    max_false_call_rate: float | None,
    max_harmful_call_fraction: float | None,
    min_acquire_recall: float,
) -> dict[str, Any]:
    if points < 3:
        raise ValueError("points must be at least 3")
    margins = observations.margins.astype(np.float64)
    threshold_grid = np.quantile(margins, np.linspace(0.0, 1.0, points))
    thresholds = np.unique(
        np.concatenate(
            [
                threshold_grid,
                np.asarray(
                    [
                        current_stop_margin,
                        0.0,
                        float(np.nextafter(margins.min(), -np.inf)),
                        float(np.nextafter(margins.max(), np.inf)),
                    ]
                ),
            ]
        )
    )
    frontier = [point(observations, stop_margin=float(value)) for value in thresholds]
    feasible = [
        value
        for value in frontier
        if is_feasible(
            value,
            max_false_call_rate=max_false_call_rate,
            max_harmful_call_fraction=max_harmful_call_fraction,
            min_acquire_recall=min_acquire_recall,
        )
    ]
    selected_feasible = (
        max(
            feasible,
            key=lambda value: (
                value["mean_chosen_utility"],
                -value["acquire_rate"],
            ),
        )
        if feasible
        else None
    )
    closest_infeasible = min(
        frontier,
        key=lambda value: constraint_distance(
            value,
            max_false_call_rate=max_false_call_rate,
            max_harmful_call_fraction=max_harmful_call_fraction,
            min_acquire_recall=min_acquire_recall,
        ),
    )
    return {
        "current_checkpoint": point(observations, stop_margin=current_stop_margin),
        "zero_margin": point(observations, stop_margin=0.0),
        "constraints": {
            "max_false_call_rate": max_false_call_rate,
            "max_harmful_call_fraction": max_harmful_call_fraction,
            "min_acquire_recall": min_acquire_recall,
        },
        "feasible_point_count": len(feasible),
        "selected_feasible": selected_feasible,
        "closest_infeasible": closest_infeasible,
        "frontier": frontier,
    }


def main() -> None:
    from activemap.inference import SelectorPredictor
    from activemap.training.data import load_selector_samples
    from activemap.training.selector import split_fit_calibration_samples

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.0)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--group-key", default="source_episode")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--points", type=int, default=101)
    parser.add_argument("--max-false-call-rate", type=float)
    parser.add_argument("--max-harmful-call-fraction", type=float)
    parser.add_argument("--min-acquire-recall", type=float, default=0.0)
    parser.add_argument(
        "--candidate-decision-mode", choices=("rank", "value", "hybrid")
    )
    parser.add_argument(
        "--terminal-gate-mode", choices=("context", "value", "hybrid")
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frontier report: {args.output}")
    if not 0.0 <= args.calibration_fraction < 1.0:
        raise ValueError("calibration fraction must be in [0, 1)")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    for name, value in (
        ("max false call rate", args.max_false_call_rate),
        ("max harmful call fraction", args.max_harmful_call_fraction),
        ("min acquire recall", args.min_acquire_recall),
    ):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    samples = load_selector_samples(args.samples, split=args.split)
    partition = "declared_split"
    if args.calibration_fraction > 0.0:
        _, samples = split_fit_calibration_samples(
            samples,
            fraction=args.calibration_fraction,
            seed=args.calibration_seed,
            group_key=args.group_key,
        )
        partition = "train_only_grouped_calibration_holdout"
    if any(
        sample.split == "test" or sample.metadata.get("test_assets_read") is True
        for sample in samples
    ):
        raise ValueError("STOP-frontier audit rejects test provenance")
    predictor = SelectorPredictor(args.checkpoint, device=args.device, stop_margin_override=0.0)
    if args.candidate_decision_mode or args.terminal_gate_mode:
        predictor.model.config = replace(
            predictor.model.config,
            candidate_decision_mode=(
                args.candidate_decision_mode
                or predictor.model.config.candidate_decision_mode
            ),
            terminal_gate_mode=(
                args.terminal_gate_mode or predictor.model.config.terminal_gate_mode
            ),
        )
    observations = margin_observations(
        predictor,
        samples,
        device=args.device,
        batch_size=args.batch_size,
    )
    result = {
        "schema_version": "selector-stop-frontier-audit-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "samples": str(args.samples.resolve()),
        "samples_sha256": sha256(args.samples),
        "split": args.split,
        "partition": partition,
        "sample_count": len(samples),
        "group_key": args.group_key,
        "calibration_fraction": args.calibration_fraction,
        "calibration_seed": args.calibration_seed,
        "checkpoint_stop_margin": predictor.checkpoint_stop_margin,
        "decision_override": {
            "candidate_decision_mode": predictor.model.config.candidate_decision_mode,
            "terminal_gate_mode": predictor.model.config.terminal_gate_mode,
        },
        "test_assets_read": False,
        **summarize_frontier(
            observations,
            current_stop_margin=predictor.checkpoint_stop_margin,
            points=args.points,
            max_false_call_rate=args.max_false_call_rate,
            max_harmful_call_fraction=args.max_harmful_call_fraction,
            min_acquire_recall=args.min_acquire_recall,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
