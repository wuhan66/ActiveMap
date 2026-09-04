#!/usr/bin/env python3
"""Evaluate a C5 evidence selector against immutable executable outcomes.

The usual selector evaluator reports acquisition utility and recall.  This
runner additionally resolves each selected evidence item (or the frozen
initial anchor for STOP) to its cached executable map outcome.  It is limited
to train/validation state files so it cannot accidentally open test assets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from activemap.features import AblationSpec
from activemap.nn.selector import EvidenceSelector, SelectorConfig
from activemap.training.data import (
    SelectorDataset,
    SelectorFeatureNormalizer,
    collate_selector_batch,
    load_selector_samples,
)


def _normalizer(payload: dict[str, list[float]] | None) -> SelectorFeatureNormalizer | None:
    return SelectorFeatureNormalizer.from_dict(payload) if payload is not None else None


def _direct_evidence_id(sample: Any) -> str:
    evidence_id = sample.metadata.get("initial_evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError(f"{sample.sample_id}: missing frozen initial evidence anchor")
    return evidence_id


def _outcome(sample: Any, evidence_id: str) -> dict[str, Any]:
    outcomes = sample.metadata.get("executable_outcomes")
    if not isinstance(outcomes, dict) or evidence_id not in outcomes:
        raise ValueError(f"{sample.sample_id}: missing executable outcome for {evidence_id}")
    outcome = outcomes[evidence_id]
    required = {
        "terminal_score_before_cost",
        "final_raster_iou",
        "quality_gain",
        "false_edit",
        "missed_edit",
    }
    missing = required - outcome.keys()
    if missing:
        raise ValueError(f"{sample.sample_id}: incomplete executable outcome: {sorted(missing)}")
    return outcome


def _is_stop_prediction(*, sample_id: str, candidate_count: int, prediction: int, stop_index: int) -> bool:
    if prediction == stop_index:
        return True
    if prediction < 0 or prediction >= candidate_count:
        raise ValueError(
            f"{sample_id}: model selected padded candidate index {prediction}"
        )
    return False


def _risk_guarded_choice(
    candidate_logits: np.ndarray,
    stop_logit: float,
    false_edit_risks: list[float],
    max_candidate_risk: float | None,
) -> tuple[int | None, bool]:
    """Return a local candidate index or ``None`` for STOP under a risk cap.

    ``false_edit_risks`` are updater-probability-derived deployment features,
    not executable ground-truth outcomes.  The guard only removes risky
    candidates; STOP remains available even when every candidate is filtered.
    """

    logits = np.asarray(candidate_logits, dtype=np.float64)
    risks = np.asarray(false_edit_risks, dtype=np.float64)
    if logits.ndim != 1 or risks.ndim != 1 or logits.shape != risks.shape:
        raise ValueError("candidate logits and false-edit risks must be aligned vectors")
    if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(risks)):
        raise ValueError("candidate logits and false-edit risks must be finite")
    if np.any((risks < 0.0) | (risks > 1.0)):
        raise ValueError("false-edit risks must be in [0, 1]")
    if max_candidate_risk is not None and not 0.0 <= max_candidate_risk <= 1.0:
        raise ValueError("max candidate risk must be in [0, 1]")

    unguarded = int(np.argmax(np.concatenate([logits, [float(stop_logit)]])))
    if max_candidate_risk is None:
        return (None if unguarded == len(logits) else unguarded), False
    guarded_logits = logits.copy()
    guarded_logits[risks > max_candidate_risk] = -np.inf
    guarded = int(np.argmax(np.concatenate([guarded_logits, [float(stop_logit)]])))
    return (None if guarded == len(logits) else guarded), guarded != unguarded


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("empty evaluation rows")
    count = len(rows)
    mean = lambda key: float(np.mean([float(row[key]) for row in rows]))
    return {
        "sample_count": float(count),
        "quality_cost_utility": mean("quality_cost_utility"),
        "terminal_score_before_cost": mean("terminal_score_before_cost"),
        "final_raster_iou": mean("final_raster_iou"),
        "quality_gain": mean("quality_gain"),
        "false_edit_rate": mean("false_edit"),
        "missed_edit_rate": mean("missed_edit"),
        "mean_additional_cost": mean("additional_cost"),
        "call_rate": mean("called"),
        "harmful_call_fraction": float(
            sum(bool(row["harmful_call"]) for row in rows)
            / max(sum(bool(row["called"]) for row in rows), 1)
        ),
    }


def _cluster_bootstrap(rows: list[dict[str, Any]], *, draws: int, seed: int) -> dict[str, list[float]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["aoi_id"]), []).append(row)
    names = sorted(groups)
    if not names:
        raise ValueError("no AOI groups for bootstrap")
    rng = np.random.default_rng(seed)
    series: dict[str, list[float]] = {}
    for _ in range(draws):
        sample = [row for name in rng.choice(names, size=len(names), replace=True) for row in groups[str(name)]]
        for key, value in _aggregate(sample).items():
            if key != "sample_count":
                series.setdefault(key, []).append(value)
    return {
        key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for key, values in series.items()
    }


@torch.no_grad()
def evaluate(
    checkpoint_path: Path,
    samples_path: Path,
    *,
    device: torch.device,
    batch_size: int,
    split: str = "val",
    max_candidate_risk: float | None = None,
) -> list[dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = EvidenceSelector(SelectorConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    samples = load_selector_samples(samples_path, split=split)
    if any(sample.split == "test" for sample in samples):
        raise ValueError("C5 evaluation forbids test records")
    dataset = SelectorDataset(
        samples,
        AblationSpec(**checkpoint["ablation"]),
        _normalizer(checkpoint.get("feature_normalizer")),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_selector_batch)
    rows: list[dict[str, Any]] = []
    offset = 0
    stop_margin = float(checkpoint.get("stop_margin", 0.0))
    for batch in loader:
        evidence = batch["evidence"].to(device)
        hypothesis = batch["hypothesis"].to(device)
        state = batch["state"].to(device)
        mask = batch["mask"].to(device)
        logits = model(evidence, hypothesis, state, mask)
        logits[:, -1] += stop_margin
        predicted = logits.argmax(dim=-1).cpu().tolist()
        # ``collate_selector_batch`` places STOP after the batch-wide padded
        # candidate width. It is not located after each sample's local catalog.
        stop_index = logits.shape[1] - 1
        for local_index, prediction in enumerate(predicted):
            sample = samples[offset + local_index]
            candidate_count = len(sample.evidence_ids)
            guarded_local_index, risk_guard_triggered = _risk_guarded_choice(
                logits[local_index, :candidate_count].detach().cpu().numpy(),
                float(logits[local_index, stop_index].detach().cpu()),
                sample.false_edit_risks,
                max_candidate_risk,
            )
            raw_prediction = prediction
            prediction = stop_index if guarded_local_index is None else guarded_local_index
            called = not _is_stop_prediction(
                sample_id=sample.sample_id,
                candidate_count=candidate_count,
                prediction=prediction,
                stop_index=stop_index,
            )
            if called:
                evidence_id = str(sample.evidence_ids[prediction])
                additional_cost = float(sample.evidence_costs[prediction])
                utility = float(sample.oracle_utilities[prediction])
            else:
                evidence_id = _direct_evidence_id(sample)
                additional_cost = 0.0
                utility = float(sample.stop_utility)
            outcome = _outcome(sample, evidence_id)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "source_episode": str(sample.metadata.get("source_episode", sample.sample_id)),
                    "aoi_id": str(sample.metadata.get("aoi_id", "unknown")),
                    "called": bool(called),
                    "risk_guard_triggered": bool(risk_guard_triggered),
                    "risk_cap": max_candidate_risk,
                    "raw_selected_index": int(raw_prediction),
                    "selected_evidence_id": evidence_id,
                    "quality_cost_utility": utility,
                    "terminal_score_before_cost": float(outcome["terminal_score_before_cost"]),
                    "final_raster_iou": float(outcome["final_raster_iou"]),
                    "quality_gain": float(outcome["quality_gain"]),
                    "false_edit": bool(outcome["false_edit"]),
                    "missed_edit": bool(outcome["missed_edit"]),
                    "additional_cost": additional_cost,
                    "harmful_call": bool(called and utility < float(sample.stop_utility)),
                }
            )
        offset += len(predicted)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--max-candidate-risk", type=float, default=None)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.max_candidate_risk is not None and not 0.0 <= args.max_candidate_risk <= 1.0:
        raise ValueError("--max-candidate-risk must be in [0, 1]")
    rows = evaluate(
        args.checkpoint,
        args.samples,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        split=args.split,
        max_candidate_risk=args.max_candidate_risk,
    )
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "per_sample.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "updater-conditioned-selector-executable-evaluation-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "samples": str(args.samples.resolve()),
        "split": args.split,
        "test_assets_read": False,
        "max_candidate_risk": args.max_candidate_risk,
        "metrics": _aggregate(rows),
        "aoi_count": len({str(row["aoi_id"]) for row in rows}),
        "bootstrap_draws": args.bootstrap_draws,
        "aoi_cluster_bootstrap_95_ci": _cluster_bootstrap(rows, draws=args.bootstrap_draws, seed=args.bootstrap_seed),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
