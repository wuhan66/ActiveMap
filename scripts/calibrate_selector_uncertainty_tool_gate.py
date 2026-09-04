#!/usr/bin/env python3
"""Calibrate a train-only post-acquisition uncertainty gate for a selector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from activemap.agent.tools import CounterfactualBeliefUpdater
from activemap.inference import SelectorPredictor
from activemap.training.data import (
    SelectorDataset,
    collate_selector_batch,
    load_selector_samples,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target-call-rate", type=float, default=0.15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not 0.0 < args.target_call_rate < 1.0:
        raise ValueError("target-call-rate must be in (0, 1)")
    predictor = SelectorPredictor(args.checkpoint, device=args.device)
    samples = load_selector_samples(args.states, split="train")
    loader = DataLoader(
        SelectorDataset(
            samples,
            predictor.ablation,
            predictor.feature_normalizer,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_selector_batch,
    )
    uncertainties: list[float] = []
    task_ids = set()
    offset = 0
    for batch in loader:
        evidence = batch["evidence"].to(predictor.device)
        hypothesis = batch["hypothesis"].to(predictor.device)
        state = batch["state"].to(predictor.device)
        mask = batch["mask"].to(predictor.device)
        with torch.no_grad():
            logits = predictor.model(evidence, hypothesis, state, mask)
            logits[:, -1] += predictor.stop_margin
            predictions = logits.argmax(dim=-1).cpu().tolist()
        for index, prediction in enumerate(predictions):
            sample = samples[offset + index]
            if prediction == logits.shape[1] - 1:
                continue
            selected = list(sample.metadata.get("selected_evidence_ids", []))
            selected.append(sample.evidence_ids[int(prediction)])
            belief = CounterfactualBeliefUpdater(sample).fuse(selected)
            uncertainties.append(float(belief.uncertainty))
            task_ids.add(str(sample.metadata["source_episode"]))
        offset += len(predictions)
    for sample in samples:
        if bool(sample.metadata.get("test_assets_read", False)):
            raise ValueError(f"test-marked state: {sample.sample_id}")
    state_count = len(samples)
    if len(uncertainties) < 2:
        raise ValueError("selector produced fewer than two train acquisitions")
    values = np.asarray(uncertainties, dtype=np.float64)
    threshold = float(
        np.quantile(values, 1.0 - args.target_call_rate, method="higher")
    )
    observed_call_rate = float(np.mean(values >= threshold))
    summary = {
        "schema_version": "uncertainty-tool-gate-summary-v1",
        "split": "train",
        "input_state_count": state_count,
        "eligible_state_count": len(uncertainties),
        "task_count": len(task_ids),
        "selector_acquire_rate": len(uncertainties) / state_count,
        "target_call_rate": args.target_call_rate,
        "observed_train_call_rate": observed_call_rate,
        "selected": {
            "feature": "belief_uncertainty",
            "threshold": threshold,
            "direction": "greater_or_equal",
        },
        "outcome_labels_used": False,
        "validation_metrics_used": False,
        "test_assets_read": False,
        "sources": {
            "states": str(args.states.resolve()),
            "states_sha256": sha256(args.states),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
        },
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "gate.json").write_text(
        json.dumps(
            {
                "schema_version": "belief-uncertainty-gate-v1",
                "threshold": threshold,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
