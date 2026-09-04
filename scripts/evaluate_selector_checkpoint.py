#!/usr/bin/env python3
"""Evaluate a frozen selector checkpoint with AOI-cluster bootstrap intervals."""

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


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    count = len(rows)
    calls = sum(bool(row["called"]) for row in rows)
    target_acquires = sum(bool(row["target_acquire"]) for row in rows)
    return {
        "sample_count": float(count),
        "mean_utility": sum(float(row["utility"]) for row in rows) / count,
        "mean_regret": sum(float(row["regret"]) for row in rows) / count,
        "call_rate": calls / count,
        "false_call_rate": sum(bool(row["false_call"]) for row in rows) / count,
        "harmful_call_fraction": sum(bool(row["harmful_call"]) for row in rows)
        / max(calls, 1),
        "acquire_recall": sum(
            bool(row["called"]) and bool(row["target_acquire"]) for row in rows
        )
        / max(target_acquires, 1),
        "exact_acquire_recall": sum(bool(row["exact_acquire"]) for row in rows)
        / max(target_acquires, 1),
    }


def cluster_bootstrap(
    rows: list[dict[str, Any]], *, draws: int, seed: int
) -> dict[str, list[float]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["aoi_id"]), []).append(row)
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {}
    for _ in range(draws):
        sampled_names = rng.choice(names, size=len(names), replace=True)
        sampled_rows = [row for name in sampled_names for row in groups[str(name)]]
        metrics = aggregate_rows(sampled_rows)
        for name, value in metrics.items():
            if name != "sample_count":
                values.setdefault(name, []).append(value)
    return {
        name: [
            float(np.quantile(metric_values, 0.025)),
            float(np.quantile(metric_values, 0.975)),
        ]
        for name, metric_values in values.items()
    }


def normalizer_from_dict(payload: dict[str, list[float]] | None):
    if payload is None:
        return None
    return SelectorFeatureNormalizer.from_dict(payload)


@torch.no_grad()
def evaluate(
    checkpoint_path: Path,
    *,
    device: torch.device,
    batch_size: int,
    stop_margin_override: float | None = None,
) -> list[dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = EvidenceSelector(SelectorConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    samples = load_selector_samples(Path(checkpoint["data_path"]), split="val")
    ablation = AblationSpec(**checkpoint["ablation"])
    normalizer = normalizer_from_dict(checkpoint.get("feature_normalizer"))
    loader = DataLoader(
        SelectorDataset(samples, ablation, normalizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_selector_batch,
    )
    stop_margin = (
        float(stop_margin_override)
        if stop_margin_override is not None
        else float(checkpoint.get("stop_margin", 0.0))
    )
    rows: list[dict[str, Any]] = []
    offset = 0
    for batch in loader:
        evidence = batch["evidence"].to(device)
        hypothesis = batch["hypothesis"].to(device)
        state = batch["state"].to(device)
        mask = batch["mask"].to(device)
        utilities = batch["utilities"].to(device)
        logits = model(evidence, hypothesis, state, mask)
        logits[:, -1] += stop_margin
        predictions = logits.argmax(dim=-1)
        targets = utilities.argmax(dim=-1)
        best_evidence = logits[:, :-1].argmax(dim=-1)
        best_evidence_scores = (
            logits[:, :-1].gather(1, best_evidence[:, None]).squeeze(1)
        )
        acquire_utilities = (
            utilities[:, :-1].gather(1, best_evidence[:, None]).squeeze(1)
        )
        chosen = utilities.gather(1, predictions[:, None]).squeeze(1)
        oracle = utilities.max(dim=-1).values
        stop_index = logits.shape[1] - 1
        for index in range(len(batch["sample_ids"])):
            sample = samples[offset + index]
            predicted = int(predictions[index].cpu())
            target = int(targets[index].cpu())
            called = predicted != stop_index
            target_acquire = target != stop_index
            utility = float(chosen[index].cpu())
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "source_episode": str(
                        sample.metadata.get("source_episode", sample.sample_id)
                    ),
                    "aoi_id": str(sample.metadata.get("aoi_id", "unknown")),
                    "called": called,
                    "target_acquire": target_acquire,
                    "false_call": called and not target_acquire,
                    "harmful_call": called and utility < sample.stop_utility,
                    "exact_acquire": called and target_acquire and predicted == target,
                    "utility": utility,
                    "oracle_utility": float(oracle[index].cpu()),
                    "regret": float((oracle[index] - chosen[index]).cpu()),
                    "predicted_index": predicted if called else -1,
                    "target_index": target if target_acquire else -1,
                    "decision_margin": float(
                        (best_evidence_scores[index] - logits[index, -1]).cpu()
                    ),
                    "best_evidence_index": int(best_evidence[index].cpu()),
                    "acquire_utility": float(acquire_utilities[index].cpu()),
                    "stop_utility": float(utilities[index, -1].cpu()),
                }
            )
        offset += len(batch["sample_ids"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260721)
    parser.add_argument(
        "--stop-margin-override",
        type=float,
        default=None,
        help="Evaluate a frozen train-calibrated STOP margin instead of the checkpoint value.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    effective_stop_margin = (
        float(args.stop_margin_override)
        if args.stop_margin_override is not None
        else float(checkpoint.get("stop_margin", 0.0))
    )
    rows = evaluate(
        args.checkpoint,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        stop_margin_override=args.stop_margin_override,
    )
    with (args.output_dir / "per_sample.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "selector-checkpoint-evaluation-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "split": "val",
        "test_assets_read": False,
        "effective_stop_margin": effective_stop_margin,
        "stop_margin_source": (
            "override" if args.stop_margin_override is not None else "checkpoint"
        ),
        "metrics": aggregate_rows(rows),
        "aoi_count": len({str(row["aoi_id"]) for row in rows}),
        "bootstrap_draws": args.bootstrap_draws,
        "aoi_cluster_bootstrap_95_ci": cluster_bootstrap(
            rows, draws=args.bootstrap_draws, seed=args.bootstrap_seed
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
