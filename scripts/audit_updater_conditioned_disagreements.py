#!/usr/bin/env python3
"""Audit stale/refreshed selector disagreements on one immutable state cache.

This is the data-readiness gate for the C5 residual-correction branch.  It
compares two selector checkpoints on the same updater-f1 states and records
only decisions that differ.  The cache supplies executable outcomes for both
choices, but no test state is accepted.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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


def _choice_records(
    checkpoint_path: Path,
    samples_path: Path,
    *,
    device: torch.device,
    batch_size: int,
    split: str,
) -> tuple[list[Any], dict[str, dict[str, Any]]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = EvidenceSelector(SelectorConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    samples = load_selector_samples(samples_path, split=split)
    if any(sample.split == "test" for sample in samples):
        raise ValueError("C5 disagreement audit forbids test records")
    dataset = SelectorDataset(
        samples,
        AblationSpec(**checkpoint["ablation"]),
        _normalizer(checkpoint.get("feature_normalizer")),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_selector_batch)
    choices: dict[str, dict[str, Any]] = {}
    offset = 0
    stop_margin = float(checkpoint.get("stop_margin", 0.0))
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["evidence"].to(device),
                batch["hypothesis"].to(device),
                batch["state"].to(device),
                batch["mask"].to(device),
            )
            logits[:, -1] += stop_margin
            stop_index = logits.shape[1] - 1
            for local_index in range(logits.shape[0]):
                sample = samples[offset + local_index]
                candidate_logits = logits[local_index, : len(sample.evidence_ids)].detach().cpu().numpy()
                stop_logit = float(logits[local_index, stop_index].detach().cpu())
                index = int(np.argmax(np.concatenate([candidate_logits, [stop_logit]])))
                is_stop = index == len(candidate_logits)
                evidence_id = str(sample.metadata["initial_evidence_id"]) if is_stop else str(sample.evidence_ids[index])
                choices[str(sample.sample_id)] = {
                    "index": None if is_stop else index,
                    "evidence_id": evidence_id,
                    "score": stop_logit if is_stop else float(candidate_logits[index]),
                    "stop": is_stop,
                }
            offset += logits.shape[0]
    return samples, choices


def _outcome(sample: Any, evidence_id: str) -> dict[str, Any]:
    outcome = sample.metadata.get("executable_outcomes", {}).get(evidence_id)
    if not isinstance(outcome, dict):
        raise ValueError(f"{sample.sample_id}: missing outcome for {evidence_id}")
    return outcome


def _row(sample: Any, stale: dict[str, Any], refreshed: dict[str, Any]) -> dict[str, Any]:
    stale_outcome = _outcome(sample, stale["evidence_id"])
    refreshed_outcome = _outcome(sample, refreshed["evidence_id"])
    utility = lambda choice: float(sample.stop_utility) if choice["stop"] else float(sample.oracle_utilities[choice["index"]])
    stale_utility = utility(stale)
    refreshed_utility = utility(refreshed)
    return {
        "sample_id": str(sample.sample_id),
        "source_episode": str(sample.metadata.get("source_episode", sample.sample_id)),
        "aoi_id": str(sample.metadata.get("aoi_id", "unknown")),
        "edit_type": str(sample.edit_type),
        "stale_choice": stale,
        "refreshed_choice": refreshed,
        "stale_utility": stale_utility,
        "refreshed_utility": refreshed_utility,
        "utility_delta": refreshed_utility - stale_utility,
        "stale_raster_iou": float(stale_outcome["final_raster_iou"]),
        "refreshed_raster_iou": float(refreshed_outcome["final_raster_iou"]),
        "raster_iou_delta": float(refreshed_outcome["final_raster_iou"]) - float(stale_outcome["final_raster_iou"]),
        "stale_false_edit": bool(stale_outcome["false_edit"]),
        "refreshed_false_edit": bool(refreshed_outcome["false_edit"]),
        "stale_missed_edit": bool(stale_outcome["missed_edit"]),
        "refreshed_missed_edit": bool(refreshed_outcome["missed_edit"]),
        "stale_risk": None if stale["stop"] else float(sample.false_edit_risks[stale["index"]]),
        "refreshed_risk": None if refreshed["stop"] else float(sample.false_edit_risks[refreshed["index"]]),
    }


def _summary(rows: list[dict[str, Any]], sample_count: int) -> dict[str, Any]:
    by_edit: dict[str, int] = Counter(row["edit_type"] for row in rows)
    by_pair: dict[str, int] = Counter(
        f"{'STOP' if row['stale_choice']['stop'] else 'ACQUIRE'}->{('STOP' if row['refreshed_choice']['stop'] else 'ACQUIRE')}"
        for row in rows
    )
    rates = {
        "utility_improved": np.mean([row["utility_delta"] > 0.0 for row in rows]) if rows else 0.0,
        "iou_improved": np.mean([row["raster_iou_delta"] > 0.0 for row in rows]) if rows else 0.0,
        "false_edit_regressed": np.mean([
            (not row["stale_false_edit"]) and row["refreshed_false_edit"] for row in rows
        ]) if rows else 0.0,
        "missed_edit_improved": np.mean([
            row["stale_missed_edit"] and (not row["refreshed_missed_edit"]) for row in rows
        ]) if rows else 0.0,
    }
    return {
        "schema_version": "updater-conditioned-disagreement-audit-v1",
        "sample_count": sample_count,
        "disagreement_count": len(rows),
        "disagreement_rate": len(rows) / max(sample_count, 1),
        "by_edit_type": dict(sorted(by_edit.items())),
        "by_action_pair": dict(sorted(by_pair.items())),
        "rates_within_disagreements": {key: float(value) for key, value in rates.items()},
        "test_assets_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stale_checkpoint", type=Path)
    parser.add_argument("refreshed_checkpoint", type=Path)
    parser.add_argument("states", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    device = torch.device(args.device)
    samples, stale_choices = _choice_records(
        args.stale_checkpoint, args.states, device=device, batch_size=args.batch_size, split=args.split
    )
    refreshed_samples, refreshed_choices = _choice_records(
        args.refreshed_checkpoint, args.states, device=device, batch_size=args.batch_size, split=args.split
    )
    if [sample.sample_id for sample in samples] != [sample.sample_id for sample in refreshed_samples]:
        raise ValueError("stale and refreshed sample ordering differs")
    rows = [
        _row(sample, stale_choices[str(sample.sample_id)], refreshed_choices[str(sample.sample_id)])
        for sample in samples
        if stale_choices[str(sample.sample_id)]["evidence_id"] != refreshed_choices[str(sample.sample_id)]["evidence_id"]
    ]
    summary = _summary(rows, len(samples))
    summary.update({
        "split": args.split,
        "stale_checkpoint": str(args.stale_checkpoint.resolve()),
        "refreshed_checkpoint": str(args.refreshed_checkpoint.resolve()),
        "states": str(args.states.resolve()),
    })
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "per_disagreement.jsonl").open("x", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, separators=(",", ":")) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
