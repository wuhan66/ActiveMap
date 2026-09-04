#!/usr/bin/env python3
"""Sweep train-only STOP constraints and report the validation safety frontier."""

from __future__ import annotations

import argparse
import hashlib
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
from activemap.training.selector import (
    select_stop_margin,
    split_fit_calibration_samples,
)
from scripts.evaluate_selector_checkpoint import aggregate_rows, evaluate


def _candidate(value: str) -> dict[str, float | str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "candidate must use LABEL:MAX_HARMFUL_FRACTION:MIN_ACQUIRE_RECALL"
        )
    label, harmful, recall = parts
    if not label or not label.replace("_", "").isalnum():
        raise argparse.ArgumentTypeError("candidate label must be filesystem-safe")
    harmful_value, recall_value = float(harmful), float(recall)
    if not 0.0 <= harmful_value <= 1.0 or not 0.0 <= recall_value <= 1.0:
        raise argparse.ArgumentTypeError("candidate constraints must be in [0, 1]")
    return {
        "label": label,
        "max_harmful_call_fraction": harmful_value,
        "min_acquire_recall": recall_value,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalizer(payload: dict[str, list[float]] | None):
    return SelectorFeatureNormalizer.from_dict(payload) if payload else None


@torch.no_grad()
def calibration_inputs(
    checkpoint_path: Path,
    *,
    device: torch.device,
    batch_size: int,
    calibration_fraction: float,
    calibration_group_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = EvidenceSelector(SelectorConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    samples = load_selector_samples(Path(checkpoint["data_path"]), split="train")
    _, calibration_samples = split_fit_calibration_samples(
        samples,
        fraction=calibration_fraction,
        seed=int(checkpoint["seed"]),
        group_key=calibration_group_key,
    )
    loader = DataLoader(
        SelectorDataset(
            calibration_samples,
            AblationSpec(**checkpoint["ablation"]),
            _normalizer(checkpoint.get("feature_normalizer")),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_selector_batch,
    )
    margins: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    stops: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in loader:
        evidence = batch["evidence"].to(device)
        hypothesis = batch["hypothesis"].to(device)
        state = batch["state"].to(device)
        mask = batch["mask"].to(device)
        utilities = batch["utilities"].to(device)
        logits = model(evidence, hypothesis, state, mask)
        evidence_logits = logits[:, :-1]
        best = evidence_logits.argmax(dim=-1)
        best_scores = evidence_logits.gather(1, best[:, None]).squeeze(1)
        chosen = utilities[:, :-1].gather(1, best[:, None]).squeeze(1)
        stop = utilities[:, -1]
        margins.append((best_scores - logits[:, -1]).cpu().numpy())
        deltas.append((chosen - stop).cpu().numpy())
        stops.append(stop.cpu().numpy())
        targets.append((utilities[:, :-1].max(dim=-1).values > stop).cpu().numpy())
    return tuple(np.concatenate(values) for values in (margins, deltas, stops, targets))


def apply_margin(rows: list[dict[str, Any]], margin: float) -> list[dict[str, Any]]:
    evaluated = []
    for source in rows:
        called = float(source["decision_margin"]) > margin
        utility = float(source["acquire_utility"] if called else source["stop_utility"])
        target_acquire = bool(source["target_acquire"])
        evaluated.append(
            {
                **source,
                "called": called,
                "false_call": called and not target_acquire,
                "harmful_call": called
                and float(source["acquire_utility"]) < float(source["stop_utility"]),
                "exact_acquire": called
                and target_acquire
                and int(source["best_evidence_index"]) == int(source["target_index"]),
                "utility": utility,
                "regret": float(source["oracle_utility"]) - utility,
                "predicted_index": int(source["best_evidence_index"]) if called else -1,
            }
        )
    return evaluated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", action="append", type=_candidate, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--calibration-group-key", default="source_episode")
    parser.add_argument("--max-false-call-rate", type=float, default=0.02)
    parser.add_argument("--validation-max-harmful", type=float, default=0.30)
    parser.add_argument("--validation-min-recall", type=float, default=0.01)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    labels = [str(row["label"]) for row in args.candidate]
    if len(set(labels)) != len(labels):
        raise ValueError("candidate labels must be unique")
    device = torch.device(args.device)
    inputs = calibration_inputs(
        args.checkpoint,
        device=device,
        batch_size=args.batch_size,
        calibration_fraction=args.calibration_fraction,
        calibration_group_key=args.calibration_group_key,
    )
    validation_rows = evaluate(
        args.checkpoint,
        device=device,
        batch_size=args.batch_size,
        stop_margin_override=0.0,
    )
    rows: list[dict[str, Any]] = []
    for candidate in args.candidate:
        calibration = select_stop_margin(
            *inputs,
            max_false_call_rate=args.max_false_call_rate,
            max_harmful_call_fraction=float(candidate["max_harmful_call_fraction"]),
            min_acquire_recall=float(candidate["min_acquire_recall"]),
        )
        validation = aggregate_rows(
            apply_margin(validation_rows, float(calibration["stop_margin"]))
        )
        eligible = bool(
            calibration["constraints_satisfied"]
            and validation["mean_utility"] > 0.0
            and validation["false_call_rate"] <= args.max_false_call_rate
            and validation["harmful_call_fraction"] <= args.validation_max_harmful
            and validation["acquire_recall"] >= args.validation_min_recall
        )
        rows.append(
            {
                **candidate,
                "calibration": calibration,
                "validation": validation,
                "eligible": eligible,
            }
        )
    feasible = [row for row in rows if row["eligible"]]
    winner = (
        max(
            feasible,
            key=lambda row: (
                row["validation"]["mean_utility"],
                -row["validation"]["false_call_rate"],
            ),
        )
        if feasible
        else None
    )
    payload = {
        "schema_version": "selector-safety-calibration-sweep-v1",
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": _sha256(args.checkpoint),
        },
        "selection_partition": "validation",
        "constraints": {
            "max_false_call_rate": args.max_false_call_rate,
            "max_harmful_call_fraction": args.validation_max_harmful,
            "min_acquire_recall": args.validation_min_recall,
        },
        "promoted": winner is not None,
        "winner": winner,
        "rows": rows,
        "test_assets_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
