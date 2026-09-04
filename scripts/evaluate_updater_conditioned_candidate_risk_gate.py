#!/usr/bin/env python3
"""Calibrate/evaluate a candidate-risk gate for C5 refreshed selection."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).parent


def _module(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _module("c5_audit", "audit_updater_conditioned_disagreements.py")
RISK = _module("c5_risk", "train_updater_conditioned_candidate_risk.py")


def _risk_features(sample: Any, index: int) -> list[float]:
    return [
        *[float(value) for value in sample.hypothesis_features],
        *[float(value) for value in sample.state_features],
        *[float(value) for value in sample.evidence_features[index]],
        float(sample.false_edit_risks[index]), float(sample.evidence_costs[index]),
    ]


def _risk_probabilities(checkpoint_path: Path, samples: list[Any], choices: dict[str, dict[str, Any]], device: torch.device) -> dict[str, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = RISK.CandidateRiskHead(int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"]), float(checkpoint["dropout"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    ids, features = [], []
    for sample in samples:
        choice = choices[str(sample.sample_id)]
        if not choice["stop"]:
            ids.append(str(sample.sample_id))
            features.append(_risk_features(sample, int(choice["index"])))
    result = {str(sample.sample_id): float("-inf") for sample in samples}
    if not features:
        return result
    matrix = (np.asarray(features, dtype=np.float32) - mean) / std
    with torch.no_grad():
        for start in range(0, len(matrix), 4096):
            scores = torch.sigmoid(model(torch.from_numpy(matrix[start:start + 4096]).to(device))).cpu().numpy()
            result.update({sample_id: float(score) for sample_id, score in zip(ids[start:start + 4096], scores)})
    return result


def _outcome(sample: Any, choice: dict[str, Any]) -> tuple[dict[str, Any], float, float]:
    evidence_id = str(sample.metadata["initial_evidence_id"]) if choice["stop"] else str(sample.evidence_ids[int(choice["index"])])
    outcome = AUDIT._outcome(sample, evidence_id)
    utility = float(sample.stop_utility) if choice["stop"] else float(sample.oracle_utilities[int(choice["index"])])
    cost = 0.0 if choice["stop"] else float(sample.evidence_costs[int(choice["index"])])
    return outcome, utility, cost


def _metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    chosen = [row["stale"] if row["risk"] >= threshold else row["refreshed"] for row in rows]
    return {
        "utility": float(np.mean([item["utility"] for item in chosen])),
        "raster_iou": float(np.mean([item["raster_iou"] for item in chosen])),
        "false_edit": float(np.mean([item["false_edit"] for item in chosen])),
        "missed_edit": float(np.mean([item["missed_edit"] for item in chosen])),
        "cost": float(np.mean([item["cost"] for item in chosen])),
        "gate_rate": float(np.mean([row["risk"] >= threshold for row in rows])),
    }


def select_threshold(rows: list[dict[str, Any]], *, max_false_edit_increase: float) -> dict[str, Any]:
    stale = _metrics(rows, float("-inf"))
    risks = np.asarray([row["risk"] for row in rows if np.isfinite(row["risk"])], dtype=np.float64)
    thresholds = [float("inf"), *[float(value) for value in np.quantile(risks, np.linspace(0.0, 1.0, 101))]]
    candidates = []
    for threshold in sorted(set(thresholds), reverse=True):
        metrics = _metrics(rows, threshold)
        feasible = (
            metrics["false_edit"] <= stale["false_edit"] + max_false_edit_increase + 1e-12
            and metrics["utility"] >= stale["utility"] - 1e-12
            and metrics["raster_iou"] >= stale["raster_iou"] - 1e-12
        )
        candidates.append({"threshold": threshold, "metrics": metrics, "feasible": feasible})
    feasible = [item for item in candidates if item["feasible"] and item["metrics"]["gate_rate"] > 0.0]
    result = {"stale_metrics": stale, "candidates": candidates}
    if not feasible:
        result.update({"status": "infeasible", "test_assets_read": False})
        return result
    selected = max(feasible, key=lambda item: (-item["metrics"]["missed_edit"], item["metrics"]["utility"], item["metrics"]["raster_iou"], -item["metrics"]["false_edit"], -item["metrics"]["cost"]))
    result.update({"status": "complete", "selected": selected, "test_assets_read": False})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("risk_checkpoint", type=Path)
    parser.add_argument("stale_checkpoint", type=Path)
    parser.add_argument("refreshed_checkpoint", type=Path)
    parser.add_argument("states", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--mode", choices=("calibrate", "evaluate"), required=True)
    parser.add_argument("--risk-threshold", type=float)
    parser.add_argument("--max-false-edit-increase", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    device = torch.device(args.device)
    samples, stale_choices = AUDIT._choice_records(args.stale_checkpoint, args.states, device=device, batch_size=256, split=args.split)
    refreshed_samples, refreshed_choices = AUDIT._choice_records(args.refreshed_checkpoint, args.states, device=device, batch_size=256, split=args.split)
    if [sample.sample_id for sample in samples] != [sample.sample_id for sample in refreshed_samples]:
        raise ValueError("stale and refreshed sample ordering differs")
    risk = _risk_probabilities(args.risk_checkpoint, samples, refreshed_choices, device)
    rows = []
    for sample in samples:
        sample_id = str(sample.sample_id)
        stale_outcome, stale_utility, stale_cost = _outcome(sample, stale_choices[sample_id])
        refreshed_outcome, refreshed_utility, refreshed_cost = _outcome(sample, refreshed_choices[sample_id])
        rows.append({"sample_id": sample_id, "aoi_id": str(sample.metadata.get("aoi_id", "unknown")), "risk": risk[sample_id], "stale": {"utility": stale_utility, "raster_iou": float(stale_outcome["final_raster_iou"]), "false_edit": float(bool(stale_outcome["false_edit"])), "missed_edit": float(bool(stale_outcome["missed_edit"])), "cost": stale_cost}, "refreshed": {"utility": refreshed_utility, "raster_iou": float(refreshed_outcome["final_raster_iou"]), "false_edit": float(bool(refreshed_outcome["false_edit"])), "missed_edit": float(bool(refreshed_outcome["missed_edit"])), "cost": refreshed_cost}})
    if args.mode == "calibrate":
        summary = select_threshold(rows, max_false_edit_increase=args.max_false_edit_increase)
    else:
        if args.risk_threshold is None:
            raise ValueError("--risk-threshold is required for evaluation")
        summary = {"status": "complete", "risk_threshold": args.risk_threshold, "metrics": _metrics(rows, args.risk_threshold), "stale_metrics": _metrics(rows, float("-inf")), "test_assets_read": False}
    summary.update({"mode": args.mode, "split": args.split, "sample_count": len(rows), "risk_checkpoint": str(args.risk_checkpoint.resolve()), "states": str(args.states.resolve())})
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
