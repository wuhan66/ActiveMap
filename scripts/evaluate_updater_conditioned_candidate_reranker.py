#!/usr/bin/env python3
"""Train/dev-calibrated C5 risk-aware candidate reranking.

Unlike the earlier C5 fallback, this policy does not first select one refreshed
candidate and then decide whether to substitute the stale action.  It scores
every deployment-visible candidate with the frozen refreshed selector and the
train-only candidate-risk head, then selects the safest high-value candidate
or STOP.  Calibration never reads test data and may only freeze a penalty and
risk cap; evaluation applies those values once to a held-out validation cache.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from activemap.features import AblationSpec
from activemap.nn.selector import EvidenceSelector, SelectorConfig
from activemap.training.data import (
    SelectorDataset,
    collate_selector_batch,
    load_selector_samples,
)

ROOT = Path(__file__).parent


def _module(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EVAL = _module("c5_selector_eval", "evaluate_updater_conditioned_selector.py")
RISK = _module("c5_candidate_risk", "train_updater_conditioned_candidate_risk.py")


def choose_candidate(
    logits: np.ndarray,
    stop_logit: float,
    risks: np.ndarray,
    *,
    risk_penalty: float,
    max_risk: float,
) -> int | None:
    """Choose a candidate or STOP from public logits and predicted risks."""

    if logits.ndim != 1 or risks.shape != logits.shape:
        raise ValueError("candidate logits and risks must have identical 1-D shapes")
    if risk_penalty < 0.0 or not 0.0 <= max_risk <= 1.0:
        raise ValueError("invalid reranker hyperparameter")
    adjusted = logits.astype(np.float64, copy=True) - risk_penalty * risks
    adjusted[risks > max_risk] = -np.inf
    index = int(np.argmax(np.concatenate([adjusted, [float(stop_logit)]])))
    return None if index == len(adjusted) else index


def _risk_features(sample: Any, index: int) -> list[float]:
    return [
        *map(float, sample.hypothesis_features),
        *map(float, sample.state_features),
        *map(float, sample.evidence_features[index]),
        float(sample.false_edit_risks[index]),
        float(sample.evidence_costs[index]),
    ]


def _load_risk(path: Path, device: torch.device) -> tuple[Any, np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = RISK.CandidateRiskHead(
        int(payload["input_dim"]), int(payload["hidden_dim"]), float(payload["dropout"])
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, np.asarray(payload["feature_mean"], dtype=np.float32), np.asarray(payload["feature_std"], dtype=np.float32)


@torch.no_grad()
def collect_rows(
    selector_path: Path, risk_path: Path, states: Path, *, split: str, device: torch.device, batch_size: int
) -> list[dict[str, Any]]:
    checkpoint = torch.load(selector_path, map_location=device, weights_only=False)
    selector = EvidenceSelector(SelectorConfig(**checkpoint["model_config"]))
    selector.load_state_dict(checkpoint["state_dict"])
    selector.to(device).eval()
    samples = load_selector_samples(states, split=split)
    if not samples or any(sample.split == "test" for sample in samples):
        raise ValueError("C5 reranking permits train/validation samples only")
    dataset = SelectorDataset(samples, AblationSpec(**checkpoint["ablation"]), EVAL._normalizer(checkpoint.get("feature_normalizer")))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_selector_batch)
    risk_model, mean, std = _load_risk(risk_path, device)
    result: list[dict[str, Any]] = []
    offset = 0
    for batch in loader:
        logits = selector(batch["evidence"].to(device), batch["hypothesis"].to(device), batch["state"].to(device), batch["mask"].to(device))
        logits[:, -1] += float(checkpoint.get("stop_margin", 0.0))
        for local in range(logits.shape[0]):
            sample = samples[offset + local]
            count = len(sample.evidence_ids)
            features = (np.asarray([_risk_features(sample, i) for i in range(count)], dtype=np.float32) - mean) / std
            risks = torch.sigmoid(risk_model(torch.from_numpy(features).to(device))).cpu().numpy().astype(float)
            candidate_logits = logits[local, :count].cpu().numpy().astype(float)
            stop_logit = float(logits[local, -1].cpu())
            candidates = []
            for index, evidence_id in enumerate(sample.evidence_ids):
                outcome = EVAL._outcome(sample, str(evidence_id))
                candidates.append({"utility": float(sample.oracle_utilities[index]), "raster_iou": float(outcome["final_raster_iou"]), "false_edit": float(bool(outcome["false_edit"])), "missed_edit": float(bool(outcome["missed_edit"])), "cost": float(sample.evidence_costs[index])})
            stop_outcome = EVAL._outcome(sample, EVAL._direct_evidence_id(sample))
            result.append({"sample_id": str(sample.sample_id), "aoi_id": str(sample.metadata.get("aoi_id", "unknown")), "candidate_logits": candidate_logits.tolist(), "stop_logit": stop_logit, "risks": risks.tolist(), "candidates": candidates, "stop": {"utility": float(sample.stop_utility), "raster_iou": float(stop_outcome["final_raster_iou"]), "false_edit": float(bool(stop_outcome["false_edit"])), "missed_edit": float(bool(stop_outcome["missed_edit"])), "cost": 0.0}})
        offset += logits.shape[0]
    return result


def metrics(rows: list[dict[str, Any]], penalty: float, cap: float) -> dict[str, float]:
    chosen = []
    for row in rows:
        index = choose_candidate(np.asarray(row["candidate_logits"]), float(row["stop_logit"]), np.asarray(row["risks"]), risk_penalty=penalty, max_risk=cap)
        chosen.append(row["stop"] if index is None else row["candidates"][index])
    return {key: float(np.mean([item[key] for item in chosen])) for key in ("utility", "raster_iou", "false_edit", "missed_edit", "cost")}


def calibrate(rows: list[dict[str, Any]], *, stale: dict[str, float]) -> dict[str, Any]:
    all_risks = np.asarray([risk for row in rows for risk in row["risks"]], dtype=np.float64)
    caps = sorted({1.0, *[float(v) for v in np.quantile(all_risks, np.linspace(0.05, 1.0, 20))]})
    penalties = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    candidates = []
    for penalty in penalties:
        for cap in caps:
            value = metrics(rows, penalty, cap)
            feasible = value["false_edit"] <= stale["false_edit"] + 1e-12 and value["utility"] >= stale["utility"] - 1e-12 and value["raster_iou"] >= stale["raster_iou"] - 1e-12
            candidates.append({"risk_penalty": penalty, "max_risk": cap, "metrics": value, "feasible": feasible})
    feasible = [row for row in candidates if row["feasible"]]
    output = {"schema_version": "c5-candidate-rerank-calibration-v1", "candidate_count": len(candidates), "candidates": candidates, "test_assets_read": False}
    if not feasible:
        return {**output, "status": "infeasible"}
    selected = max(feasible, key=lambda row: (-row["metrics"]["missed_edit"], row["metrics"]["utility"], row["metrics"]["raster_iou"], -row["metrics"]["false_edit"], -row["metrics"]["cost"]))
    return {**output, "status": "complete", "selected": selected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selector", type=Path)
    parser.add_argument("risk_checkpoint", type=Path)
    parser.add_argument("states", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--mode", choices=("calibrate", "evaluate"), required=True)
    parser.add_argument("--stale-metrics", type=Path, help="JSON containing utility/raster_iou/false_edit")
    parser.add_argument("--risk-penalty", type=float)
    parser.add_argument("--max-risk", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    rows = collect_rows(args.selector, args.risk_checkpoint, args.states, split=args.split, device=torch.device(args.device), batch_size=args.batch_size)
    if args.mode == "calibrate":
        if args.stale_metrics is None:
            raise ValueError("--stale-metrics is required for calibration")
        stale = json.loads(args.stale_metrics.read_text(encoding="utf-8"))
        stale = stale.get("stale_metrics", stale)
        summary = calibrate(rows, stale={key: float(stale[key]) for key in ("utility", "raster_iou", "false_edit")})
    else:
        if args.risk_penalty is None or args.max_risk is None:
            raise ValueError("evaluation requires --risk-penalty and --max-risk")
        summary = {"schema_version": "c5-candidate-rerank-evaluation-v1", "status": "complete", "metrics": metrics(rows, args.risk_penalty, args.max_risk), "risk_penalty": args.risk_penalty, "max_risk": args.max_risk, "test_assets_read": False}
    summary.update({"mode": args.mode, "split": args.split, "sample_count": len(rows)})
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
