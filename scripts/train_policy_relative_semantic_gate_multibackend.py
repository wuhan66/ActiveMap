#!/usr/bin/env python3
"""Train one no-leak semantic CALL/STOP gate across multiple backend seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from activemap.agent.active_catalog_tool_gate import (
    policy_relative_semantic_gate_features,
)
from activemap.agent.evidence_value_head import StructuredMapActionPredictor
from activemap.agent.identifiers import public_task_id
from activemap.agent.post_tool_action_adapter import PostToolActionAdapterPredictor
from activemap.agent.tool_belief_data import PostAcquisitionToolPairExample
from activemap.agent.tool_sft import terminal_reward
from activemap.agent.tools import belief_from_features
from activemap.evaluation.episode_utility import UTILITY_PROFILES
from activemap.models import EditOperation
from activemap.selector_records import SelectorSample
from scripts.train_policy_relative_semantic_gate import _fit, _policy_metrics

EDIT_ORDER = list(EditOperation)


@dataclass(frozen=True)
class BackendSpec:
    label: str
    train: Path
    val: Path
    adapter: Path


def _parse_backend(value: str) -> BackendSpec:
    label, separator, payload = value.partition("=")
    parts = payload.split(",")
    if not separator or not label or len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "backend must use LABEL=TRAIN_JSONL,VAL_JSONL,ADAPTER_PT"
        )
    if not label.replace("_", "").isalnum():
        raise argparse.ArgumentTypeError("backend label must be alphanumeric")
    return BackendSpec(label, *(Path(part) for part in parts))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_states(path: Path) -> list[SelectorSample]:
    rows = [
        SelectorSample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("empty structured state file")
    return rows


def _semantic_by_task(
    paths: tuple[Path, Path],
) -> dict[str, PostAcquisitionToolPairExample]:
    result: dict[str, PostAcquisitionToolPairExample] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = PostAcquisitionToolPairExample.model_validate_json(line)
            if row.semantic_result is None:
                raise ValueError(f"missing semantic result: {row.example_id}")
            previous = result.get(row.task_id)
            if previous is not None:
                if previous.semantic_result.outputs != row.semantic_result.outputs:
                    raise ValueError(
                        f"task has inconsistent semantic results: {row.task_id}"
                    )
                continue
            result[row.task_id] = row
    if not result:
        raise ValueError("empty semantic result files")
    return result


def consensus_beneficial_labels(
    direct_reward: np.ndarray,
    semantic_reward: np.ndarray,
    utility_cost: np.ndarray,
    *,
    minimum_votes: int,
) -> np.ndarray:
    if semantic_reward.shape != utility_cost.shape:
        raise ValueError("semantic reward and utility cost shapes differ")
    if semantic_reward.ndim != 2 or semantic_reward.shape[1] != len(direct_reward):
        raise ValueError("backend reward matrix shape mismatch")
    if not 1 <= minimum_votes <= semantic_reward.shape[0]:
        raise ValueError("minimum_votes must be within backend count")
    gains = semantic_reward - utility_cost - direct_reward[None, :]
    return (
        (np.sum(gains > 0.0, axis=0) >= minimum_votes)
        & (np.mean(gains, axis=0) > 0.0)
    ).astype(np.int64)


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = (
        "accuracy",
        "macro_f1",
        "false_edit_rate",
        "missed_edit_rate",
        "call_rate",
        "mean_cost",
        "mean_utility_cost",
        "mean_utility",
    )
    return {
        key: float(np.mean([metric[key] for metric in metrics]))
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("structured_checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--backend",
        action="append",
        type=_parse_backend,
        required=True,
    )
    parser.add_argument("--minimum-votes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--false-edit-delta", type=float, default=0.02)
    parser.add_argument(
        "--utility-profile",
        choices=tuple(UTILITY_PROFILES),
        default="balanced",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    labels = [backend.label for backend in args.backend]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate backend label")
    if len(args.backend) < 2:
        raise ValueError("multi-backend gate requires at least two backends")

    structured = StructuredMapActionPredictor(
        str(args.structured_checkpoint), device="cpu"
    )
    backend_rows = [
        _semantic_by_task((backend.train, backend.val))
        for backend in args.backend
    ]
    adapters = [
        PostToolActionAdapterPredictor(backend.adapter, device="cpu")
        for backend in args.backend
    ]
    records: dict[str, list[Any]] = {
        key: []
        for key in ("features", "target", "direct", "group", "split")
    }
    semantic_predictions: list[list[int]] = [[] for _ in args.backend]
    raw_costs: list[list[float]] = [[] for _ in args.backend]
    utility_costs: list[list[float]] = [[] for _ in args.backend]
    skipped = 0
    for sample in _read_states(args.states):
        task_id = public_task_id(str(sample.metadata["source_episode"]))
        semantic_examples = [rows.get(task_id) for rows in backend_rows]
        if any(example is None for example in semantic_examples):
            skipped += 1
            continue
        _, direct_edit = structured.predict(sample)
        records["features"].append(
            policy_relative_semantic_gate_features(sample)
        )
        records["target"].append(
            EDIT_ORDER.index(EditOperation(str(sample.metadata["gt_edit"])))
        )
        records["direct"].append(EDIT_ORDER.index(direct_edit))
        records["group"].append(task_id)
        records["split"].append(sample.split)
        budget = float(sample.metadata["budget"])
        for index, (adapter, example) in enumerate(
            zip(adapters, semantic_examples, strict=True)
        ):
            assert example is not None and example.semantic_result is not None
            prediction = adapter.predict(
                belief_from_features(sample),
                [example.semantic_result],
            )
            semantic_predictions[index].append(EDIT_ORDER.index(prediction))
            raw_cost = float(example.semantic_result.cost)
            raw_costs[index].append(raw_cost)
            utility_costs[index].append(
                UTILITY_PROFILES[args.utility_profile].cost
                * min(raw_cost / budget, 1.0)
            )

    arrays = {key: np.asarray(value) for key, value in records.items()}
    semantic = np.asarray(semantic_predictions, dtype=np.int64)
    raw_cost = np.asarray(raw_costs, dtype=np.float64)
    utility_cost = np.asarray(utility_costs, dtype=np.float64)
    train_index = np.flatnonzero(arrays["split"] == "train")
    val_index = np.flatnonzero(arrays["split"] == "val")
    if not len(train_index) or not len(val_index):
        raise ValueError("gate data lacks train or validation support")

    target = arrays["target"].astype(np.int64)
    direct = arrays["direct"].astype(np.int64)
    direct_reward = np.asarray(
        [
            terminal_reward(EDIT_ORDER[int(gt)], EDIT_ORDER[int(pred)])
            for gt, pred in zip(target, direct, strict=True)
        ],
        dtype=np.float64,
    )
    semantic_reward = np.asarray(
        [
            [
                terminal_reward(EDIT_ORDER[int(gt)], EDIT_ORDER[int(pred)])
                for gt, pred in zip(target, backend_prediction, strict=True)
            ]
            for backend_prediction in semantic
        ],
        dtype=np.float64,
    )
    beneficial = consensus_beneficial_labels(
        direct_reward[train_index],
        semantic_reward[:, train_index],
        utility_cost[:, train_index],
        minimum_votes=args.minimum_votes,
    )
    if len(np.unique(beneficial)) != 2:
        raise ValueError("consensus labels contain only one class")

    c_values = (0.01, 0.1, 1.0, 10.0)
    thresholds = np.linspace(0.1, 0.9, 33)
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    direct_metrics = _policy_metrics(
        target[train_index],
        direct[train_index],
        semantic[0, train_index],
        np.zeros(len(train_index), dtype=bool),
        raw_cost[0, train_index],
        utility_cost[0, train_index],
    )
    best = None
    for c_value in c_values:
        probability = np.zeros(len(train_index), dtype=np.float64)
        for fit_index, holdout_index in splitter.split(
            arrays["features"][train_index],
            beneficial,
            arrays["group"][train_index],
        ):
            model = _fit(c_value, args.seed)
            model.fit(
                arrays["features"][train_index][fit_index],
                beneficial[fit_index],
            )
            probability[holdout_index] = model.predict_proba(
                arrays["features"][train_index][holdout_index]
            )[:, 1]
        for threshold in thresholds:
            call = probability >= threshold
            backend_metrics = [
                _policy_metrics(
                    target[train_index],
                    direct[train_index],
                    semantic[index, train_index],
                    call,
                    raw_cost[index, train_index],
                    utility_cost[index, train_index],
                )
                for index in range(len(args.backend))
            ]
            feasible = all(
                metric["false_edit_rate"]
                <= direct_metrics["false_edit_rate"] + args.false_edit_delta
                for metric in backend_metrics
            )
            utilities = [metric["mean_utility"] for metric in backend_metrics]
            macro_f1 = [metric["macro_f1"] for metric in backend_metrics]
            candidate = (
                feasible,
                float(np.mean(utilities)),
                float(np.min(utilities)),
                float(np.mean(macro_f1)),
                -float(np.mean(call)),
                -c_value,
                -float(threshold),
                c_value,
                float(threshold),
                backend_metrics,
            )
            if best is None or candidate[:7] > best[:7]:
                best = candidate
    assert best is not None
    c_value, threshold, train_backend_metrics = best[7], best[8], best[9]
    gate = _fit(c_value, args.seed)
    gate.fit(arrays["features"][train_index], beneficial)
    val_probability = gate.predict_proba(arrays["features"][val_index])[:, 1]
    val_call = val_probability >= threshold
    val_direct = _policy_metrics(
        target[val_index],
        direct[val_index],
        semantic[0, val_index],
        np.zeros(len(val_index), dtype=bool),
        raw_cost[0, val_index],
        utility_cost[0, val_index],
    )
    val_backends = {}
    for index, backend in enumerate(args.backend):
        metrics = _policy_metrics(
            target[val_index],
            direct[val_index],
            semantic[index, val_index],
            val_call,
            raw_cost[index, val_index],
            utility_cost[index, val_index],
        )
        val_backends[backend.label] = {
            "selective": metrics,
            "deltas": {
                key: metrics[key] - val_direct[key]
                for key in val_direct
            },
        }
    aggregate = _mean_metrics(
        [row["selective"] for row in val_backends.values()]
    )

    args.output_dir.mkdir(parents=True)
    joblib.dump(gate, args.output_dir / "gate.joblib")
    scaler = gate.named_steps["standardscaler"]
    classifier = gate.named_steps["logisticregression"]
    portable_gate = {
        "schema_version": "linear-probability-gate-v1",
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficient": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
    }
    (args.output_dir / "gate.json").write_text(
        json.dumps(portable_gate, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "policy-relative-semantic-multibackend-gate-v1",
        "train_examples": int(len(train_index)),
        "val_examples": int(len(val_index)),
        "skipped_states": skipped,
        "backend_labels": labels,
        "minimum_votes": args.minimum_votes,
        "train_consensus_beneficial_rate": float(np.mean(beneficial)),
        "selected": {
            "C": c_value,
            "threshold": threshold,
            "train_backend_metrics": {
                backend.label: metric
                for backend, metric in zip(
                    args.backend, train_backend_metrics, strict=True
                )
            },
        },
        "val": {
            "direct": val_direct,
            "backends": val_backends,
            "mean_selective": aggregate,
            "all_backends_positive_utility_delta": all(
                row["deltas"]["mean_utility"] > 0.0
                for row in val_backends.values()
            ),
            "all_backends_safe": all(
                row["selective"]["false_edit_rate"]
                <= val_direct["false_edit_rate"] + args.false_edit_delta
                for row in val_backends.values()
            ),
        },
        "sources": {
            "states": _sha256(args.states),
            "structured_checkpoint": _sha256(args.structured_checkpoint),
            "backends": {
                backend.label: {
                    "train": _sha256(backend.train),
                    "val": _sha256(backend.val),
                    "adapter": _sha256(backend.adapter),
                }
                for backend in args.backend
            },
        },
        "selection_protocol": (
            "train-only-stratified-grouped-OOF;"
            "mean-then-worst-backend-utility"
        ),
        "runtime_backend_calls": 1,
        "ensemble_runtime_cost_hidden": False,
        "utility_profile": args.utility_profile,
        "test_assets_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
