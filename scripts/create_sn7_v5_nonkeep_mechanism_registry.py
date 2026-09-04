#!/usr/bin/env python3
"""Freeze the validation-only V5 non-KEEP policy-mechanism contract.

The registry is created before a Stage-B selector or validation writeback
exists.  It binds the common support, independent updater seeds, selector
templates, stochastic controls, and statistical plan so later results cannot
silently substitute a selector-specific Safe Commit gate or a different sample
population.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.calibrate_sn7_v5_common_safe_commit import POLICIES as DEPLOYABLE_POLICIES


SEEDS = (20260817, 20260818, 20260819)
ALL_POLICIES = (*DEPLOYABLE_POLICIES, "oracle")
RANDOM_CONTROL_SEEDS = (20260829, 20260830, 20260831, 20260832, 20260833)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_seed_path(value: str) -> tuple[int, Path]:
    raw_seed, separator, raw_path = value.partition("=")
    if not separator or not raw_path:
        raise argparse.ArgumentTypeError("records must use SEED=PATH")
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("SEED must be an integer") from exc
    if seed not in SEEDS:
        raise argparse.ArgumentTypeError(f"SEED must be one of {SEEDS}")
    return seed, Path(raw_path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def require_false(payload: dict[str, Any], field: str, source: Path) -> None:
    if payload.get(field) is not False:
        raise ValueError(f"{source} must record {field}=false")


def normalize_template(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove only the two declared generic-vs-policy-relative differences."""

    result = copy.deepcopy(payload)
    for key in ("seed", "output_dir"):
        result.pop(key, None)
    data = result.get("data")
    training = result.get("training")
    ablation = result.get("ablation")
    if not isinstance(data, dict) or not isinstance(training, dict) or not isinstance(ablation, dict):
        raise ValueError("selector template lacks data, training, or ablation mapping")
    data.pop("samples", None)
    training.pop("device", None)
    ablation.pop("name", None)
    ablation.pop("condition_on_hypothesis", None)
    return result


def validate_selector_templates(generic_path: Path, policy_relative_path: Path) -> dict[str, Any]:
    generic = yaml.safe_load(generic_path.read_text(encoding="utf-8"))
    policy_relative = yaml.safe_load(policy_relative_path.read_text(encoding="utf-8"))
    if not isinstance(generic, dict) or not isinstance(policy_relative, dict):
        raise ValueError("selector templates must be YAML mappings")
    generic_ablation = generic.get("ablation")
    policy_ablation = policy_relative.get("ablation")
    if not isinstance(generic_ablation, dict) or not isinstance(policy_ablation, dict):
        raise ValueError("selector templates lack ablation mapping")
    if generic_ablation.get("condition_on_hypothesis") is not False:
        raise ValueError("generic template must hide typed-hypothesis conditioning")
    if policy_ablation.get("condition_on_hypothesis") is not True:
        raise ValueError("policy-relative template must enable typed-hypothesis conditioning")
    if normalize_template(generic) != normalize_template(policy_relative):
        raise ValueError("selector templates differ outside their declared hypothesis condition")
    return {
        "generic": {"path": str(generic_path.resolve()), "sha256": sha256(generic_path)},
        "policy_relative": {
            "path": str(policy_relative_path.resolve()),
            "sha256": sha256(policy_relative_path),
        },
        "capacity_matched": True,
        "only_declared_difference": "ablation.condition_on_hypothesis",
    }


def validate_authorization(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != "sn7-v5b-three-seed-headroom-authorization-v1":
        raise ValueError("unexpected V5 headroom authorization schema")
    if payload.get("authorization") != "matched_nonkeep_factorial_writeback":
        raise ValueError("V5 authorization does not permit matched non-KEEP writeback")
    require_false(payload, "test_assets_read", path)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("V5 authorization lacks records")
    by_seed = {int(record.get("seed", -1)): record for record in records if isinstance(record, dict)}
    if tuple(sorted(by_seed)) != SEEDS:
        raise ValueError("V5 authorization does not contain exactly the registered seeds")
    for seed in SEEDS:
        record = by_seed[seed]
        if record.get("passed") is not True or record.get("test_assets_read") is not False:
            raise ValueError(f"V5 authorization is not passing and isolated for seed {seed}")
        state_hash = record.get("state_file_sha256")
        if not isinstance(state_hash, str) or len(state_hash) != 64:
            raise ValueError(f"V5 authorization lacks validation-state hash for seed {seed}")
    return {"path": str(path.resolve()), "sha256": sha256(path), "records": by_seed}


def validate_train_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = 0
    episodes: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("split") != "train":
                raise ValueError(f"{path}:{line_number} is not an original train selector state")
            if row.get("test_assets_read") is True:
                raise ValueError(f"{path}:{line_number} is test-contaminated")
            source_episode = row.get("source_episode") or row.get("metadata", {}).get("source_episode")
            if not isinstance(source_episode, str) or not source_episode:
                raise ValueError(f"{path}:{line_number} lacks source episode")
            episodes.add(source_episode)
            rows += 1
    if rows == 0:
        raise ValueError(f"{path} contains no train selector states")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "rows": rows,
        "source_episode_count": len(episodes),
    }


def validate_path_records(
    records: list[tuple[int, Path]], *, label: str, validator: Any
) -> dict[str, Any]:
    values = dict(records)
    if len(values) != len(records) or tuple(sorted(values)) != SEEDS:
        raise ValueError(f"{label} records must contain each registered seed exactly once")
    return {str(seed): validator(values[seed]) for seed in SEEDS}


def build_registry(
    *,
    authorization_path: Path,
    generic_template: Path,
    policy_relative_template: Path,
    train_states: list[tuple[int, Path]],
    updater_checkpoints: list[tuple[int, Path]],
    bootstrap_seed: int,
) -> dict[str, Any]:
    authorization = validate_authorization(authorization_path)
    states = validate_path_records(train_states, label="train-state", validator=validate_train_state)
    checkpoints = validate_path_records(
        updater_checkpoints,
        label="updater-checkpoint",
        validator=lambda path: {"path": str(path.resolve()), "sha256": sha256(path)},
    )
    return {
        "schema_version": "sn7-v5-nonkeep-policy-mechanism-registry-v1",
        "split": "train,val",
        "analysis_role": "validation_only_mechanism_test",
        "test_assets_read": False,
        "registered_updater_seeds": list(SEEDS),
        "authorization": {key: value for key, value in authorization.items() if key != "records"},
        "authorized_validation_state_sha256": {
            str(seed): authorization["records"][seed]["state_file_sha256"] for seed in SEEDS
        },
        "selector_templates": validate_selector_templates(generic_template, policy_relative_template),
        "train_states": states,
        "updater_checkpoints": checkpoints,
        "policies": {
            "direct": {"trainable": False, "evidence": "none"},
            "random": {
                "trainable": False,
                "evidence": "deterministic_feasible_random",
                "episode_hash_seeds": list(RANDOM_CONTROL_SEEDS),
            },
            "uncertainty": {
                "trainable": False,
                "evidence": "train_only_rate_matched_scalar_uncertainty",
            },
            "generic": {
                "trainable": True,
                "template": "generic",
                "typed_hypothesis_features": False,
            },
            "policy_relative": {
                "trainable": True,
                "template": "policy_relative",
                "typed_hypothesis_features": True,
            },
            "forced": {"trainable": False, "evidence": "always_feasible"},
            "oracle": {"trainable": False, "evidence": "counterfactual_shortlist_only"},
        },
        "common_safe_commit": {
            "train_only": True,
            "policy_blind": True,
            "policies": list(DEPLOYABLE_POLICIES),
            "replay_iou_threshold": 0.99,
            "require_topology": True,
            "threshold_count": 101,
        },
        "rate_matching": {
            "reference_policy": "policy_relative",
            "policies": ["uncertainty", "generic", "policy_relative"],
            "calibration_split": "train",
            "validation_tolerance": 0.0025,
        },
        "statistics": {
            "bootstrap": "hierarchical_model_seed_then_aoi_paired",
            "bootstrap_repetitions": 10000,
            "bootstrap_seed": bootstrap_seed,
            "minimum_aois": 4,
            "minimum_rows_per_operation_per_seed": 20,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--generic-template", required=True, type=Path)
    parser.add_argument("--policy-relative-template", required=True, type=Path)
    parser.add_argument("--train-state", action="append", type=parse_seed_path, required=True)
    parser.add_argument("--updater-checkpoint", action="append", type=parse_seed_path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite mechanism registry: {args.output}")
    result = build_registry(
        authorization_path=args.authorization,
        generic_template=args.generic_template,
        policy_relative_template=args.policy_relative_template,
        train_states=args.train_state,
        updater_checkpoints=args.updater_checkpoint,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
