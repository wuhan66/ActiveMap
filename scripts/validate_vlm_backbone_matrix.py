#!/usr/bin/env python3
"""Validate the frozen fairness contract for the VLM backbone experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != "activemap-vlm-backbone-matrix-v1":
        raise ValueError("unsupported VLM matrix schema")
    if payload.get("test_assets_read") is not False:
        raise ValueError("VLM matrix must keep test assets frozen")
    data = payload.get("data")
    protocol = payload.get("common_protocol")
    backbones = payload.get("backbones")
    if not isinstance(data, dict) or not isinstance(protocol, dict):
        raise ValueError("matrix requires data and common_protocol mappings")
    if not isinstance(backbones, list) or len(backbones) < 3:
        raise ValueError("matrix requires at least three VLM backbones")
    if any("test" in str(key).lower() for key in data):
        raise ValueError("matrix data section must not reference test assets")
    ids = [str(row.get("id", "")) for row in backbones]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("backbone ids must be unique and nonempty")
    roles = {str(row.get("role")): row for row in backbones}
    required_roles = {"primary", "same_scale_ablation", "architecture_robustness"}
    if not required_roles <= roles.keys():
        raise ValueError("matrix is missing a required backbone role")
    fixed_seeds = [int(seed) for seed in protocol.get("fixed_seeds", [])]
    if len(fixed_seeds) < 3 or len(set(fixed_seeds)) != len(fixed_seeds):
        raise ValueError("common protocol requires at least three unique fixed seeds")
    for role in ("primary", "same_scale_ablation"):
        seeds = [int(seed) for seed in roles[role].get("seeds", [])]
        if seeds != fixed_seeds:
            raise ValueError(f"{role} must use all fixed seeds in the declared order")
    if roles["primary"].get("parameter_scale") != roles["same_scale_ablation"].get(
        "parameter_scale"
    ):
        raise ValueError("primary and same-scale ablation parameter scales differ")
    if protocol.get("visual_tower_lora") is not False:
        raise ValueError("default backbone comparison must freeze visual-tower LoRA")
    gate = protocol.get("gate", {})
    if gate.get("selection") != "stratified_task_grouped_oof_train_only":
        raise ValueError("gate selection must be task-grouped and train-only")
    evaluation = protocol.get("evaluation", {})
    if evaluation.get("baseline") != "same_backbone_direct_vlm":
        raise ValueError("hierarchical policy must compare against the same direct VLM")
    return {
        "schema_version": "activemap-vlm-backbone-matrix-validation-v1",
        "experiment_id": payload.get("experiment_id"),
        "backbone_count": len(backbones),
        "backbone_ids": ids,
        "fixed_seeds": fixed_seeds,
        "primary": roles["primary"]["id"],
        "same_scale_ablation": roles["same_scale_ablation"]["id"],
        "architecture_robustness": roles["architecture_robustness"]["id"],
        "test_assets_read": False,
        "valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("VLM matrix root must be a mapping")
    result = validate(payload)
    text = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
