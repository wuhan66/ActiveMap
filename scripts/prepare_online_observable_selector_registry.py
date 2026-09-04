#!/usr/bin/env python3
"""Freeze online-observable selector checkpoints into a controller registry."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
from typing import Any

import torch
import yaml

from activemap.features import ONLINE_OBSERVABLE_STATE_CONTRACT


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_assignment(value: str) -> tuple[int, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SEED=VALUE")
    seed, payload = value.split("=", 1)
    try:
        return int(seed), payload
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be an integer") from exc


def registry_path(path: Path, storage_root: Path) -> str:
    try:
        return "${STORAGE_ROOT}/" + path.resolve().relative_to(storage_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_online_selector(path: Path) -> tuple[str, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("data_contract") != ONLINE_OBSERVABLE_STATE_CONTRACT:
        raise ValueError(f"selector lacks online-observable contract: {path}")
    stop_margin = checkpoint.get("stop_margin")
    if not isinstance(stop_margin, (int, float)):
        raise ValueError(f"selector lacks calibrated stop margin: {path}")
    return sha256_path(path), float(stop_margin)


def build_registry(
    base_path: Path,
    output_path: Path,
    *,
    storage_root: Path,
    selectors: dict[int, Path],
    component_sources: dict[int, int],
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    if not selectors:
        raise ValueError("at least one selector checkpoint is required")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict) or base.get("test_assets_read") is not False:
        raise ValueError("base registry must be validation-only")
    base_seeds = base.get("seed_artifacts")
    if not isinstance(base_seeds, dict):
        raise ValueError("base registry lacks seed_artifacts")

    frozen_artifacts: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for seed, selector in sorted(selectors.items()):
        if not selector.is_file():
            raise FileNotFoundError(selector)
        source_seed = component_sources.get(seed)
        if source_seed is None:
            raise ValueError(f"selector seed {seed} lacks a component source")
        source = base_seeds.get(str(source_seed))
        if not isinstance(source, dict):
            raise ValueError(f"base registry lacks component source seed {source_seed}")
        selector_hash, stop_margin = load_online_selector(selector)
        artifacts = copy.deepcopy(source)
        artifacts["selector"] = {
            "path": registry_path(selector, storage_root),
            "sha256": selector_hash,
            "stop_margin": stop_margin,
            "data_contract": dict(ONLINE_OBSERVABLE_STATE_CONTRACT),
        }
        frozen_artifacts[str(seed)] = artifacts
        provenance[str(seed)] = {
            "selector_checkpoint": str(selector.resolve()),
            "selector_sha256": selector_hash,
            "component_source_seed": source_seed,
        }

    registry = copy.deepcopy(base)
    registry.update(
        {
            "schema_version": "sn7-online-observable-full-controller-registry-v1",
            "status": "validation_pilot_only",
            "method": "online_observable_step0_recurrent_belief",
            "seeds": sorted(selectors),
            "test_policy": "not_authorized",
            "test_assets_read": False,
            "seed_artifacts": frozen_artifacts,
            "online_state_contract": dict(ONLINE_OBSERVABLE_STATE_CONTRACT),
            "provenance": {
                "base_registry": {
                    "path": str(base_path.resolve()),
                    "sha256": sha256_path(base_path),
                },
                "component_reuse": provenance,
            },
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    return registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_registry", type=Path)
    parser.add_argument("output_registry", type=Path)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument(
        "--selector", action="append", type=parse_assignment, required=True, metavar="SEED=PATH"
    )
    parser.add_argument(
        "--component-source",
        action="append",
        type=parse_assignment,
        required=True,
        metavar="SEED=LEGACY_SEED",
    )
    args = parser.parse_args()
    selector_pairs = dict(args.selector)
    source_pairs = {seed: int(value) for seed, value in args.component_source}
    if len(selector_pairs) != len(args.selector) or len(source_pairs) != len(args.component_source):
        raise ValueError("duplicate selector or component-source assignment")
    build_registry(
        args.base_registry,
        args.output_registry,
        storage_root=args.storage_root,
        selectors={seed: Path(path) for seed, path in selector_pairs.items()},
        component_sources=source_pairs,
    )
    print(args.output_registry.resolve())


if __name__ == "__main__":
    main()
